"""The /api/chat flow. `run()` picks one answer path and yields Server-Sent
Events: "stage" (progress label), "token" (answer text), then "done" (final
answer + sources) or "error".

    document ID in the query  -> answer_document
    router: metadata          -> answer_metadata
    router: out_of_scope      -> answer_out_of_scope
    router: conversational    -> answer_conversational
    otherwise                 -> answer_rag (retrieve, rerank, generate, verify)
"""
import json
import re
import time

from . import config, llm, prompts
from .generation import (
    build_background_block,
    build_context,
    build_rag_system_prompt,
    generate_answer,
    retry_if_uncited,
    route_query,
    summarize_last_answer,
)
from .postprocess import (
    cite_table_source_columns,
    drop_bulk_citations,
    remap_citations,
    strip_artifacts,
    strip_doc_echo,
    strip_unsupported_citations,
)
from .retrieval import (
    expand_and_retrieve,
    fetch_document,
    hybrid_search,
    metadata_query,
    rerank,
    tokenize_terms,
)
from .verification import check_answer_grounded, verify_citations

DOC_ID = re.compile(r'\b(\d{3}-\d{5}-\d{5})\b')


# --- SSE helpers ------------------------------------------------------------------
def sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def stage(label):
    return sse("stage", {"label": label})


def done(answer, sources, query_type, timings=None):
    body = {"answer": answer, "sources": sources, "query_type": query_type}
    if timings:
        body["timings"] = timings
    return sse("done", body)


def _stream_to_client(messages, temperature):
    """Stream model tokens to the client as they arrive; returns the full text
    (use as `text = yield from _stream_to_client(...)`)."""
    full = ""
    for delta in llm.stream(messages, temperature=temperature):
        full += delta
        yield sse("token", {"text": delta})
    return full


def _replay(answer):
    """Animate an already-final answer into the UI.

    A flat 15ms/token made this the most expensive phase of the request: a
    ~800-token research answer spent 12s here against 7s of actual work. The
    whole replay is budgeted instead, so long answers speed up."""
    tokens = re.findall(r'\S+\s*', answer)
    delay = min(config.REPLAY_MAX_DELAY_S, config.REPLAY_BUDGET_S / max(len(tokens), 1))
    for tok in tokens:
        yield sse("token", {"text": tok})
        if delay > 0:
            time.sleep(delay)


def _ms_since(t):
    return int((time.time() - t) * 1000)


def _within_budget(pages, max_chars):
    """Leading pages whose combined text fits max_chars (always at least one).
    Whole documents went to the model uncapped: a 370-page record is far past
    the context window, and every request for it paid for the full text."""
    kept, used = [], 0
    for page in pages:
        used += len(page[0] or "")
        if kept and used > max_chars:
            break
        kept.append(page)
    return kept


def _source_refs(pages):
    return [{"filename": r[1], "page": r[2]} for r in pages]


# --- Entry point ------------------------------------------------------------------
def run(query, history):
    t0, timings = time.time(), {}
    try:
        doc_id = DOC_ID.search(query)
        if doc_id:
            yield from answer_document(query, doc_id.group(1), t0)
            return

        yield stage("Routing...")
        t = time.time()
        route = route_query(query, history)
        timings["router_ms"] = _ms_since(t)
        print(f"[route] type={route.query_type} rewritten={route.rewritten_query!r}")

        if route.query_type == "metadata" and route.metadata_filter:
            if route.metadata_filter.column in config.METADATA_COLUMNS:
                yield from answer_metadata(query, route.metadata_filter, t0, timings)
                return
            route.query_type = "simple"  # unknown column: answer from the pages instead
        if route.query_type == "out_of_scope":
            yield from answer_out_of_scope(t0, timings)
        elif not route.needs_retrieval or route.query_type == "conversational":
            yield from answer_conversational(query, history, t0, timings)
        else:
            yield from answer_rag(query, history, route, t0, timings)
    except Exception as e:
        # Never stream exception text: provider and DB errors embed internal
        # details (org ids, quota usage, DB host/IP/user).
        print(f"Error in /api/chat stream: {e!r}")
        yield sse("error", {"message": "Something went wrong while answering. Please try again."})


# --- Answer paths -----------------------------------------------------------------
def answer_document(query, doc_id, t0):
    """Summarise / answer about one document, given all of its pages."""
    yield stage("Fetching document...")
    filename, pages = fetch_document(doc_id)
    if not pages:
        yield done(f"Document **{filename}** was not found in the archive. Please verify the document ID.", [], "document")
        return
    yield stage("Generating...")
    total = len(pages)
    pages = _within_budget(pages, config.DOC_CONTEXT_CHARS)
    note = "" if len(pages) == total else (
        f"\n\n(Only the first {len(pages)} of {total} pages fit; say so if the answer may be in later pages.)")
    ctx = build_context(pages)
    instructions = prompts.render("document-agent.txt", filename=filename)
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"DOCUMENT PAGES:\n{ctx}{note}\n\nUSER INQUIRY: {query}"},
    ]
    full = yield from _stream_to_client(messages, config.GEN_TEMPERATURE)
    # The done event replaces the streamed text, so a cleaned or retried
    # answer still reaches the client.
    answer = retry_if_uncited(strip_doc_echo(full), query, pages, instructions, ctx)
    yield done(answer, _source_refs(pages), "document", {"total_ms": _ms_since(t0)})


def answer_metadata(query, metadata_filter, t0, timings):
    """Archive-level counts ("how many pages have handwriting?") plus samples."""
    yield stage("Running metadata query...")
    col, val = metadata_filter.column, metadata_filter.value
    total_pages, total_docs, samples = metadata_query(col, val)
    ctx = "\n\n".join([
        f"METADATA QUERY RESULTS for {col} = {val}:",
        f"Total matching pages: {total_pages}",
        f"Total matching documents: {total_docs}",
        "",
        "SAMPLE DOCUMENTS:",
        build_context([(r[0] or "(no content)", r[1], r[2]) for r in samples], cap=500),
    ])
    yield stage("Generating...")
    messages = [{"role": "user", "content": prompts.render("metadata.txt", ctx=ctx, query=query)}]
    full = yield from _stream_to_client(messages, config.GEN_TEMPERATURE)
    yield done(strip_artifacts(full), _source_refs(samples), "metadata", {"total_ms": _ms_since(t0), **timings})


def answer_out_of_scope(t0, timings):
    """Skip retrieval and generation; return the exact refusal line."""
    yield stage("Out of scope")
    yield from _replay(config.REFUSAL_LINE)
    timings["total_ms"] = _ms_since(t0)
    yield done(config.REFUSAL_LINE, [], "out_of_scope", timings)


def answer_conversational(query, history, t0, timings):
    """Greetings and questions about the tool itself; no retrieval."""
    yield stage("Responding...")
    messages = [{"role": "system", "content": prompts.load("conversational.txt")}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-20:]]
    messages.append({"role": "user", "content": query})
    full = yield from _stream_to_client(messages, temperature=0.5)
    yield done(full, [], "conversational", {"total_ms": _ms_since(t0), **timings})


def answer_rag(query, history, route, t0, timings):
    """Retrieve -> rerank -> generate -> grounding check (-> expand and
    regenerate) -> citation verification -> stream the final answer.

    The answer is generated silently and only streamed once it has passed
    the checks, so the user never sees text that later changes."""
    search_terms = [t for t in route.search_terms if t and t.strip()] or [query]
    context_limit = config.CONTEXT_LIMIT_SIMPLE if route.query_type == "simple" else config.CONTEXT_LIMIT_RESEARCH
    print(f"[rag] type={route.query_type} terms={search_terms}")

    # FTS gets ONLY the router's keywords: the full question pollutes
    # ts_rank_cd with common words ("role", "played", "files") that match
    # admin pages. The vector leg gets the full rewritten question.
    yield stage("Retrieving documents...")
    t = time.time()
    ts_input = " ".join(search_terms).strip()
    candidates = hybrid_search(ts_input, route.rewritten_query, tokenize_terms(ts_input), or_terms=search_terms)
    timings["retrieve_ms"] = _ms_since(t)
    print(f"[rag] retrieved={len(candidates)} (hybrid)")

    yield stage("Reranking...")
    t = time.time()
    picked = rerank(candidates, route.rewritten_query, context_limit, search_terms)
    timings["rerank_ms"] = _ms_since(t)

    system_prompt = build_rag_system_prompt(route.query_type)
    prior_summary = summarize_last_answer(history)
    background = build_background_block(search_terms)

    yield stage("Generating...")
    t = time.time()
    answer = generate_answer(query, picked, system_prompt, prior_summary, route.query_type, background)
    timings["generate_ms"] = _ms_since(t)

    yield stage("Checking answer...")
    t = time.time()
    grounded, reason = check_answer_grounded(answer, picked, route.rewritten_query, route.query_type)
    timings["ground_ms"] = _ms_since(t)
    print(f"[rag] grounding {'PASSED' if grounded else 'FAILED'}: {reason}")

    if not grounded:
        # One expansion round, no second grounding check: the citation
        # verifier below cleans up the regenerated answer instead.
        yield stage("Expanding search...")
        t = time.time()
        candidates = expand_and_retrieve(route.rewritten_query, reason, candidates)
        timings["expand_ms"] = _ms_since(t)
        yield stage("Reranking expanded results...")
        picked = rerank(candidates, route.rewritten_query, context_limit, search_terms)
        yield stage("Regenerating with new sources...")
        answer = generate_answer(query, picked, system_prompt, prior_summary, route.query_type, background)

    yield stage("Verifying citations...")
    t = time.time()
    answer = drop_bulk_citations(cite_table_source_columns(answer, len(picked)))
    unsupported = verify_citations(answer, picked)
    timings["cite_verify_ms"] = _ms_since(t)
    if unsupported:
        print(f"[rag] unsupported citations (unit, source, re-pointed to): {unsupported}")
        answer = strip_unsupported_citations(answer, unsupported)

    # remap_citations renumbers [N] to 1..N in citation order, so the source
    # panel must follow that order or every link points at the wrong page.
    # Uncited pages are appended after the cited ones.
    all_sources = _source_refs(picked)
    answer, cited = remap_citations(answer, all_sources)
    sources = cited + [s for s in all_sources if s not in cited]

    yield stage("Streaming answer...")
    yield from _replay(answer)
    timings["total_ms"] = _ms_since(t0)
    print(f"[timing] {timings}")
    yield done(answer, sources, route.query_type, timings)
