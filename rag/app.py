import os
import re
import json
import time
import threading
from contextlib import contextmanager

import psycopg2
from flask import Flask, request, jsonify, redirect, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='frontend/dist', static_url_path='/')
CORS(app)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NARA_BASE_URL = "https://storage.googleapis.com/jfkweb-prod"

# Main LLM client: Groq + llama for everything.
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
    MODEL = "llama-3.3-70b-versatile"
    LLM_PROVIDER = "groq"
    print(f"LLM provider: Groq, model={MODEL}")
else:
    print("WARNING: GROQ_API_KEY not found; /api/chat will 500.")
    client = None
    MODEL = "llama-3.3-70b-versatile"
    LLM_PROVIDER = "none"

# Judge/rerank/router/expansion/citation-verify calls always go to Groq + llama.
# These are cheap, high-volume utility calls; using the premium MODEL for them
# balloons cost without measurable quality gain.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
if GROQ_API_KEY:
    judge_client = Groq(api_key=GROQ_API_KEY)
    print(f"Judge/rerank provider: Groq, model={JUDGE_MODEL}")
else:
    # No Groq key — fall back to the main client so judges still work (at main-model cost).
    judge_client = client
    JUDGE_MODEL = MODEL
    print(f"Judge/rerank fallback: using main client ({MODEL}) — set GROQ_API_KEY for cheap judges")


# Embedding client — used for hybrid retrieval (FTS ∪ vector).
# Same model used to backfill jfk_pages.embedding; query-side must match.
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 512  # Matryoshka-truncated; stored on-disk as halfvec(512) for ~6x disk savings.
embed_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if not embed_client:
    print("WARNING: OPENAI_API_KEY not set; hybrid retrieval will fall back to FTS-only.")


def token_limit_kwargs(limit):
    """OpenAI's GPT-5 family renamed `max_tokens` → `max_completion_tokens`.
    Groq still uses `max_tokens`. Branch so the same cap works everywhere."""
    if LLM_PROVIDER == "openai":
        return {"max_completion_tokens": limit}
    return {"max_tokens": limit}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_prompts_local = os.path.join(os.path.dirname(__file__), 'prompts')
_prompts_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
PROMPTS_DIR = _prompts_local if os.path.isdir(_prompts_local) else _prompts_root


def load_prompt(filename, fallback=""):
    """Production path: read prompts only from rag/prompts/<filename>. The
    old `optimized/` override slot has been retired — optimizer outputs now
    live in rag/prompts/archive/optimized/ for reference and are promoted to
    live by copying over the base file, so production never reads an
    unreviewed override."""
    base_path = os.path.join(PROMPTS_DIR, filename)
    if os.path.exists(base_path):
        with open(base_path) as f:
            return f.read()
    return fallback


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def ensure_fts_index():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_jfk_pages_content_fts
            ON jfk_pages USING GIN (to_tsvector('english', content))
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("FTS index ready.")
    except Exception as e:
        print(f"FTS index creation skipped: {e}")


if DATABASE_URL:
    ensure_fts_index()


@contextmanager
def db_cursor():
    """Short-lived connection. Held ONLY for the duration of a DB read.
    Prevents holding idle connections across multi-second LLM roundtrips."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        yield cur
        cur.close()
    finally:
        conn.close()


_stats_cache = {"ts": 0, "total_p": 0, "hw_p": 0, "stamp_p": 0, "redact_p": 0}
_STATS_TTL = 300
_stats_lock = threading.Lock()


def get_archive_stats():
    """Cached archive-level counts (5-min TTL). Thread-safe refresh."""
    now = time.time()
    if now - _stats_cache["ts"] < _STATS_TTL and _stats_cache["ts"] > 0:
        return dict(_stats_cache)
    with _stats_lock:
        if now - _stats_cache["ts"] < _STATS_TTL and _stats_cache["ts"] > 0:
            return dict(_stats_cache)
        with db_cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*),
                    COUNT(*) FILTER (WHERE includes_handwriting = true),
                    COUNT(*) FILTER (WHERE has_stamps = true),
                    COUNT(*) FILTER (WHERE has_redactions = true)
                FROM jfk_pages
            """)
            row = cur.fetchone()
            _stats_cache.update({
                "ts": now,
                "total_p": row[0],
                "hw_p": row[1],
                "stamp_p": row[2],
                "redact_p": row[3],
            })
    return dict(_stats_cache)


# Minimal English stopword set for expansion-term filtering.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "his", "her", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "they", "this", "to", "was", "were", "will",
    "who", "whom", "with", "would", "you", "your", "i", "me", "my", "we",
    "us", "our", "what", "when", "where", "why", "how", "did", "do", "does",
    "about", "but", "not", "no", "yes", "s", "t", "re", "d", "ll", "m",
}


def _tokenize_terms(text):
    """Split query text into content terms for ILIKE fallback. Strips
    stopwords/punctuation so naive OR queries don't explode into noise."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
    terms = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 2]
    return terms[:6]  # cap to keep fallback query tractable


def fts_search(ts_input, ilike_terms=None):
    """FTS leg of hybrid retrieval. Strong on proper nouns and rare terms;
    weak on semantic/paraphrase match. ILIKE fallback when tsquery returns nothing."""
    ilike_terms = ilike_terms or _tokenize_terms(ts_input)
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT content, filename, page_number
            FROM (
                SELECT DISTINCT ON (left(content, 200)) content, filename, page_number,
                    ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s), 2) AS rank_score
                FROM jfk_pages
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            ) sub
            ORDER BY rank_score DESC, length(content) ASC
            LIMIT 30
            """,
            [ts_input, ts_input],
        )
        rows = cur.fetchall()
        if not rows and ilike_terms:
            where_clauses = [f"content ILIKE %s" for _ in ilike_terms]
            cur.execute(
                f"""
                SELECT content, filename, page_number
                FROM (
                    SELECT DISTINCT ON (left(content, 200)) content, filename, page_number
                    FROM jfk_pages
                    WHERE ({' OR '.join(where_clauses)})
                ) sub
                ORDER BY length(content) DESC
                LIMIT 30
                """,
                [f"%{t}%" for t in ilike_terms],
            )
            rows = cur.fetchall()
    return rows


def _embed_query(text):
    """Embed a single query string. Returns list[float] or None on failure."""
    if not embed_client or not text or not text.strip():
        return None
    try:
        resp = embed_client.embeddings.create(
            model=EMBED_MODEL,
            input=text[:8000],
            dimensions=EMBED_DIM,
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"[rag] embed failed: {e}")
        return None


def vector_search(query_text, limit=30):
    """Vector leg of hybrid retrieval. Strong on semantic/paraphrase and
    summary-style content; weak on exact name matching."""
    vec = _embed_query(query_text)
    if vec is None:
        return []
    vec_lit = "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
    with db_cursor() as cur:
        cur.execute(
            """
            SELECT content, filename, page_number
            FROM jfk_pages
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::halfvec
            LIMIT %s
            """,
            [vec_lit, limit],
        )
        return cur.fetchall()


def hybrid_search(ts_input, semantic_query, ilike_terms=None):
    """Union of FTS + vector candidates, deduped on (filename, page_number).
    FTS gets keyword-only input; vector gets the full rewritten question —
    each leg is fed what it's best at."""
    fts_rows = fts_search(ts_input, ilike_terms)
    vec_rows = vector_search(semantic_query, limit=30)

    merged, seen = [], set()
    # Interleave FTS-first so exact matches aren't drowned by semantic neighbors.
    for rows in (fts_rows, vec_rows):
        for r in rows:
            key = (r[1], r[2])
            if key in seen:
                continue
            # Also dedupe near-identical content (same first 200 chars).
            content_key = r[0][:200].strip()
            if content_key in seen:
                continue
            seen.add(key)
            seen.add(content_key)
            merged.append(r)
    return merged


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------
def sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def final_event(answer, sources, query_type, timings=None):
    """Produce a final `done` SSE event. Answer sent whole — caller decides
    whether tokens were streamed incrementally beforehand."""
    body = {"answer": answer, "sources": sources, "query_type": query_type}
    if timings:
        body["timings"] = timings
    return sse("done", body)


# ---------------------------------------------------------------------------
# Text post-processing
# ---------------------------------------------------------------------------
def strip_artifacts(text):
    text = re.sub(r'\$\\boxed\{([^}]*)\}\$', r'\1', text)
    text = re.sub(r'(?m)^.*The final answer is:?.*$', '', text)
    text = re.sub(r'(?m)^#+?\s*Step \d+:.*$', '', text)
    # Refusal line should never carry citations — it claims no facts.
    text = re.sub(
        r'(The retrieved documents do not contain sufficient information to answer this query\.?)(\s*\[\d+\])+',
        r'\1',
        text,
    )
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def strip_doc_echo(text):
    text = re.sub(
        r'^\s*(?:#+\s*)?(?:User\s*Inquiry|USER\s*INQUIRY|The user (?:has requested|is asking)|Based on your question|The provided document pages?)[^\n]*\n+',
        '',
        text,
        flags=re.IGNORECASE,
    )
    return strip_artifacts(text)


def remap_citations(answer_text, all_sources):
    """Keep only actually-cited sources and renumber them 1..N."""
    cited_nums = sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', answer_text)))
    if not cited_nums:
        return answer_text, all_sources
    new_sources, remap = [], {}
    for new_idx, old_num in enumerate(cited_nums, 1):
        old_idx = old_num - 1
        if 0 <= old_idx < len(all_sources):
            new_sources.append(all_sources[old_idx])
            remap[old_num] = new_idx
    for old_num in sorted(remap.keys(), reverse=True):
        answer_text = answer_text.replace(f'[{old_num}]', f'[__CITE_{remap[old_num]}__]')
    for new_num in remap.values():
        answer_text = answer_text.replace(f'[__CITE_{new_num}__]', f'[{new_num}]')
    return answer_text, new_sources


# ---------------------------------------------------------------------------
# Router / history
# ---------------------------------------------------------------------------
def build_history_context(history, user_truncate=800, assistant_truncate=600):
    """Serialize recent turns for router. Gives more assistant content than
    before so 'tell me more' references resolve meaningfully — now uses a
    head+tail slice when an answer is long."""
    if not history:
        return ""
    recent = history[-6:]
    lines = []
    for msg in recent:
        role = msg.get('role', 'user')
        if role not in ('user', 'assistant'):
            continue
        content = msg.get('content', '') or ''
        cap = assistant_truncate if role == 'assistant' else user_truncate
        if len(content) > cap:
            # head + tail — tail often has the concrete sources/details the
            # follow-up is about to reference.
            half = cap // 2
            content = content[:half] + ' ... ' + content[-half:]
        lines.append(f"{role}: {content}")
    return "\n\nConversation history (for context):\n" + "\n".join(lines)


def summarize_last_answer(history, char_cap=900):
    """Short summary of the most recent assistant answer, for the RAG agent
    so follow-ups like 'tell me more' can add information without repeating
    what was already said."""
    for msg in reversed(history):
        if msg.get('role') == 'assistant':
            content = (msg.get('content') or '').strip()
            if not content:
                return ""
            if len(content) <= char_cap:
                return content
            half = char_cap // 2
            return content[:half] + ' ... ' + content[-half:]
    return ""


def route_query(query, history):
    """Router agent: classifies query type, rewrites pronouns, extracts keywords."""
    analysis_prompt = load_prompt('router.txt').replace('{query}', query) + build_history_context(history)
    try:
        res = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        return {
            "search_terms": data.get('keywords', query.split()),
            "query_type": data.get('type', 'research'),
            "needs_retrieval": data.get('needs_retrieval', True),
            "metadata_filter": data.get('metadata_filter', None),
            "rewritten_query": data.get('rewritten_query', query) or query,
        }
    except Exception as e:
        print(f"Router failed: {e}")
        return {
            "search_terms": query.split(),
            "query_type": 'research',
            "needs_retrieval": True,
            "metadata_filter": None,
            "rewritten_query": query,
        }


# ---------------------------------------------------------------------------
# RAG pipeline helpers
# ---------------------------------------------------------------------------
_WS_RE = re.compile(r'\s+')


def _normalize_ws(s):
    # OCR splits phrases across newlines (e.g. "Robert\nFulton"), so any
    # exact-substring search on raw content under-counts hits. Collapse
    # whitespace before searching.
    return _WS_RE.sub(' ', s)


def _rerank_snippet(content, terms, width=400):
    # Center the snippet on the first query-term hit so the judge sees the
    # match in context. Slicing from position 0 hides the match whenever the
    # page starts with a routing header (common in JFK records).
    norm = _normalize_ws(content).lower()
    hits = [norm.find(t.lower()) for t in terms if t and t.lower() in norm]
    if not hits:
        return content[:width]
    # Hit positions are in the normalized string; that's fine for windowing —
    # serve the slice from normalized text so the judge isn't fighting OCR
    # newlines either.
    norm_full = _normalize_ws(content)
    start = max(0, min(hits) - 80)
    return norm_full[start:start + width]


def _dedupe_picks(candidates, picked_indices, context_limit, fill_pool):
    # Drop near-duplicate pages (same first 200 chars after whitespace
    # normalization) so three copies of the same memo don't crowd out a
    # second, distinct passage. Backfill from the next-ranked candidates.
    seen_keys = set()
    out, out_idx = [], set()
    def key(r):
        return _normalize_ws(r[0])[:200].strip().lower()
    for i in picked_indices:
        if i in out_idx:
            continue
        k = key(candidates[i])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        out_idx.add(i)
        out.append(candidates[i])
        if len(out) >= context_limit:
            return out
    for i in fill_pool:
        if len(out) >= context_limit:
            break
        if i in out_idx:
            continue
        k = key(candidates[i])
        if k in seen_keys:
            continue
        seen_keys.add(k)
        out_idx.add(i)
        out.append(candidates[i])
    return out


def rerank(candidates, rewritten_query, context_limit, search_terms=None):
    if len(candidates) <= context_limit:
        return candidates[:context_limit]
    terms = [t for t in (search_terms or rewritten_query.split()) if t]
    snippets = [
        f"[{idx}] {r[1]}, Page {r[2]}: {_rerank_snippet(r[0], terms).replace(chr(10), ' ').strip()}"
        for idx, r in enumerate(candidates)
    ]
    prompt = (
        load_prompt('reranker.txt')
        .replace('{context_limit}', str(context_limit))
        .replace('{rewritten_query}', rewritten_query)
        .replace('{snippets}', '\n'.join(snippets))
    )
    try:
        res = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = json.loads(res.choices[0].message.content)
        indices = list(raw.values())[0] if isinstance(raw, dict) else raw
        valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
        fill_pool = [i for i in range(len(candidates)) if i not in set(valid)]
        return _dedupe_picks(candidates, valid, context_limit, fill_pool)
    except Exception as e:
        print(f"Rerank failed, using FTS order: {e}")
        return _dedupe_picks(candidates, list(range(len(candidates))), context_limit, [])


def build_context(picked):
    if not picked:
        return "NO SEARCH RESULTS FOUND."
    parts = [f"[{i}] Source: {r[1]}, Page {r[2]}\n{r[0]}" for i, r in enumerate(picked, 1)]
    return "\n\n".join(parts)


def build_rag_system_prompt(query_type, stats):
    instructions = load_prompt('rag-simple.txt' if query_type == 'simple' else 'rag-research.txt')
    suffix = (
        load_prompt('rag-system-suffix.txt')
        .replace('{total_p}', f"{stats['total_p']:,}")
        .replace('{hw_p}', f"{stats['hw_p']:,}")
        .replace('{stamp_p}', f"{stats['stamp_p']:,}")
        .replace('{redact_p}', f"{stats['redact_p']:,}")
    )
    return f"{instructions}\n{suffix}\n"


def build_rag_user_prompt(query, ctx, prior_answer_summary, query_type="research"):
    prior_block = ""
    if prior_answer_summary:
        prior_block = (
            "PREVIOUS ASSISTANT ANSWER (for reference only — do NOT repeat, add NEW information):\n"
            f"\"\"\"\n{prior_answer_summary}\n\"\"\"\n\n"
        )

    # Format reminder goes LAST because small models weight the tail of the
    # prompt far more than the head; the long system prompt's format rules
    # are otherwise forgotten by the time llama starts generating.
    format_reminder = load_prompt(
        'rag-format-simple.txt' if query_type == 'simple' else 'rag-format-research.txt'
    )

    return (
        load_prompt('rag-user-template.txt')
        .replace('{prior_block}', prior_block)
        .replace('{ctx}', ctx)
        .replace('{query}', query)
        .replace('{format_reminder}', format_reminder)
    )


def generate_answer_stream(query, picked, system_prompt, prior_answer_summary, query_type="research"):
    """Stream the generation. Yields (kind, payload) tuples:
    ("token", text) for incremental text, ("done", full_text) at end."""
    ctx = build_context(picked)
    user_prompt = build_rag_user_prompt(query, ctx, prior_answer_summary, query_type)
    full_text = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full_text += delta
                yield ("token", delta)
    except Exception as e:
        print(f"Streaming failed, falling back to non-streaming: {e}")
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        full_text = res.choices[0].message.content
        yield ("token", full_text)
    yield ("done", full_text)


def generate_answer_nonstream(query, picked, system_prompt, prior_answer_summary, query_type="research"):
    ctx = build_context(picked)
    user_prompt = build_rag_user_prompt(query, ctx, prior_answer_summary, query_type)
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    text = strip_artifacts(res.choices[0].message.content)
    if picked and not re.search(r'\[\d+\]', text):
        retry_prompt = (
            load_prompt('rag-retry-nocite.txt')
            .replace('{ctx}', ctx)
            .replace('{query}', query)
        )
        retry = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        retry_text = strip_artifacts(retry.choices[0].message.content)
        if re.search(r'\[\d+\]', retry_text):
            text = retry_text
    return text


def check_answer_grounded(answer, picked, rewritten_query, query_type):
    """Grounded = (a) addresses the correct subject AND (b) claims are supported
    by the retrieved sources. Judge sees both answer and sources, so it can
    spot hallucinations/subject drift, not just refusals."""
    if query_type not in ("simple", "research"):
        return True, ""
    sources_block = build_context(picked)[:4000]
    judge_prompt = (
        load_prompt('grounding-judge.txt')
        .replace('{rewritten_query}', rewritten_query)
        .replace('{sources_block}', sources_block)
        .replace('{answer}', answer[:3500])
    )
    try:
        res = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        return bool(data.get("grounded", True)), data.get("reason", "")
    except Exception as e:
        print(f"Grounding check skipped: {e}")
        return True, ""


def verify_citations(answer, picked):
    """Per-citation check: does source [N] actually support the sentence that
    cites it? Returns a list of unsupported citation numbers. Cheap safeguard
    against the LLM attaching arbitrary [N] to fabricated claims."""
    if not picked or not re.search(r'\[\d+\]', answer):
        return []
    # Give the verifier every source, each capped so no single doc dominates.
    # A global char cap would silently hide later sources and falsely mark
    # their citations unsupported.
    per_source_cap = 3500
    parts = [
        f"[{i}] Source: {r[1]}, Page {r[2]}\n{r[0][:per_source_cap]}"
        for i, r in enumerate(picked, 1)
    ]
    sources_block = "\n\n".join(parts)
    prompt = (
        load_prompt('citation-verify.txt')
        .replace('{sources_block}', sources_block)
        .replace('{answer}', answer[:3500])
    )
    try:
        res = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        bad = data.get("unsupported", [])
        return [int(x) for x in bad if isinstance(x, (int, str)) and str(x).isdigit()]
    except Exception as e:
        print(f"Citation verification skipped: {e}")
        return []


def expand_and_retrieve(rewritten_query, reason, seed_results):
    """Generate 3 alternative phrasings, retrieve, merge with seed_results."""
    prompt = (
        load_prompt('expansion.txt')
        .replace('{rewritten_query}', rewritten_query)
        .replace('{reason}', reason)
    )
    try:
        res = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        expanded = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
    except Exception as e:
        print(f"Expansion failed: {e}")
        return seed_results
    if not expanded:
        return seed_results
    print(f"Expansion queries: {expanded}")
    merged = list(seed_results)
    seen = {r[0][:200].strip() for r in merged}
    for eq in expanded:
        # Expansion queries are LLM-generated full sentences — feed them to
        # both legs so we gain semantic recall, not just keyword matches.
        for r in hybrid_search(eq, eq, _tokenize_terms(eq)):
            key = r[0][:200].strip()
            if key not in seen:
                merged.append(r)
                seen.add(key)
    return merged


# ---------------------------------------------------------------------------
# Main chat endpoint — SSE streaming with stage events
# ---------------------------------------------------------------------------
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    query = data.get('query')
    history = data.get('history', [])

    if not query:
        return jsonify({"error": "No query provided"}), 400
    if not client:
        return jsonify({"error": "LLM client not configured"}), 500

    def generate():
        timings = {}
        t0 = time.time()

        def stage(label):
            yield_val = sse("stage", {"label": label})
            return yield_val

        try:
            # -- Document ID shortcut --------------------------------------
            doc_id_match = re.search(r'\b(\d{3}-\d{5}-\d{5})\b', query)
            if doc_id_match:
                yield sse("stage", {"label": "Fetching document..."})
                doc_id = doc_id_match.group(1)
                filename = f"{doc_id}.pdf"
                with db_cursor() as cur:
                    cur.execute(
                        "SELECT content, filename, page_number FROM jfk_pages WHERE filename = %s ORDER BY page_number",
                        (filename,),
                    )
                    doc_results = cur.fetchall()
                if not doc_results:
                    yield final_event(
                        f"Document **{filename}** was not found in the archive. Please verify the document ID.",
                        [], "document")
                    return
                yield sse("stage", {"label": "Generating..."})
                parts = [f"[{i}] Source: {r[1]}, Page {r[2]}\n{r[0]}" for i, r in enumerate(doc_results, 1)]
                ctx = "\n\n".join(parts)
                instructions = load_prompt('document-agent.txt').replace('{filename}', filename)
                full = ""
                try:
                    stream = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": f"DOCUMENT PAGES:\n{ctx}\n\nUSER INQUIRY: {query}"},
                        ],
                        temperature=0.3,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            full += delta
                            yield sse("token", {"text": delta})
                except Exception as e:
                    print(f"Doc streaming failed, using non-stream: {e}")
                    res = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": instructions},
                            {"role": "user", "content": f"DOCUMENT PAGES:\n{ctx}\n\nUSER INQUIRY: {query}"},
                        ],
                        temperature=0.3,
                    )
                    full = res.choices[0].message.content
                    yield sse("token", {"text": full})
                cleaned = strip_doc_echo(full)
                # Replay cleaned answer so the frontend shows final cleaned text
                # (only if cleaning stripped something visible — otherwise skip)
                yield final_event(
                    cleaned,
                    [{"filename": r[1], "page": r[2]} for r in doc_results],
                    "document",
                    {"total_ms": int((time.time() - t0) * 1000)},
                )
                return

            # -- Router -----------------------------------------------------
            yield sse("stage", {"label": "Routing..."})
            t_router = time.time()
            routed = route_query(query, history)
            timings["router_ms"] = int((time.time() - t_router) * 1000)
            query_type = routed["query_type"]
            rewritten_query = routed["rewritten_query"]
            print(f"[route] type={query_type} rewritten={rewritten_query!r}")

            # -- Metadata shortcut -----------------------------------------
            if query_type == "metadata" and routed["metadata_filter"]:
                yield sse("stage", {"label": "Running metadata query..."})
                allowed_columns = {
                    'has_redactions', 'includes_handwriting', 'has_stamps',
                    'has_tables', 'has_forms', 'is_typewritten', 'document_type',
                }
                mf = routed["metadata_filter"]
                col, val = mf.get('column', ''), mf.get('value', True)
                if col in allowed_columns:
                    with db_cursor() as cur:
                        if isinstance(val, bool):
                            cur.execute(f"SELECT COUNT(*) FROM jfk_pages WHERE {col} = %s", (val,))
                            total_matching = cur.fetchone()[0]
                            cur.execute(f"SELECT COUNT(DISTINCT file_id) FROM jfk_pages WHERE {col} = %s", (val,))
                            docs_matching = cur.fetchone()[0]
                            cur.execute(f"""
                                SELECT DISTINCT ON (file_id) content, filename, page_number, file_id
                                FROM jfk_pages WHERE {col} = %s AND content IS NOT NULL
                                ORDER BY file_id, page_number LIMIT 10
                            """, (val,))
                        else:
                            cur.execute(f"SELECT COUNT(*) FROM jfk_pages WHERE {col} ILIKE %s", (f"%{val}%",))
                            total_matching = cur.fetchone()[0]
                            cur.execute(f"SELECT COUNT(DISTINCT file_id) FROM jfk_pages WHERE {col} ILIKE %s", (f"%{val}%",))
                            docs_matching = cur.fetchone()[0]
                            cur.execute(f"""
                                SELECT DISTINCT ON (file_id) content, filename, page_number, file_id
                                FROM jfk_pages WHERE {col} ILIKE %s AND content IS NOT NULL
                                ORDER BY file_id, page_number LIMIT 10
                            """, (f"%{val}%",))
                        samples = cur.fetchall()

                    ctx_parts = [
                        f"METADATA QUERY RESULTS for {col} = {val}:",
                        f"Total matching pages: {total_matching}",
                        f"Total matching documents: {docs_matching}",
                        "",
                        "SAMPLE DOCUMENTS:",
                    ]
                    sources_list = []
                    for idx, r in enumerate(samples, 1):
                        snippet = r[0][:500] if r[0] else "(no content)"
                        ctx_parts.append(f"[{idx}] Source: {r[1]}, Page {r[2]}\n{snippet}")
                        sources_list.append({"filename": r[1], "page": r[2]})
                    meta_prompt = (
                        load_prompt('metadata.txt')
                        .replace('{ctx}', "\n\n".join(ctx_parts))
                        .replace('{query}', query)
                    )
                    yield sse("stage", {"label": "Generating..."})
                    full = ""
                    try:
                        stream = client.chat.completions.create(
                            model=MODEL,
                            messages=[{"role": "user", "content": meta_prompt}],
                            temperature=0.3,
                            stream=True,
                        )
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content if chunk.choices else None
                            if delta:
                                full += delta
                                yield sse("token", {"text": delta})
                    except Exception as e:
                        print(f"Meta streaming failed: {e}")
                        res = client.chat.completions.create(
                            model=MODEL,
                            messages=[{"role": "user", "content": meta_prompt}],
                            temperature=0.3,
                        )
                        full = res.choices[0].message.content
                        yield sse("token", {"text": full})
                    yield final_event(strip_artifacts(full), sources_list, "metadata",
                                      {"total_ms": int((time.time() - t0) * 1000), **timings})
                    return
                # Invalid column — fall through to search
                query_type = "simple"

            # -- Out-of-scope ---------------------------------------------
            # Short-circuit: skip retrieval + generation entirely. Emit the
            # exact refusal line the eval/citation-verification contract
            # expects, streamed token-by-token so the frontend animates it
            # the same way as a real answer.
            if query_type == "out_of_scope":
                refusal = "The retrieved documents do not contain sufficient information to answer this query."
                yield sse("stage", {"label": "Out of scope"})
                for tok in re.findall(r'\S+\s*', refusal):
                    yield sse("token", {"text": tok})
                    time.sleep(0.015)
                timings["total_ms"] = int((time.time() - t0) * 1000)
                yield final_event(refusal, [], "out_of_scope", timings)
                return

            # -- Conversational -------------------------------------------
            if not routed["needs_retrieval"] or query_type == "conversational":
                yield sse("stage", {"label": "Responding..."})
                conv_messages = [{"role": "system", "content": load_prompt('conversational.txt')}]
                for msg in history[-20:]:
                    role = msg.get('role', 'user')
                    if role in ('user', 'assistant'):
                        conv_messages.append({"role": role, "content": msg['content']})
                conv_messages.append({"role": "user", "content": query})
                full = ""
                try:
                    stream = client.chat.completions.create(
                        model=MODEL,
                        messages=conv_messages,
                        temperature=0.5,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            full += delta
                            yield sse("token", {"text": delta})
                except Exception as e:
                    print(f"Conversational streaming failed: {e}")
                    res = client.chat.completions.create(
                        model=MODEL, messages=conv_messages, temperature=0.5,
                    )
                    full = res.choices[0].message.content
                    yield sse("token", {"text": full})
                yield final_event(full, [], "conversational",
                                  {"total_ms": int((time.time() - t0) * 1000), **timings})
                return

            # -- Search / RAG ---------------------------------------------
            search_terms = [t for t in routed["search_terms"] if t and t.strip()]
            if not search_terms:
                search_terms = [query]
            context_limit = 15 if query_type == "simple" else 20
            print(f"[rag] type={query_type} terms={search_terms}")

            stats = get_archive_stats()

            yield sse("stage", {"label": "Retrieving documents..."})
            t_retr = time.time()
            # Use ONLY the router-extracted search terms for FTS. Feeding the full
            # rewritten question pollutes ts_rank_cd with common words ("role",
            # "played", "files") that match admin/policy docs and bury the
            # actually-relevant pages. Fall back to tokenized rewritten_query only
            # if the router returned no terms.
            if search_terms:
                ts_input = ' '.join(search_terms).strip()
            else:
                ts_input = rewritten_query
            # Hybrid: FTS on keywords (proper-noun precision) ∪ vector on the
            # rewritten question (semantic/summary recall). Rerank picks from
            # the union.
            unique_results = hybrid_search(ts_input, rewritten_query, _tokenize_terms(ts_input))
            timings["retrieve_ms"] = int((time.time() - t_retr) * 1000)
            print(f"[rag] retrieved={len(unique_results)} (hybrid)")

            yield sse("stage", {"label": "Reranking..."})
            t_rr = time.time()
            final_results = rerank(unique_results, rewritten_query, context_limit, search_terms)
            timings["rerank_ms"] = int((time.time() - t_rr) * 1000)

            system_prompt = build_rag_system_prompt(query_type, stats)
            prior_summary = summarize_last_answer(history)

            # First generation — collected silently (not streamed to client yet).
            # We stream only the final verified answer after grounding + citation checks.
            yield sse("stage", {"label": "Generating..."})
            t_gen = time.time()
            full_text = ""
            for kind, payload in generate_answer_stream(query, final_results, system_prompt, prior_summary, query_type):
                if kind == "token":
                    full_text += payload
                else:
                    full_text = payload
            timings["generate_ms"] = int((time.time() - t_gen) * 1000)
            answer_text = strip_artifacts(full_text)

            # Post-gen grounding check (sees sources)
            yield sse("stage", {"label": "Checking answer..."})
            t_g = time.time()
            grounded, ground_reason = check_answer_grounded(answer_text, final_results, rewritten_query, query_type)
            timings["ground_ms"] = int((time.time() - t_g) * 1000)
            print(f"[rag] grounding {'PASSED' if grounded else 'FAILED'}: {ground_reason}")

            retried = False
            if not grounded:
                retried = True
                print(f"[rag] expanding after failed grounding.")
                yield sse("stage", {"label": "Expanding search..."})
                t_exp = time.time()
                unique_results = expand_and_retrieve(rewritten_query, ground_reason, unique_results)
                timings["expand_ms"] = int((time.time() - t_exp) * 1000)

                yield sse("stage", {"label": "Reranking expanded results..."})
                final_results = rerank(unique_results, rewritten_query, context_limit, search_terms)

                # Second generation — also silent; we still stream only the final answer.
                # No second grounding check: we trust the expanded-regen output and let
                # verify_citations strip bad [N] markers below instead of refusing outright.
                yield sse("stage", {"label": "Regenerating with new sources..."})
                answer_text = generate_answer_nonstream(query, final_results, system_prompt, prior_summary, query_type)

            # Citation verification — strip bad citations (or regenerate if many bad)
            yield sse("stage", {"label": "Verifying citations..."})
            t_cv = time.time()
            unsupported = verify_citations(answer_text, final_results)
            timings["cite_verify_ms"] = int((time.time() - t_cv) * 1000)
            if unsupported:
                print(f"[rag] unsupported citations: {unsupported}")
                # Strip unsupported [N] markers from the text
                for n in sorted(set(unsupported), reverse=True):
                    answer_text = re.sub(rf'\s*\[{n}\]', '', answer_text)

            all_sources = [{"filename": r[1], "page": r[2]} for r in final_results]
            answer_text, _ = remap_citations(answer_text, all_sources)
            # Always surface every reranked source in the panel, even if the
            # generator cited only a subset (or none survived verification).
            sources_out = all_sources

            # Stream the final, verified answer to the frontend. Small delay per
            # chunk so the UI animates in rather than arriving as one block.
            yield sse("stage", {"label": "Streaming answer..."})
            tokens_out = re.findall(r'\S+\s*', answer_text)
            for tok in tokens_out:
                yield sse("token", {"text": tok})
                time.sleep(0.015)

            timings["total_ms"] = int((time.time() - t0) * 1000)
            print(f"[timing] {timings}")
            yield final_event(answer_text, sources_out, query_type, timings)

        except Exception as e:
            print(f"Error in /api/chat stream: {e}")
            yield sse("error", {"message": str(e)})

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# ---------------------------------------------------------------------------
# Stats / analyze / pdf routes
# ---------------------------------------------------------------------------
@app.route('/api/stats', methods=['GET'])
def stats_route():
    try:
        with db_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM jfk_pages")
            total_pages = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT file_id) FROM jfk_pages")
            total_docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE content IS NOT NULL AND length(trim(content)) > 0")
            pages_with_content = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT file_id) FROM jfk_pages WHERE content IS NOT NULL AND length(trim(content)) > 0")
            docs_with_content = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE includes_handwriting = true")
            handwritten_pages = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_stamps = true")
            stamped_pages = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_redactions = true")
            redacted_pages = cur.fetchone()[0]
            cur.execute("SELECT document_type, COUNT(*) as count FROM jfk_pages GROUP BY document_type ORDER BY count DESC LIMIT 5")
            doc_types = cur.fetchall()
        page_pct = (pages_with_content / total_pages * 100) if total_pages > 0 else 0
        doc_pct = (docs_with_content / total_docs * 100) if total_docs > 0 else 0
        return jsonify({
            "total_pages": total_pages, "total_docs": total_docs,
            "pages_with_content": pages_with_content, "docs_with_content": docs_with_content,
            "page_content_pct": round(page_pct, 1), "doc_content_pct": round(doc_pct, 1),
            "handwritten_pages": handwritten_pages, "stamped_pages": stamped_pages,
            "redacted_pages": redacted_pages,
            "document_types": [{"type": r[0], "count": r[1]} for r in doc_types],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    action = data.get('action')
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if not client:
        return jsonify({"error": "LLM client not configured"}), 500
    if action == 'names':
        prompt = load_prompt('analyze-names.txt').replace('{text}', text)
    elif action == 'summarize':
        prompt = load_prompt('analyze-summarize.txt').replace('{text}', text)
    else:
        return jsonify({"error": "Invalid action"}), 400
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return jsonify({"result": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/pdf/<filename>', methods=['GET'])
def get_pdf(filename):
    file_id = filename.replace('.pdf', '').replace('.PDF', '')
    return redirect(f"{NARA_BASE_URL}/{file_id}.pdf")


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.errorhandler(404)
def not_found(e):
    if not request.path.startswith('/api'):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({"error": "Not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
