"""Everything that asks the model to write: routing the question, building the
prompts (sources, background, history), generating the answer and the
forced-citation retry."""
import re
from concurrent.futures import ThreadPoolExecutor

from . import config, llm, prompts
from .postprocess import HAS_CITATION, is_refusal, strip_artifacts
from .retrieval import tokenize_terms
from .schemas import Route, RouterReply


# --- Conversation history ---------------------------------------------------------
def sanitize_history(raw):
    """Trim conversation history to a bounded size. History is echoed into
    LLM prompts, so an unbounded list is a direct cost-amplification vector."""
    if not isinstance(raw, list):
        return []
    clean = []
    for msg in raw[-config.MAX_HISTORY_MSGS:]:
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            clean.append({"role": msg["role"], "content": content[:config.MAX_HISTORY_CHARS]})
    return clean


def _head_tail(text, cap):
    # Keep both ends of a long message: the tail often has the concrete
    # sources/details a follow-up is about to reference.
    if len(text) <= cap:
        return text
    half = cap // 2
    return text[:half] + ' ... ' + text[-half:]


def build_history_context(history, user_truncate=800, assistant_truncate=600):
    """Serialize recent turns for the router so follow-ups ("what about his
    wife?", "tell me more") can be resolved."""
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:
        role = msg.get('role', 'user')
        if role not in ('user', 'assistant'):
            continue
        cap = assistant_truncate if role == 'assistant' else user_truncate
        lines.append(f"{role}: {_head_tail(msg.get('content', '') or '', cap)}")
    return "\n\nConversation history (for context):\n" + "\n".join(lines)


def summarize_last_answer(history, char_cap=900):
    """Short summary of the most recent assistant answer, for the RAG agent
    so follow-ups like 'tell me more' can add information without repeating
    what was already said."""
    for msg in reversed(history):
        if msg.get('role') == 'assistant':
            return _head_tail((msg.get('content') or '').strip(), char_cap)
    return ""


# --- Router -----------------------------------------------------------------------
def route_query(query, history):
    """Classify the question, resolve pronouns from history and extract search
    keywords. On any failure, treat it as a research question over its content
    words. Search terms are capped at MAX_SEARCH_TERMS either way: each one
    can cost an LLM call downstream, and neither the router reply nor the raw
    query is a trustworthy bound."""
    prompt = prompts.render("router.txt", query=query) + build_history_context(history)
    try:
        reply = llm.complete_json(prompt, RouterReply)
    except Exception as e:
        print(f"Router failed: {e!r}")
        return Route(query_type="research", rewritten_query=query, search_terms=tokenize_terms(query)[:config.MAX_SEARCH_TERMS])
    keywords = tokenize_terms(query) if reply.keywords is None else reply.keywords
    return Route(
        query_type=reply.type,
        rewritten_query=reply.rewritten_query or query,
        search_terms=keywords[:config.MAX_SEARCH_TERMS],
        needs_retrieval=reply.needs_retrieval,
        metadata_filter=reply.metadata_filter,
    )


# --- Sources ----------------------------------------------------------------------
# OCR text occasionally contains "[7]"-style form-field numbers (~0.3% of
# pages). Inside a source block they read exactly like citation markers.
_OCR_BRACKET_NUM = re.compile(r'\[(\d{1,3})\]')


def build_context(picked, cap=None):
    """The one place sources are laid out for any model: generator, retry,
    grounding judge and citation verifier all see identical numbering and
    boundaries. Raw OCR pages have their own blank lines, headers and page
    stamps, so "[N] Source: ..." + text left it ambiguous where one source
    ended and the next began; explicit open/close tags make every passage
    attributable to exactly one [N]."""
    if not picked:
        return "NO SEARCH RESULTS FOUND."
    parts = []
    for i, r in enumerate(picked, 1):
        text = r[0] or ""  # pages with no OCR text have content NULL
        text = text if cap is None else text[:cap]
        text = _OCR_BRACKET_NUM.sub(r'(\1)', text)
        parts.append(f'<source id="{i}" file="{r[1]}" page="{r[2]}">\n{text.strip()}\n</source id="{i}">')
    return "\n\n".join(parts)


# --- Background (entity briefs) -----------------------------------------------------
_entity_brief_cache = {}


def entity_brief(name):
    """One-sentence canonical identification of a JFK-context entity, or None
    for generic terms, ambiguous names, or anything the model is unsure of.
    Cached in-process so a given name is only resolved once per worker."""
    cleaned = (name or "").strip()
    # Skip short terms and ones that don't look like proper nouns.
    if len(cleaned) < 3 or not any(w[:1].isupper() for w in cleaned.split()):
        return None
    key = cleaned.lower()
    if key in _entity_brief_cache:
        return _entity_brief_cache[key]
    try:
        out = llm.complete(
            [{"role": "user", "content": prompts.render("entity-brief.txt", name=cleaned)}],
            max_tokens=80,
        ).strip()
        # Strip a leading "Identification:" if the model echoes it.
        out = re.sub(r'^Identification:\s*', '', out, flags=re.IGNORECASE).strip()
        brief = None if not out or out.upper().startswith('NONE') or len(out) > 280 else out
    except Exception as e:
        print(f"[entity_brief] failed for {cleaned!r}: {e}")
        brief = None
    _entity_brief_cache[key] = brief
    return brief


def build_background_block(search_terms):
    """Optional BACKGROUND block from router-extracted entity names; empty
    string if no usable identifications."""
    seen, unique_terms = set(), []
    for term in search_terms or []:
        key = (term or '').strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_terms.append(term)
    # One paid LLM call per term: cap it per request.
    unique_terms = unique_terms[:config.MAX_ENTITY_BRIEFS]
    if not unique_terms:
        return ""
    # One LLM call per term; run them concurrently rather than back to back
    # (threads are greenlets under the gevent worker).
    with ThreadPoolExecutor(max_workers=len(unique_terms)) as pool:
        lines = [f"- {b}" for b in pool.map(entity_brief, unique_terms) if b]
    if not lines:
        return ""
    return (
        "BACKGROUND (mainstream historical identification — uncited; use only "
        "to identify the subject in one sentence at the start of the answer):\n"
        + "\n".join(lines)
        + "\n\n"
    )


# --- Answer prompts ---------------------------------------------------------------
def build_rag_system_prompt(query_type):
    instructions = prompts.load('rag-simple.txt' if query_type == 'simple' else 'rag-research.txt')
    return f"{instructions}\n{prompts.load('rag-system-suffix.txt')}\n"


def build_rag_user_prompt(query, ctx, prior_answer_summary, query_type="research", background_block=""):
    prior_block = ""
    if prior_answer_summary:
        prior_block = (
            "PREVIOUS ASSISTANT ANSWER (for reference only — do NOT repeat, add NEW information):\n"
            f"\"\"\"\n{prior_answer_summary}\n\"\"\"\n\n"
        )
    # Format reminder goes LAST because small models weight the tail of the
    # prompt far more than the head; tested: without it the model writes
    # non-ASCII citation brackets ~2x as often and simple answers run ~60% long.
    format_reminder = prompts.load('rag-format-simple.txt' if query_type == 'simple' else 'rag-format-research.txt')
    return prompts.render(
        "rag-user-template.txt",
        prior_block=prior_block, background_block=background_block or "",
        ctx=ctx, query=query, format_reminder=format_reminder,
    )


def generate_answer(query, picked, system_prompt, prior_answer_summary, query_type="research", background_block=""):
    """Write the answer from the picked sources; one forced-citation retry if
    it came back uncited."""
    ctx = build_context(picked)
    user_prompt = build_rag_user_prompt(query, ctx, prior_answer_summary, query_type, background_block)
    text = llm.complete(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=config.GEN_TEMPERATURE,
    )
    return retry_if_uncited(strip_artifacts(text), query, picked, system_prompt, ctx)


def retry_if_uncited(text, query, picked, system_prompt, ctx=None):
    """One forced-citation retry when an answer over real sources carries no
    [N] at all. The refusal line is left alone: it is meant to be uncited."""
    if not picked or HAS_CITATION.search(text) or is_refusal(text):
        return text
    retry_prompt = prompts.render("rag-retry-nocite.txt", ctx=ctx or build_context(picked), query=query)
    try:
        retry_text = strip_artifacts(llm.complete(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": retry_prompt}],
            temperature=config.GEN_TEMPERATURE, max_tokens=1500,
        ))
    except Exception as e:
        print(f"[rag] no-citation retry failed: {e}")
        return text
    return retry_text if HAS_CITATION.search(retry_text) else text
