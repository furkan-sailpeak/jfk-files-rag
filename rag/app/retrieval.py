"""Finding candidate pages: hybrid search (Postgres full-text ∪ pgvector),
LLM reranking, and query expansion when a first answer fails grounding.

A page is a (content, filename, page_number) tuple throughout."""
import re
import threading

from . import config, db, llm, prompts
from .postprocess import normalize_ws
from .schemas import ExpansionReply, RerankReply

# --- Query terms ----------------------------------------------------------------
# Minimal English stopword set for expansion-term filtering.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "his", "her", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "they", "this", "to", "was", "were", "will",
    "who", "whom", "with", "would", "you", "your", "i", "me", "my", "we",
    "us", "our", "what", "when", "where", "why", "how", "did", "do", "does",
    "about", "but", "not", "no", "yes", "s", "t", "re", "d", "ll", "m",
}


def tokenize_terms(text):
    """Split query text into content terms. Strips stopwords/punctuation so
    naive OR queries don't explode into noise."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
    terms = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 2]
    return terms[:6]  # cap to keep fallback query tractable


# --- Full-text leg ----------------------------------------------------------------
def _fts_query(cur, tsquery_sql, params):
    cur.execute(
        f"""
        SELECT content, filename, page_number
        FROM (
            SELECT DISTINCT ON (left(content, 200)) content, filename, page_number,
                ts_rank_cd(to_tsvector('english', content), {tsquery_sql}, 2) AS rank_score
            FROM {config.PAGES_TABLE}
            WHERE to_tsvector('english', content) @@ {tsquery_sql}
        ) sub
        ORDER BY rank_score DESC, length(content) ASC
        LIMIT 30
        """,
        params + params,
    )
    return cur.fetchall()


def fts_search(ts_input, or_terms=None):
    """FTS leg of hybrid retrieval. Strong on proper nouns and rare terms;
    weak on semantic/paraphrase match. Index-backed (GIN on tsvector), so this
    stays in the low hundreds of milliseconds.

    plainto_tsquery ANDs every word, so one OCR-mangled or absent term (or an
    expansion query phrased as a full sentence) zeroes the whole leg. When the
    AND query is thin, retry with the terms OR'd -- ts_rank_cd still ranks
    pages matching more terms first. Each term stays AND'd internally, so a
    multi-word name like "Mexico City" is not split."""
    with db.cursor() as cur:
        rows = _fts_query(cur, "plainto_tsquery('english', %s)", [ts_input])
        terms = [t for t in (or_terms or []) if t and t.strip()]
        if len(rows) >= config.FTS_MIN_HITS or len(terms) < 2:
            return rows
        or_sql = " || ".join("plainto_tsquery('english', %s)" for _ in terms)
        or_rows = _fts_query(cur, f"({or_sql})", terms)
    seen = {(r[1], r[2]) for r in rows}
    return rows + [r for r in or_rows if (r[1], r[2]) not in seen][:30 - len(rows)]


def ilike_search(ilike_terms):
    """Last-resort substring scan, used ONLY when both FTS and vector search
    come back empty.

    `content ILIKE '%term%'` cannot use the GIN index, so this is a sequential
    scan of the whole table: measured at 11.7s against ~84k pages, returning
    ~20k unranked rows for a query like "Kostikov Oswald Mexico City Soviet".
    It then keeps the LONGEST of those, which is close to meaningless ranking.
    Kept only as a safety net for when we would otherwise return nothing."""
    if not ilike_terms:
        return []
    where_clauses = " OR ".join("content ILIKE %s" for _ in ilike_terms)
    try:
        with db.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (config.ILIKE_TIMEOUT_MS,))
            cur.execute(
                f"""
                SELECT content, filename, page_number
                FROM (
                    SELECT DISTINCT ON (left(content, 200)) content, filename, page_number
                    FROM {config.PAGES_TABLE}
                    WHERE ({where_clauses})
                ) sub
                ORDER BY length(content) DESC
                LIMIT 30
                """,
                [f"%{t}%" for t in ilike_terms],
            )
            return cur.fetchall()
    except Exception as e:
        print(f"[rag] ILIKE fallback skipped ({e})")
        return []


# --- Vector leg -------------------------------------------------------------------
# bge v1.5 is trained with this instruction on the query side only; documents
# were embedded without it.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_local_embedder = None
_local_embedder_lock = threading.Lock()
print(f"Embeddings: {config.EMBED_PROVIDER} ({config.EMBED_MODEL} -> {config.EMBED_COLUMN})")


def _get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        with _local_embedder_lock:
            if _local_embedder is None:
                from fastembed import TextEmbedding
                _local_embedder = TextEmbedding(config.EMBED_MODEL, cache_dir=config.FASTEMBED_CACHE_PATH)
    return _local_embedder


def embed_query(text):
    """Embed a single query string. Returns list[float] or None on failure."""
    if not text or not text.strip():
        return None
    try:
        if config.EMBED_PROVIDER != "openai":
            vec = next(iter(_get_local_embedder().embed([BGE_QUERY_PREFIX + text[:2000]])))
            return vec.tolist()
        if not config.OPENAI_API_KEY:
            return None
        from openai import OpenAI
        resp = OpenAI(api_key=config.OPENAI_API_KEY).embeddings.create(
            model=config.EMBED_MODEL, input=text[:8000], dimensions=config.EMBED_DIM,
        )
        return resp.data[0].embedding
    except Exception as e:
        # Swallowed so a provider outage degrades to FTS-only instead of a
        # failed request -- but it then fails silently on every query, so make
        # it impossible to miss in the logs.
        print(f"[rag] WARNING: query embedding failed, vector leg disabled for this query: {e}")
        return None


def vector_search(query_text, limit=30):
    """Vector leg of hybrid retrieval. Strong on semantic/paraphrase and
    summary-style content; weak on exact name matching."""
    vec = embed_query(query_text)
    if vec is None:
        return []
    vec_lit = "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
    with db.cursor() as cur:
        cur.execute("SET LOCAL ivfflat.probes = %s", (config.IVFFLAT_PROBES,))
        cur.execute("SET LOCAL hnsw.ef_search = %s", (config.HNSW_EF_SEARCH,))
        # EMBED_COLUMN is one of two hard-coded names, never user input.
        cur.execute(
            f"""
            SELECT content, filename, page_number
            FROM {config.PAGES_TABLE}
            WHERE {config.EMBED_COLUMN} IS NOT NULL
            ORDER BY {config.EMBED_COLUMN} <=> %s::halfvec
            LIMIT %s
            """,
            [vec_lit, limit],
        )
        return cur.fetchall()


def hybrid_search(ts_input, semantic_query, ilike_terms=None, or_terms=None):
    """Union of FTS + vector candidates, deduped on (filename, page_number).
    FTS gets keyword-only input; vector gets the full rewritten question —
    each leg is fed what it's best at."""
    fts_rows = fts_search(ts_input, or_terms or ilike_terms)
    vec_rows = vector_search(semantic_query, limit=30)

    # The substring fallback is a full table scan, so it only earns its cost
    # when both indexed legs came back empty.
    if not fts_rows and not vec_rows:
        fts_rows = ilike_search(ilike_terms or tokenize_terms(ts_input))

    merged, seen = [], set()
    # FTS first so exact matches aren't drowned by semantic neighbors.
    for rows in (fts_rows, vec_rows):
        for r in rows:
            key = (r[1], r[2])
            # Also dedupe near-identical content (same first 200 chars).
            content_key = r[0][:200].strip()
            if key in seen or content_key in seen:
                continue
            seen.add(key)
            seen.add(content_key)
            merged.append(r)
    return merged


# --- Rerank -----------------------------------------------------------------------
def _rerank_snippet(content, terms, width=400):
    # Center the snippet on the first query-term hit so the judge sees the
    # match in context. Slicing from position 0 hides the match whenever the
    # page starts with a routing header (common in JFK records).
    norm = normalize_ws(content)
    lower = norm.lower()
    hits = [lower.find(t.lower()) for t in terms if t and t.lower() in lower]
    if not hits:
        return content[:width]
    start = max(0, min(hits) - 80)
    return norm[start:start + width]


def _dedupe_picks(candidates, picked_indices, context_limit, fill_pool):
    # Drop near-duplicate pages (same first 200 chars after whitespace
    # normalization) so three copies of the same memo don't crowd out a
    # second, distinct passage. Backfill from the next-ranked candidates.
    seen_keys, out, out_idx = set(), [], set()
    for i in list(picked_indices) + list(fill_pool):
        if len(out) >= context_limit:
            break
        if i in out_idx:
            continue
        key = normalize_ws(candidates[i][0])[:200].strip().lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out_idx.add(i)
        out.append(candidates[i])
    return out


def rerank(candidates, rewritten_query, context_limit, search_terms=None):
    """LLM picks the `context_limit` most relevant candidates; falls back to
    retrieval order if the call fails."""
    if len(candidates) <= context_limit:
        return candidates[:context_limit]
    terms = [t for t in (search_terms or rewritten_query.split()) if t]
    snippets = [
        f"[{idx}] {r[1]}, Page {r[2]}: {_rerank_snippet(r[0], terms).replace(chr(10), ' ').strip()}"
        for idx, r in enumerate(candidates)
    ]
    prompt = prompts.render(
        "reranker.txt",
        context_limit=context_limit, rewritten_query=rewritten_query, snippets="\n".join(snippets),
    )
    try:
        # Rerank is the one stage that needs the larger model plus a hard
        # schema. Under plain json_object mode gpt-oss serialises the ranking
        # as a single concatenated string ("024681012...") instead of an array
        # of integers. The 20B fails even WITH the schema; the 120B is reliable.
        reply = llm.complete_json(prompt, RerankReply, judge=False, strict_schema=True)
        valid = [i for i in reply.indices if 0 <= i < len(candidates)]
        fill_pool = [i for i in range(len(candidates)) if i not in set(valid)]
        return _dedupe_picks(candidates, valid, context_limit, fill_pool)
    except Exception as e:
        print(f"Rerank failed, using FTS order: {e}")
        return _dedupe_picks(candidates, range(len(candidates)), context_limit, [])


# --- Query expansion --------------------------------------------------------------
def expand_and_retrieve(rewritten_query, reason, seed_results):
    """Generate 3 alternative phrasings, retrieve, merge with seed_results."""
    prompt = prompts.render("expansion.txt", rewritten_query=rewritten_query, reason=reason)
    try:
        reply = llm.complete_json(prompt, ExpansionReply, temperature=0.4)
        expanded = [q for q in reply.queries if q.strip()]
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
        for r in hybrid_search(eq, eq, tokenize_terms(eq)):
            key = r[0][:200].strip()
            if key not in seen:
                merged.append(r)
                seen.add(key)
    return merged


# --- Direct lookups (no ranking) ----------------------------------------------------
def fetch_document(doc_id):
    """All pages of one document, by NARA record number ("104-10004-10143").
    Returns (filename, pages); pages is empty if no such document.

    ~2% of file_ids carry a suffix ("_redacted", "_multirif", " (C06931192)"),
    so an exact match on the bare ID missed them. Prefer the exact file;
    otherwise take the first suffixed variant."""
    with db.cursor() as cur:
        cur.execute(
            f"""
            SELECT file_id FROM {config.PAGES_TABLE}
            WHERE file_id = %s OR file_id LIKE %s
            GROUP BY file_id
            ORDER BY (file_id = %s) DESC, file_id
            LIMIT 1
            """,
            (doc_id, f"{doc_id}%", doc_id),
        )
        match = cur.fetchone()
        if not match:
            return f"{doc_id}.pdf", []
        cur.execute(
            f"SELECT content, filename, page_number FROM {config.PAGES_TABLE} WHERE file_id = %s ORDER BY page_number",
            (match[0],),
        )
        return f"{match[0]}.pdf", cur.fetchall()


def metadata_query(column, value, n_samples=10):
    """Archive-level counts for one metadata property, plus the first page of
    up to n_samples matching documents. `column` must be in
    config.METADATA_COLUMNS (it is interpolated into the SQL).
    Returns (matching_pages, matching_documents, sample_pages)."""
    assert column in config.METADATA_COLUMNS, column
    # Booleans match exactly; strings (document_type) match as a substring.
    cond, param = (f"{column} = %s", value) if isinstance(value, bool) else (f"{column} ILIKE %s", f"%{value}%")
    with db.cursor() as cur:
        cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT file_id) FROM {config.PAGES_TABLE} WHERE {cond}", (param,))
        total_pages, total_docs = cur.fetchone()
        cur.execute(f"""
            SELECT DISTINCT ON (file_id) content, filename, page_number
            FROM {config.PAGES_TABLE} WHERE {cond} AND content IS NOT NULL
            ORDER BY file_id, page_number LIMIT %s
        """, (param, n_samples))
        return total_pages, total_docs, cur.fetchall()
