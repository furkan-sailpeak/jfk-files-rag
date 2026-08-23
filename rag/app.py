import os
import re
import json
import time
import threading
from contextlib import contextmanager

import jwt
import psycopg2
from psycopg2 import pool as pgpool
from flask import Flask, request, jsonify, redirect, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='frontend/dist', static_url_path='/')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- CORS -------------------------------------------------------------------
# Bare CORS(app) allows every origin, which lets any third-party site drive
# this backend on our LLM budget. Lock to our own domains in production;
# fall back to permissive only when ALLOWED_ORIGINS is unset (local dev).
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if ALLOWED_ORIGINS:
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})
    print(f"CORS locked to: {ALLOWED_ORIGINS}")
else:
    CORS(app)
    print("WARNING: ALLOWED_ORIGINS unset — CORS is open to all origins (dev mode).")

# --- Auth (Supabase JWT) ----------------------------------------------------
# Supabase supports two token-signing schemes and which one a project uses
# depends on when it was created:
#   * Asymmetric (ES256/RS256) — current default. Tokens are verified against
#     the project's public JWKS endpoint. Nothing secret goes in the env.
#   * HS256 — the legacy shared "JWT Secret". Still valid on older projects.
# Support both so this works regardless of project age, and so a future
# key rotation to asymmetric doesn't lock every user out.
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

_jwks_client = None
_jwks_lock = threading.Lock()


def _get_jwks_client():
    """Lazily build a JWKS client. PyJWT caches the fetched keys, so this is
    one network call per process, not per request."""
    global _jwks_client
    if _jwks_client is not None or not SUPABASE_URL:
        return _jwks_client
    with _jwks_lock:
        if _jwks_client is None:
            _jwks_client = jwt.PyJWKClient(
                f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json",
                cache_keys=True,
            )
    return _jwks_client


AUTH_CONFIGURED = bool(SUPABASE_URL or SUPABASE_JWT_SECRET)
if not AUTH_CONFIGURED:
    print("WARNING: neither SUPABASE_URL nor SUPABASE_JWT_SECRET set — "
          "authentication disabled; the sign-in quota gate is not enforced.")

# --- Abuse / cost limits ----------------------------------------------------
# TEMPORARILY DISABLED (see todo.md): the whole sign-in + quota gate is off by
# default so the thesis demo runs without a login wall. Set GATE_ENABLED=1 to
# turn the limiter back on; nothing below was removed.
GATE_ENABLED = os.getenv("GATE_ENABLED", "0").lower() in ("1", "true", "yes")

# Anonymous visitors get a small free trial so media traffic can try the tool
# without signing up; signed-in users get a real daily quota.
ANON_DAILY_LIMIT = int(os.getenv("ANON_DAILY_LIMIT", "3"))
USER_DAILY_LIMIT = int(os.getenv("USER_DAILY_LIMIT", "50"))
BURST_PER_MIN = int(os.getenv("BURST_PER_MIN", "5"))

# Hard caps on anything that reaches an LLM call — unbounded input is a
# direct cost-drain vector on a public endpoint.
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "2000"))
MAX_HISTORY_MSGS = int(os.getenv("MAX_HISTORY_MSGS", "20"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "4000"))
MAX_ANALYZE_CHARS = int(os.getenv("MAX_ANALYZE_CHARS", "20000"))

# Cosmetic token-replay pacing for the final answer. Total replay is capped at
# REPLAY_BUDGET_S regardless of answer length; per-token delay never exceeds
# REPLAY_MAX_DELAY_S so short answers still animate visibly. Set the budget to
# 0 to disable the animation entirely.
REPLAY_BUDGET_S = float(os.getenv("REPLAY_BUDGET_S", "1.5"))
REPLAY_MAX_DELAY_S = float(os.getenv("REPLAY_MAX_DELAY_S", "0.015"))

NARA_BASE_URL = "https://storage.googleapis.com/jfkweb-prod"

# Main LLM client. llama-3.3-70b-versatile was decommissioned by Groq on
# 2026-08-16 and now returns 404 model_not_found; gpt-oss-120b is Groq's
# recommended replacement. Overridable so a swap needs no code change.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
    MODEL = DEFAULT_MODEL
    LLM_PROVIDER = "groq"
    print(f"LLM provider: Groq, model={MODEL}")
else:
    print("WARNING: GROQ_API_KEY not found; /api/chat will 500.")
    client = None
    MODEL = DEFAULT_MODEL
    LLM_PROVIDER = "none"

# Judge/rerank/router/expansion/citation-verify are cheap, high-volume utility
# calls that emit small structured JSON rather than prose. The 20B model is
# sized for exactly that; using the 120B here balloons cost and latency for no
# measurable quality gain. Raise to the 120B via JUDGE_MODEL if eval scores
# regress on these stages.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")
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


def reasoning_kwargs(model_name):
    """gpt-oss models emit chain-of-thought before the answer. Left at the
    defaults that costs real latency (0.40s to first token vs 0.14s) and,
    worse, the hidden reasoning still counts against max_tokens — a small cap
    can be consumed entirely by reasoning, returning empty content.

    'hidden' keeps reasoning out of the response, 'low' keeps it short.
    Returns {} for non-reasoning models so this stays a no-op after a swap.
    """
    if "gpt-oss" in model_name:
        return {"reasoning_effort": "low", "reasoning_format": "hidden"}
    return {}


GEN_KW = reasoning_kwargs(MODEL)
JUDGE_KW = reasoning_kwargs(JUDGE_MODEL)


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
def ensure_schema():
    """One-time bootstrap: FTS index + the rate-limit counter table."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_jfk_pages_content_fts
            ON jfk_pages USING GIN (to_tsvector('english', content))
        """)
        # Rate-limit counters live in Postgres rather than process memory so
        # the limit is shared across gunicorn workers. An in-memory counter
        # would multiply every quota by the worker count.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rate_limits (
                bucket      text PRIMARY KEY,
                count       integer NOT NULL DEFAULT 0,
                window_start timestamptz NOT NULL DEFAULT now()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_rate_limits_window
            ON rate_limits (window_start)
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("Schema ready (FTS index + rate_limits).")
    except Exception as e:
        print(f"Schema bootstrap skipped: {e}")


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------
# Previously every db_cursor() opened a fresh connection — a full TCP + TLS
# handshake to Supabase per query, several per request. Under launch traffic
# that exhausts Supabase's connection limit long before the app itself
# saturates. Pool instead; connections are still returned promptly so none is
# held across a multi-second LLM roundtrip.
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "8"))
_db_pool = None


def init_db_pool():
    global _db_pool
    if not DATABASE_URL or _db_pool is not None:
        return
    try:
        _db_pool = pgpool.ThreadedConnectionPool(DB_POOL_MIN, DB_POOL_MAX, DATABASE_URL)
        print(f"DB pool ready ({DB_POOL_MIN}-{DB_POOL_MAX} connections).")
    except Exception as e:
        print(f"DB pool init failed, falling back to per-request connects: {e}")
        _db_pool = None


if DATABASE_URL:
    ensure_schema()
    init_db_pool()


@contextmanager
def db_cursor():
    """Pooled connection, returned as soon as the DB work is done.
    Never held across an LLM roundtrip."""
    conn = _db_pool.getconn() if _db_pool else psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
        cur.close()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if _db_pool:
            _db_pool.putconn(conn)
        else:
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


# ---------------------------------------------------------------------------
# Auth (Supabase JWT) + rate limiting
# ---------------------------------------------------------------------------
def get_user():
    """Decode a Supabase access token from the Authorization header.
    Returns {"id", "email"} for a valid token, else None (anonymous).
    Verification is local (HS256 against the project JWT secret) so it costs
    no network roundtrip per request."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    token = header[7:].strip()
    if not token:
        return None
    try:
        # The header tells us which scheme signed this token; pick the
        # matching verification path rather than guessing.
        alg = jwt.get_unverified_header(token).get("alg", "")
        if alg.startswith("HS"):
            if not SUPABASE_JWT_SECRET:
                return None
            key, algorithms = SUPABASE_JWT_SECRET, ["HS256"]
        else:
            client_jwks = _get_jwks_client()
            if not client_jwks:
                return None
            key = client_jwks.get_signing_key_from_jwt(token).key
            algorithms = ["ES256", "RS256"]

        payload = jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience="authenticated",
        )
        return {"id": payload.get("sub"), "email": payload.get("email")}
    except Exception as e:
        print(f"[auth] rejected token: {e}")
        return None


def _client_ip():
    """Railway sits behind a proxy, so remote_addr is the proxy. Take the
    first hop from X-Forwarded-For, which is the real client."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _bump(bucket):
    """Atomically increment a counter bucket and return its new value."""
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO rate_limits (bucket, count) VALUES (%s, 1)
            ON CONFLICT (bucket) DO UPDATE SET count = rate_limits.count + 1
            RETURNING count
            """,
            (bucket,),
        )
        return cur.fetchone()[0]


def check_rate_limit(user):
    """Two windows: a per-minute burst guard and a daily quota. Buckets carry
    the window in the key, so they expire implicitly — no sweeper needed on
    the hot path.

    Returns (allowed, reason, remaining_today).

    Fails OPEN on database error: a rate-limiter outage should not take the
    whole app down during a launch. The burst guard still caps the blast
    radius if that ever happens.
    """
    if not GATE_ENABLED:
        return True, "", USER_DAILY_LIMIT

    now = time.gmtime()
    day = time.strftime("%Y-%m-%d", now)
    minute = time.strftime("%Y-%m-%dT%H:%M", now)

    if user:
        ident, daily_cap = f"user:{user['id']}", USER_DAILY_LIMIT
    else:
        # When auth isn't configured there is no sign-in to upgrade to, so
        # applying the small unauthenticated allowance would strand the user
        # at a dead end. Fall back to the full quota; the burst guard still
        # applies.
        ident = f"ip:{_client_ip()}"
        daily_cap = ANON_DAILY_LIMIT if AUTH_CONFIGURED else USER_DAILY_LIMIT

    try:
        if _bump(f"{ident}:m:{minute}") > BURST_PER_MIN:
            return False, "burst", 0
        used = _bump(f"{ident}:d:{day}")
    except Exception as e:
        print(f"[ratelimit] check failed, allowing request: {e}")
        return True, "", daily_cap

    if used > daily_cap:
        return False, "daily", 0
    return True, "", max(0, daily_cap - used)


def sanitize_history(raw):
    """Trim conversation history to a bounded size. History is echoed into
    LLM prompts, so an unbounded list is a direct cost-amplification vector."""
    if not isinstance(raw, list):
        return []
    clean = []
    for msg in raw[-MAX_HISTORY_MSGS:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        clean.append({"role": role, "content": content[:MAX_HISTORY_CHARS]})
    return clean


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


def fts_search(ts_input):
    """FTS leg of hybrid retrieval. Strong on proper nouns and rare terms;
    weak on semantic/paraphrase match. Index-backed (GIN on tsvector), so this
    stays in the low hundreds of milliseconds."""
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
        return cur.fetchall()


# Worst-case guard for the unindexed fallback below. Without it a single
# pathological query can pin a connection for the whole request timeout.
ILIKE_TIMEOUT_MS = int(os.getenv("ILIKE_TIMEOUT_MS", "4000"))


def ilike_search(ilike_terms):
    """Last-resort substring scan, used ONLY when both FTS and vector search
    come back empty.

    `content ILIKE '%term%'` cannot use the GIN index, so this is a sequential
    scan of the whole table: measured at 11.7s against ~84k pages, returning
    ~20k unranked rows for a query like "Kostikov Oswald Mexico City Soviet".
    It then keeps the LONGEST of those, which is close to meaningless ranking.

    Running it speculatively on every miss was costing ~33s per query-expansion
    round (3 expansion queries, none of which match plainto_tsquery because
    they are full sentences). It is kept only as a genuine safety net for when
    we would otherwise return nothing at all.
    """
    if not ilike_terms:
        return []
    where_clauses = " OR ".join("content ILIKE %s" for _ in ilike_terms)
    try:
        with db_cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = %s", (ILIKE_TIMEOUT_MS,))
            cur.execute(
                f"""
                SELECT content, filename, page_number
                FROM (
                    SELECT DISTINCT ON (left(content, 200)) content, filename, page_number
                    FROM jfk_pages
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
    fts_rows = fts_search(ts_input)
    vec_rows = vector_search(semantic_query, limit=30)

    # The substring fallback is a full table scan, so it only earns its cost
    # when both indexed legs came back empty. Previously it fired whenever FTS
    # alone missed -- which is the normal case for the full-sentence queries
    # produced by expansion, even though the vector leg had already answered.
    if not fts_rows and not vec_rows:
        fts_rows = ilike_search(ilike_terms or _tokenize_terms(ts_input))

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
# gpt-oss emits citations in the format it was trained on rather than the [6]
# this pipeline expects. Two variants seen in production:
#   【6】            -- CJK/full-width brackets
#   【2†L31-L38】    -- OpenAI's file-citation form, with a dagger and a
#                      line-range annotation invented from whole cloth
# Every downstream consumer -- remap_citations, verify_citations, the
# no-citation retry, and the frontend's link injection -- matches on ASCII
# \[\d+\] only. An un-normalised answer therefore loses its entire citation
# chain: verification is skipped, no link renders, and the grounding judge
# reports "cites a source that is not provided" -- which triggers the
# expensive expansion path on essentially every query.
#
# The \d{1,3} bound is deliberate: it keeps bracketed years such as [1963],
# which are common in this archive, from being rewritten as citations.
_CITE_BRACKETS = re.compile(
    r'[【\[［〔]\s*(\d{1,3})\s*(?:†[^】\]］〕]*)?\s*[】\]］〕]'
)


def normalize_citations(text):
    return _CITE_BRACKETS.sub(r'[\1]', text)


def strip_artifacts(text):
    text = normalize_citations(text)
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
            model=JUDGE_MODEL, **JUDGE_KW,
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
        # Rerank is the one stage that needs the larger model plus a hard
        # schema. Under plain json_object mode gpt-oss serialises the ranking
        # as a single concatenated string ("024681012...") instead of an array
        # of integers, which this function then discards — silently falling
        # back to raw FTS order on every query. The 20B fails even WITH the
        # schema; the 120B satisfies it reliably.
        res = client.chat.completions.create(
            model=MODEL, **GEN_KW,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rerank",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "indices": {"type": "array", "items": {"type": "integer"}}
                        },
                        "required": ["indices"],
                        "additionalProperties": False,
                    },
                },
            },
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


_entity_brief_cache = {}


def entity_brief(name):
    """Return a one-sentence canonical identification of a JFK-context entity,
    or None. Uses Groq + an in-memory cache so a given name is only resolved
    once per process. Returns None for generic terms, ambiguous names, or
    anything the model is not confident about."""
    if not name:
        return None
    cleaned = name.strip()
    if len(cleaned) < 3:
        return None
    # Skip terms that don't look like proper nouns (no capitalized word).
    if not any(w[:1].isupper() for w in cleaned.split()):
        return None
    key = cleaned.lower()
    if key in _entity_brief_cache:
        return _entity_brief_cache[key]
    try:
        prompt = load_prompt('entity-brief.txt').replace('{name}', cleaned)
        res = client.chat.completions.create(
            model=MODEL, **GEN_KW,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=80,
        )
        out = (res.choices[0].message.content or '').strip()
        # Strip a leading "Identification:" if the model echoes it.
        out = re.sub(r'^Identification:\s*', '', out, flags=re.IGNORECASE).strip()
        if not out or out.upper().startswith('NONE') or len(out) > 280:
            _entity_brief_cache[key] = None
            return None
        _entity_brief_cache[key] = out
        return out
    except Exception as e:
        print(f"[entity_brief] failed for {cleaned!r}: {e}")
        _entity_brief_cache[key] = None
        return None


def build_background_block(search_terms):
    """Build an optional BACKGROUND block from router-extracted entity names.
    Empty string if no usable identifications. Each line is one entity brief."""
    if not search_terms:
        return ""
    seen = set()
    lines = []
    for term in search_terms:
        key = (term or '').strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        brief = entity_brief(term)
        if brief:
            lines.append(f"- {brief}")
    if not lines:
        return ""
    return (
        "BACKGROUND (mainstream historical identification — uncited; use only "
        "to identify the subject in one sentence at the start of the answer):\n"
        + "\n".join(lines)
        + "\n\n"
    )


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


def build_rag_user_prompt(query, ctx, prior_answer_summary, query_type="research", background_block=""):
    prior_block = ""
    if prior_answer_summary:
        prior_block = (
            "PREVIOUS ASSISTANT ANSWER (for reference only — do NOT repeat, add NEW information):\n"
            f"\"\"\"\n{prior_answer_summary}\n\"\"\"\n\n"
        )

    # Format reminder goes LAST because small models weight the tail of the
    # prompt far more than the head; the long system prompt's format rules
    # are otherwise forgotten by the time the model starts generating.
    format_reminder = load_prompt(
        'rag-format-simple.txt' if query_type == 'simple' else 'rag-format-research.txt'
    )

    return (
        load_prompt('rag-user-template.txt')
        .replace('{prior_block}', prior_block)
        .replace('{background_block}', background_block or "")
        .replace('{ctx}', ctx)
        .replace('{query}', query)
        .replace('{format_reminder}', format_reminder)
    )


def generate_answer_stream(query, picked, system_prompt, prior_answer_summary, query_type="research", background_block=""):
    """Stream the generation. Yields (kind, payload) tuples:
    ("token", text) for incremental text, ("done", full_text) at end."""
    ctx = build_context(picked)
    user_prompt = build_rag_user_prompt(query, ctx, prior_answer_summary, query_type, background_block)
    full_text = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL, **GEN_KW,
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
            model=MODEL, **GEN_KW,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        full_text = res.choices[0].message.content
        yield ("token", full_text)
    yield ("done", full_text)


def generate_answer_nonstream(query, picked, system_prompt, prior_answer_summary, query_type="research", background_block=""):
    ctx = build_context(picked)
    user_prompt = build_rag_user_prompt(query, ctx, prior_answer_summary, query_type, background_block)
    res = client.chat.completions.create(
        model=MODEL, **GEN_KW,
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
            model=MODEL, **GEN_KW,
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
    # Cap each source individually rather than truncating the joined block.
    # A global cap (this was build_context(picked)[:4000]) showed the judge
    # only the first ~4 of 20 sources, so any citation past [4] looked like an
    # invented reference and grounding failed on almost every research answer
    # -- which then triggered the expensive expansion path every time.
    # verify_citations already avoids this; the judge needs the same treatment.
    per_source_cap = 1500
    sources_block = "\n\n".join(
        f"[{i}] Source: {r[1]}, Page {r[2]}\n{r[0][:per_source_cap]}"
        for i, r in enumerate(picked, 1)
    )
    judge_prompt = (
        load_prompt('grounding-judge.txt')
        .replace('{rewritten_query}', rewritten_query)
        .replace('{sources_block}', sources_block)
        .replace('{answer}', answer[:6000])
    )
    try:
        res = judge_client.chat.completions.create(
            model=JUDGE_MODEL, **JUDGE_KW,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        return bool(data.get("grounded", True)), data.get("reason", "")
    except Exception as e:
        print(f"Grounding check skipped: {e}")
        return True, ""


# The verifier judges each citation *instance* — "does source [N] support the
# sentence it is attached to" — so its verdicts have to be applied at the same
# granularity. Stripping every `[N]` in the answer because one attachment of N
# was weak deleted whole documents from an answer (and, via remap_citations,
# from the source panel) on the strength of a single marginal judgement. To
# scope a verdict we need a stable address for each occurrence, so the answer
# is split into numbered units and the verifier reports (unit, source) pairs.
#
# The split must be lossless: re.split with a capturing group keeps the
# separators, so units[i] + seps[i] reassembles the original text byte for
# byte. Over-splitting (abbreviations, "No." in a file reference) is harmless
# — it only makes units smaller, never misaligns them.
_CITE_UNIT_SPLIT = re.compile(r'(\n+|(?<=[.!?])\s+)')

# Char budget for the answer as shown to the verifier. Sentences past this are
# left unjudged and keep their citations — the failure mode is a citation that
# should have been stripped surviving, never a good one being deleted.
VERIFY_ANSWER_CHARS = int(os.getenv("VERIFY_ANSWER_CHARS", "4000"))


def _split_cite_units(text):
    """Split an answer into sentence/line units. Returns (units, separators)
    such that _join_cite_units(units, separators) == text."""
    parts = _CITE_UNIT_SPLIT.split(text)
    return parts[0::2], parts[1::2]


def _join_cite_units(units, seps):
    out = []
    for i, unit in enumerate(units):
        out.append(unit)
        if i < len(seps):
            out.append(seps[i])
    return "".join(out)


def verify_citations(answer, picked):
    """Per-citation-instance check: does source [N] actually support the
    sentence that cites it? Returns a list of (unit_index, source_num) pairs
    identifying the individual occurrences to drop. Cheap safeguard against
    the LLM attaching arbitrary [N] to fabricated claims."""
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

    units, _ = _split_cite_units(answer)
    numbered, budget = [], VERIFY_ANSWER_CHARS
    for i, unit in enumerate(units):
        stripped = unit.strip()
        if not stripped:
            continue
        line = f"U{i}: {stripped}"
        if len(line) > budget:
            break
        budget -= len(line)
        numbered.append(line)

    prompt = (
        load_prompt('citation-verify.txt')
        .replace('{sources_block}', sources_block)
        .replace('{answer}', "\n".join(numbered))
    )
    try:
        res = judge_client.chat.completions.create(
            model=JUDGE_MODEL, **JUDGE_KW,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        bad = []
        for item in data.get("unsupported", []):
            # Bare numbers are the old, unscoped verdict form. They name a
            # source but not which of its citations failed, and acting on one
            # means deleting them all — the bug this scoping exists to fix. So
            # ignore them: an unjudged citation is the safer outcome.
            if not isinstance(item, dict):
                print(f"[cite-verify] ignoring unscoped verdict: {item!r}")
                continue
            unit, src = item.get("unit"), item.get("source")
            if str(unit).isdigit() and str(src).isdigit():
                bad.append((int(unit), int(src)))
        return bad
    except Exception as e:
        print(f"Citation verification skipped: {e}")
        return []


def strip_unsupported_citations(answer, unsupported):
    """Remove only the flagged [N] occurrences, each in the one unit where the
    verifier judged it unsupported. Other citations of the same source, in
    sentences it does support, are left alone."""
    if not unsupported:
        return answer
    units, seps = _split_cite_units(answer)
    by_unit = {}
    for unit_idx, src in unsupported:
        by_unit.setdefault(unit_idx, set()).add(src)
    for unit_idx, nums in by_unit.items():
        if not 0 <= unit_idx < len(units):
            continue
        text = units[unit_idx]
        # Remove the marker only, leaving surrounding whitespace intact —
        # eating the leading space turns "April [2][5]" into "April[5]" when
        # the neighbouring citation survives. Tidy up afterwards instead.
        for n in sorted(nums, reverse=True):
            text = re.sub(rf'\[{n}\]', '', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'[ \t]+([.,;:!?])', r'\1', text)
        units[unit_idx] = text
    return _join_cite_units(units, seps)


def expand_and_retrieve(rewritten_query, reason, seed_results):
    """Generate 3 alternative phrasings, retrieve, merge with seed_results."""
    prompt = (
        load_prompt('expansion.txt')
        .replace('{rewritten_query}', rewritten_query)
        .replace('{reason}', reason)
    )
    try:
        res = judge_client.chat.completions.create(
            model=JUDGE_MODEL, **JUDGE_KW,
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
    data = request.get_json(silent=True) or {}
    query = data.get('query')
    history = sanitize_history(data.get('history'))

    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({"error": "No query provided"}), 400
    query = query.strip()[:MAX_QUERY_CHARS]
    if not client:
        return jsonify({"error": "LLM client not configured"}), 500

    # --- Gate: free trial for anonymous visitors, quota for signed-in users --
    user = get_user()
    allowed, reason, remaining = check_rate_limit(user)
    if not allowed:
        if reason == "burst":
            return jsonify({
                "error": "rate_limited",
                "message": "Too many requests in a short time. Please wait a moment and try again.",
            }), 429
        if user:
            return jsonify({
                "error": "daily_limit",
                "message": f"You've reached your daily limit of {USER_DAILY_LIMIT} requests. It resets at 00:00 UTC.",
            }), 429
        return jsonify({
            "error": "trial_exhausted",
            "message": (
                f"You've reached the limit of {ANON_DAILY_LIMIT} unauthenticated "
                "requests. Sign in to continue exploring the archive."
            ),
        }), 401

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
                        model=MODEL, **GEN_KW,
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
                        model=MODEL, **GEN_KW,
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
                            model=MODEL, **GEN_KW,
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
                            model=MODEL, **GEN_KW,
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
                        model=MODEL, **GEN_KW,
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
                        model=MODEL, **GEN_KW, messages=conv_messages, temperature=0.5,
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
            background_block = build_background_block(search_terms)

            # First generation — collected silently (not streamed to client yet).
            # We stream only the final verified answer after grounding + citation checks.
            yield sse("stage", {"label": "Generating..."})
            t_gen = time.time()
            full_text = ""
            for kind, payload in generate_answer_stream(query, final_results, system_prompt, prior_summary, query_type, background_block):
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
                answer_text = generate_answer_nonstream(query, final_results, system_prompt, prior_summary, query_type, background_block)

            # Citation verification — strip bad citations (or regenerate if many bad)
            yield sse("stage", {"label": "Verifying citations..."})
            t_cv = time.time()
            unsupported = verify_citations(answer_text, final_results)
            timings["cite_verify_ms"] = int((time.time() - t_cv) * 1000)
            if unsupported:
                print(f"[rag] unsupported citations (unit, source): {unsupported}")
                # Scoped strip: only the flagged occurrences, so a source that
                # is genuinely cited elsewhere keeps those citations.
                answer_text = strip_unsupported_citations(answer_text, unsupported)

            all_sources = [{"filename": r[1], "page": r[2]} for r in final_results]
            answer_text, cited_sources = remap_citations(answer_text, all_sources)
            # remap_citations renumbers the [N] markers in the text to 1..N in
            # citation order, so the panel MUST be reordered to match. Sending
            # the original all_sources here meant every citation resolved to
            # the wrong document: a model citing sources [2], [7], [8] has its
            # markers rewritten to [1], [2], [3], which then pointed at sources
            # 1, 2 and 3 in the unchanged panel.
            #
            # Uncited sources are appended after the cited ones, so the panel
            # still surfaces everything that was retrieved while the numbering
            # of cited entries stays exact.
            extras = [s for s in all_sources if s not in cited_sources]
            sources_out = cited_sources + extras

            # Stream the final, verified answer to the frontend. The answer is
            # already complete at this point — this replay exists purely so the
            # UI animates in rather than arriving as one block.
            #
            # A flat 15ms/token made that animation the single most expensive
            # phase of the request: a ~800-token research answer spent 12s here
            # against 7s of actual work, holding the connection almost 3x
            # longer for zero informational gain. Budget the whole replay
            # instead, so long answers speed up rather than dragging on.
            yield sse("stage", {"label": "Streaming answer..."})
            tokens_out = re.findall(r'\S+\s*', answer_text)
            delay = min(REPLAY_MAX_DELAY_S, REPLAY_BUDGET_S / max(len(tokens_out), 1))
            for tok in tokens_out:
                yield sse("token", {"text": tok})
                if delay > 0:
                    time.sleep(delay)

            timings["total_ms"] = int((time.time() - t0) * 1000)
            print(f"[timing] {timings}")
            yield final_event(answer_text, sources_out, query_type, timings)

        except Exception as e:
            print(f"Error in /api/chat stream: {e}")
            yield sse("error", {"message": str(e)})

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            # Frontend reads this to show "N unauthenticated requests left"
            # before the sign-in prompt appears.
            "X-Quota-Remaining": str(remaining),
            "X-Quota-Authenticated": "1" if user else "0",
            "Cache-Control": "no-cache",
            # Stop intermediate proxies buffering the SSE stream.
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Stats / analyze / pdf routes
# ---------------------------------------------------------------------------
# The archive is static, so these counts never change between deploys — but
# the route was recomputing eight sequential aggregates over ~70k rows on
# every page load (measured 8.7-19.4s in production). Cache aggressively and
# collapse the whole thing into ONE query so a cold cache costs one scan, not
# eight. Frontend calls this on mount, so this path is every visitor's path.
_full_stats = {"ts": 0, "data": None}
_FULL_STATS_TTL = int(os.getenv("STATS_TTL", "3600"))
_full_stats_lock = threading.Lock()


def compute_full_stats():
    with db_cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*),
                COUNT(DISTINCT file_id),
                COUNT(*) FILTER (WHERE content IS NOT NULL AND length(trim(content)) > 0),
                COUNT(DISTINCT file_id) FILTER (WHERE content IS NOT NULL AND length(trim(content)) > 0),
                COUNT(*) FILTER (WHERE includes_handwriting = true),
                COUNT(*) FILTER (WHERE has_stamps = true),
                COUNT(*) FILTER (WHERE has_redactions = true)
            FROM jfk_pages
        """)
        (total_pages, total_docs, pages_with_content, docs_with_content,
         handwritten_pages, stamped_pages, redacted_pages) = cur.fetchone()
        cur.execute("""
            SELECT document_type, COUNT(*) AS count FROM jfk_pages
            GROUP BY document_type ORDER BY count DESC LIMIT 5
        """)
        doc_types = cur.fetchall()

    page_pct = (pages_with_content / total_pages * 100) if total_pages else 0
    doc_pct = (docs_with_content / total_docs * 100) if total_docs else 0
    return {
        "total_pages": total_pages, "total_docs": total_docs,
        "pages_with_content": pages_with_content, "docs_with_content": docs_with_content,
        "page_content_pct": round(page_pct, 1), "doc_content_pct": round(doc_pct, 1),
        "handwritten_pages": handwritten_pages, "stamped_pages": stamped_pages,
        "redacted_pages": redacted_pages,
        "document_types": [{"type": r[0], "count": r[1]} for r in doc_types],
    }


@app.route('/api/stats', methods=['GET'])
def stats_route():
    now = time.time()
    if _full_stats["data"] and now - _full_stats["ts"] < _FULL_STATS_TTL:
        return jsonify(_full_stats["data"])
    try:
        with _full_stats_lock:
            # Re-check inside the lock: under a traffic spike on a cold cache,
            # only the first request should hit the database. Without this,
            # a thundering herd runs the scan once per concurrent visitor.
            if _full_stats["data"] and time.time() - _full_stats["ts"] < _FULL_STATS_TTL:
                return jsonify(_full_stats["data"])
            data = compute_full_stats()
            _full_stats.update({"ts": time.time(), "data": data})
        return jsonify(data)
    except Exception as e:
        print(f"[stats] refresh failed: {e}")
        # Serve stale rather than erroring — the numbers are cosmetic.
        if _full_stats["data"]:
            return jsonify(_full_stats["data"])
        return jsonify({"error": "stats unavailable"}), 503


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    action = data.get('action')
    text = data.get('text')
    if not text or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "No text provided"}), 400
    # Unbounded text straight into an LLM call was the softest cost-drain
    # target on the API — cap it and rate-limit it like /api/chat.
    text = text[:MAX_ANALYZE_CHARS]
    if not client:
        return jsonify({"error": "LLM client not configured"}), 500

    user = get_user()
    allowed, reason, _ = check_rate_limit(user)
    if not allowed:
        return jsonify({
            "error": "rate_limited",
            "message": "Rate limit reached. Please wait a moment or sign in.",
        }), 429
    if action == 'names':
        prompt = load_prompt('analyze-names.txt').replace('{text}', text)
    elif action == 'summarize':
        prompt = load_prompt('analyze-summarize.txt').replace('{text}', text)
    else:
        return jsonify({"error": "Invalid action"}), 400
    try:
        completion = client.chat.completions.create(
            model=MODEL, **GEN_KW,
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
