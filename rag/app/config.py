"""All environment-driven settings in one place. Importing this module loads
.env and corpus.toml; every other module reads its settings from here."""
import os
import re
import tomllib

from dotenv import load_dotenv

load_dotenv()


def _flag(name, default="0"):
    return os.getenv(name, default).lower() in ("1", "true", "yes")


# --- Credentials --------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- CORS ---------------------------------------------------------------------
# Bare CORS(app) allows every origin, which lets any third-party site drive
# this backend on our LLM budget. Lock to our own domains in production;
# fall back to permissive only when ALLOWED_ORIGINS is unset (local dev).
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# --- Security headers -------------------------------------------------------------
# Content-Security-Policy ships as Report-Only (violations appear in the
# browser console) until CSP_ENFORCE=1, so a missed origin cannot break the
# live site. img-src excludes third-party hosts on purpose: a model-emitted
# markdown image to an attacker URL would otherwise load automatically.
CSP_ENFORCE = _flag("CSP_ENFORCE")
CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",  # export window uses a <style> block
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self' https://*.supabase.co wss://*.supabase.co",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "object-src 'none'",
])

# --- Auth (Supabase JWT) --------------------------------------------------------
# Supabase supports two token-signing schemes and which one a project uses
# depends on when it was created:
#   * Asymmetric (ES256/RS256) — current default. Tokens are verified against
#     the project's public JWKS endpoint. Nothing secret goes in the env.
#   * HS256 — the legacy shared "JWT Secret". Still valid on older projects.
# Support both so this works regardless of project age, and so a future
# key rotation to asymmetric doesn't lock every user out.
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
AUTH_CONFIGURED = bool(SUPABASE_URL or SUPABASE_JWT_SECRET)

# --- Abuse / cost limits --------------------------------------------------------
# TEMPORARILY DISABLED (see admin/todo.md): the whole sign-in + quota gate is off by
# default so the thesis demo runs without a login wall. Set GATE_ENABLED=1 to
# turn the limiter back on; nothing below was removed.
GATE_ENABLED = _flag("GATE_ENABLED")

# Anonymous visitors get a small free trial so media traffic can try the tool
# without signing up; signed-in users get a real daily quota.
ANON_DAILY_LIMIT = int(os.getenv("ANON_DAILY_LIMIT", "3"))
USER_DAILY_LIMIT = int(os.getenv("USER_DAILY_LIMIT", "50"))
BURST_PER_MIN = int(os.getenv("BURST_PER_MIN", "5"))
# Reverse proxies in front of the app that append to X-Forwarded-For. The
# client address is the entry that many hops from the right; anything to its
# left is client-written and must not be trusted. Railway = 1. Set 0 when the
# app is reached directly (no proxy), so the socket address is used.
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "1"))

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

# --- Corpus ---------------------------------------------------------------------
# Everything specific to the archive lives in rag/corpus.toml (see the file
# header). Prompts read it as {{ corpus.* }}; code reads the names below.
CORPUS_FILE = os.getenv("CORPUS_FILE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus.toml"))
with open(CORPUS_FILE, "rb") as _f:
    CORPUS = tomllib.load(_f)
# Interpolated into SQL as an identifier, so it must be a plain one.
PAGES_TABLE = CORPUS["pages_table"]
ENTITIES_TABLE = CORPUS.get("entities_table", f"{PAGES_TABLE}_entities")
for _name in (PAGES_TABLE, ENTITIES_TABLE):
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", _name):
        raise ValueError(f"corpus.toml table names must be plain SQL identifiers, got {_name!r}")
PDF_BASE_URL = CORPUS["pdf_base_url"].rstrip("/")
DOC_ID_PATTERN = re.compile(rf"\b({CORPUS['doc_id_pattern']})\b")

# --- LLMs -----------------------------------------------------------------------
# llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-16 and now
# returns 404 model_not_found; gpt-oss-120b is Groq's recommended replacement.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Judge/router/expansion/citation-verify are cheap, high-volume utility calls
# that emit small structured JSON rather than prose. The 20B model is sized
# for exactly that; the 120B here balloons cost and latency for no measurable
# quality gain. Raise to the 120B via JUDGE_MODEL if quality regresses.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")

# Answer generation (RAG, document, metadata). Was 0.3; the same starter
# question then came back with different framings and citations run to run.
# The archive is fixed, so the same question should get the same answer.
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0"))

# --- Embeddings -----------------------------------------------------------------
# The query must be embedded by the same model that filled the column it is
# compared against:
#   local  -- BAAI/bge-small-en-v1.5 run in-process via fastembed (ONNX, CPU,
#             ~6ms/query, weights baked into the image). Column embedding_bge,
#             filled by backfill_embeddings_bge.py. No API, nothing to expire.
#   openai -- text-embedding-3-small (Matryoshka 512). Column embedding, filled
#             by archive/backfill_embeddings_openai.py. Needs OPENAI_API_KEY.
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").lower()
if EMBED_PROVIDER == "openai":
    EMBED_MODEL, EMBED_DIM, EMBED_COLUMN = "text-embedding-3-small", 512, "embedding"
else:
    EMBED_MODEL, EMBED_DIM, EMBED_COLUMN = "BAAI/bge-small-en-v1.5", 384, "embedding_bge"
FASTEMBED_CACHE_PATH = os.getenv("FASTEMBED_CACHE_PATH", "/app/models")

# --- Database -------------------------------------------------------------------
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "8"))
# How long a request waits for a free pooled connection before failing.
# psycopg2's pool raises at once when exhausted, which made the rate limiter
# fail open under nothing more than concurrent requests.
DB_POOL_WAIT_S = float(os.getenv("DB_POOL_WAIT_S", "5"))
RUN_SCHEMA_BOOTSTRAP = _flag("RUN_SCHEMA_BOOTSTRAP")

# --- Retrieval tuning -----------------------------------------------------------
# Below this many AND-matches, the FTS leg retries with the terms OR'd.
FTS_MIN_HITS = int(os.getenv("FTS_MIN_HITS", "10"))
# Worst-case guard for the unindexed ILIKE fallback. Without it a single
# pathological query can pin a connection for the whole request timeout.
ILIKE_TIMEOUT_MS = int(os.getenv("ILIKE_TIMEOUT_MS", "4000"))
# OpenAI column: ivfflat with lists=200. pgvector's default probes=1 scans one
# list (~0.5% of pages): measured recall@30 vs exact search was 0.55 at
# probes=1, 0.95 at probes=10, for +50ms.
# bge column: HNSW, whose default ef_search=40 is below the 30-row LIMIT plus
# filtering headroom; 100 keeps recall near exact.
IVFFLAT_PROBES = int(os.getenv("IVFFLAT_PROBES", "10"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
# Pages handed to the generator after reranking.
CONTEXT_LIMIT_SIMPLE = 15
CONTEXT_LIMIT_RESEARCH = 20
# Per-request caps on LLM fan-out. The router is only *asked* for 1-3
# keywords; one unbounded reply (or the query-words fallback) turned a single
# request into hundreds of concurrent entity-brief calls.
MAX_SEARCH_TERMS = int(os.getenv("MAX_SEARCH_TERMS", "6"))
MAX_ENTITY_BRIEFS = int(os.getenv("MAX_ENTITY_BRIEFS", "3"))
# Document lookups ("show me 104-...") send whole documents to the model;
# cap the page text so a large record cannot exceed the context window or
# cost a whole document's tokens per request (~120k chars = ~30k tokens).
DOC_CONTEXT_CHARS = int(os.getenv("DOC_CONTEXT_CHARS", "120000"))

# --- Verification ---------------------------------------------------------------
# Per-source text shown to the grounding judge and citation verifier. Pages
# average ~1.5k chars (p90 ~2.6k); 6000 means the judges see the same text
# the generator did for virtually every page. At 1500, 14 of 20 picked pages
# were truncated, the judge could not see the passage an answer quoted, and
# failed it as fabricated -- sending the same question down the expansion
# path on some runs and not others.
JUDGE_SOURCE_CHARS = int(os.getenv("JUDGE_SOURCE_CHARS", "6000"))
# Char budget for the answer as shown to the verifier. Sentences past this are
# left unjudged and keep their citations. Raised from 4000: long research
# answers (source tables) went past it and shipped unverified.
VERIFY_ANSWER_CHARS = int(os.getenv("VERIFY_ANSWER_CHARS", "8000"))

# --- Answer contract --------------------------------------------------------------
# The exact refusal line. Prompts, the out-of-scope route and the citation
# checks all match on this string.
REFUSAL_LINE = "The retrieved documents do not contain sufficient information to answer this query."

# Columns the router may filter on for archive-level (metadata) questions.
METADATA_COLUMNS = {
    "has_redactions", "includes_handwriting", "has_stamps",
    "has_tables", "has_forms", "is_typewritten", "document_type",
}
