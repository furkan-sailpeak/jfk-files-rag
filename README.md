# JFK Files Research System

A Retrieval-Augmented Generation (RAG) system for querying the 2025 declassified JFK assassination record release, built for the Master of Statistics & Data Science thesis at KU Leuven.

> *"Topic Modeling and Thematic Analysis of JFK Assassination Files Using NLP"*

Live: **https://jfk-files.org** (also served at https://jfk-files-rag-production.up.railway.app/)
Thesis repo: https://github.com/furkan-sailpeak/jfk-thesis

![JFK Files Research System](admin/screenshot.png)

## What it does

- **Grounded Q&A** over ~84,000 OCR-processed pages (2,566 documents). Every factual claim carries a `[N]` citation linked to the exact page in the National Archives PDF.
- **Two answer modes** chosen by a router: *simple* (bold lead + bulleted facts) for single-fact questions, *research* (Executive Summary + Detailed Findings) for analytical ones. Timelines and document comparisons come out as tables with a cited row per source.
- **Conversational follow-ups**: pronouns and references ("what about his wife?", "tell me more") are resolved against the last few turns before retrieval.
- **Document lookup**: a record number such as `104-10004-10143` fetches and summarises that whole document.
- **Archive-level questions**: "how many pages have handwriting?" returns exact counts plus cited sample pages.
- **Honest refusals**: questions outside the archive get a fixed refusal line instead of an invented answer.

## Architecture

**Backend:** Python 3.11 · Flask (gunicorn + gevent) · PostgreSQL 15 + pgvector 0.8 (Supabase) · Groq (`gpt-oss-120b` for answers and reranking, `gpt-oss-20b` for routing and checks) · local `BAAI/bge-small-en-v1.5` embeddings (fastembed) · Pydantic · Jinja2.
**Frontend:** React + Vite · Server-Sent Events.
**Deployment:** Docker (Railway in production, docker-compose locally).

### Request pipeline

```
User query
   │
   ├── contains a record number (104-10004-10143) ──► fetch every page of that document, answer from it
   ▼
Router (gpt-oss-20b, JSON)        type ∈ {simple, research, metadata, conversational, out_of_scope};
   │                              rewrites pronouns from history; extracts 1–3 search keywords
   ├── metadata ──────► exact counts from the DB + sample pages
   ├── out_of_scope ──► fixed refusal line
   ├── conversational ► free chat, no retrieval
   ▼
Hybrid retrieval                  full-text (tsvector, keywords; ANDed, OR fallback when thin)
   │                              ∪ vector (bge-small, halfvec(384), HNSW) on the rewritten question
   ▼
Rerank (gpt-oss-120b, strict JSON schema)  keeps the 15 / 20 most substantive pages
   ▼
Generate (gpt-oss-120b, temperature 0)     sources wrapped in <source id="N"> blocks;
   │                                       format reminder at the end of the prompt
   ▼
Grounding check (gpt-oss-20b)     on-subject and supported? if not: expand the query, re-retrieve, regenerate
   ▼
Citation verification (gpt-oss-20b)        per-sentence check of every [N]; a citation on the wrong source
   │                                       is re-pointed to the source that does contain the claim
   ▼
Renumber citations 1..M, stream the final answer (SSE: stage → token → done)
```

The answer is generated in full and checked before anything is streamed, so the user never sees text that later changes.

### Design notes

- **Why hybrid retrieval.** FTS alone missed semantic questions: the pages where Ruby shoots Oswald say *"fired … fatal shot"*, not "killed". The vector leg recovers those; FTS keeps proper-noun precision.
- **Local embeddings.** Queries are embedded in-process with the same model that embedded the corpus, so there is no embedding API to pay for or run out of. The model is baked into the Docker image.
- **One source layout for every model.** The generator, the no-citation retry, the grounding judge and the citation verifier all see identical `<source id="N">` blocks, so a citation means the same thing at every stage.
- **Format reminder at the tail.** Repeating the output format at the end of the prompt halves non-standard citation brackets and keeps simple answers to 2–5 bullets (tested with gpt-oss-120b).
- **Lenient verifier.** A wrongly deleted citation is worse for a researcher than a slightly imprecise one, so the verifier only flags clear mismatches, and re-points rather than deletes where it can.

## Corpus config

The pipeline is not tied to the JFK files. Everything specific to one document collection lives in **`rag/corpora/<name>.toml`**, selected with `CORPUS=<name>` (default `jfk`):

| Section | Used by | Holds |
|---|---|---|
| `[database]` | backend | page table name, metadata columns the router may filter on |
| `[documents]` | backend | document-ID regex (triggers whole-document lookups), an example ID, source URL template (`{file_id}`) |
| `[assistant]`, `[collection]` | prompts | assistant name and persona, how the collection is described |
| `[scope]` | router prompt | what is in scope vs. out of scope, metadata questions |
| `[examples]`, `[entities]` | router, entity-brief prompts | domain examples for keyword extraction, follow-up resolution and background identifications |

The prompts in `rag/prompts/*.j2` are generic Jinja2 templates that read these values as `{{ corpus.* }}`; with the JFK corpus they render exactly the prompts used before the split. The corpus file is validated at startup (missing keys, non-identifier table/column names, an ID regex without one capture group, a URL without `{file_id}` all fail with a clear error), and `rag/tests/test_corpus.py` renders every prompt with an unrelated placeholder corpus to check no JFK text is hard-coded.

To add a corpus: copy `rag/corpora/jfk.toml`, change every value, load the pages into the table it names, and start the app with `CORPUS=<name>`. Not covered yet: the React frontend (titles, example questions, PDF links), `/api/stats` (expects the JFK metadata columns), the embedding backfill script and `rag/prompt_check.py` (JFK test questions).

## Setup

### Prerequisites

- Docker (simplest), or Python 3.11+ and Node.js 20+
- PostgreSQL 15+ with `pgvector ≥ 0.8`
- A [Groq](https://console.groq.com/) API key

```bash
cp .env.example .env   # then fill in DATABASE_URL, GROQ_API_KEY, VITE_SUPABASE_*
```

### Run with Docker

```bash
make up      # build + start on http://localhost:8080 (then start/stop from Docker Desktop)
make logs
make down
```

### Local development without Docker

```bash
make install                 # .venv with app + dev dependencies

# Backend (serves on :5001)
cd rag && ../.venv/bin/python -m app.main

# Frontend, separate terminal (Vite proxies /api → :5001)
cd rag/frontend && npm install && npm run dev
```

### Checks

```bash
make hooks         # install the git pre-commit hook
make check         # pre-commit (ruff, whitespace, yaml…) + pytest
make test          # offline tests only: LLM, embeddings and DB are mocked
make prompt-check  # live: every LLM stage 3× on fixed questions, reports contract + run-to-run consistency (needs `make up`)
```

### Evaluation (LangSmith)

Datasets live in `rag/eval/datasets/` as `dataset_{project}_{YYYY-MM-DD}.json` (the current one: `dataset_jfk_2026-11-22.json`, 120 plain-language questions). Every dataset has the same fields:

| Field | Meaning |
|---|---|
| `id` | `q001`… (assigned on save) |
| `question` | what a user would type |
| `category` | fact, person, explanation, timeline, not_in_files, archive_stats, off_topic, chit_chat |
| `ground_truth_answer` | the reference answer, or the expected behaviour |
| `expected_sources` | pages the answer should cite: `[{"file": "104-10218-10032.pdf", "page": 165}]`; empty when no citation is expected |
| `validated` | a person has checked the answer and opened the sources |

```bash
make eval-install    # streamlit + langsmith into .venv
make eval-app        # editor: pick or create a dataset; edits save to the file, outside edits appear within ~2s;
                     # the review panel links every expected source straight to its page in the NARA PDF
make eval-sync       # push the newest dataset to LangSmith as "{project}-{date}" (matched on id)
make eval            # sync, then run every question through the pipeline as a LangSmith experiment
make eval ARGS="--dataset dataset_jfk_2026-11-22 --validated-only --category fact --description 'new reranker prompt'"
```

Needs `LANGSMITH_API_KEY` in `.env` plus the app's own keys (the pipeline runs in-process: no server, no rate limiter). Evaluators: `correctness` (yes/no LLM judge vs. the ground truth; Groq `gpt-oss-20b` by default, `EVAL_JUDGE_PROVIDER=ollama|openai` and `EVAL_JUDGE_MODEL` to switch; `--no-judge` skips it), `source_recall` (share of the expected files the answer actually cites with [N]), `retrieval_recall` (share of the expected files among all retrieved pages; high retrieval + low source recall = found but not cited), `refusal_correct` (exact refusal line on off_topic only) and `has_citations` (document-answer categories). Experiments are named after the git commit (`rag-<sha>[-dirty]`), so runs before and after a change line up in LangSmith's comparison view.

### Embedding backfill (one-time)

After loading `jfk_pages` from OCR, embed every page so the vector leg has data. A GPU makes this ~20 min instead of ~2.5 h (Apple MPS or CUDA is picked up automatically):

```bash
pip install sentence-transformers psycopg2-binary python-dotenv
python rag/backfill_embeddings_bge.py          # idempotent; resumable
python rag/backfill_embeddings_bge.py --index  # build the HNSW index once
```

The earlier OpenAI embeddings are still in the `embedding` column (`EMBED_PROVIDER=openai` switches back); their backfill script is in `rag/archive/`.

## Deployment and releases

| Branch | Environment | URL |
|---|---|---|
| `main` | production (Railway) | https://jfk-files.org |
| `dev` | staging (Railway) | the staging service's Railway domain |

- **CI** (`.github/workflows/ci.yml`) runs ruff + pytest and a frontend build on every push and PR to `dev`/`main`. Railway's "Wait for CI" holds a deploy until it passes.
- **Releases** (`.github/workflows/release.yml`): every green push to `main` (= a production deploy) gets a GitHub release, tagged by date (`v2026.10.19`, then `v2026.10.19.2` for a second release that day), with notes generated from the commits and PRs since the previous release.
- Ship to production by merging `dev` into `main` (via PR).

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat` | POST | Answer a question; `{query, history}` in, SSE stream out |
| `/api/stats` | GET | Archive statistics (cached) |
| `/api/analyze` | POST | List the names in, or summarise, a piece of text |
| `/api/pdf/<filename>` | GET | Redirect to the NARA PDF |

## Project layout

```
.
├── rag/
│   ├── app/                     # backend package — read in this order:
│   │   ├── main.py              #   Flask routes
│   │   ├── pipeline.py          #   /api/chat: one function per answer path
│   │   ├── retrieval.py         #   hybrid search, rerank, query expansion, direct lookups
│   │   ├── generation.py        #   router, prompt building, answer generation, citation retry
│   │   ├── verification.py      #   grounding judge, citation verifier
│   │   ├── postprocess.py       #   citation syntax, repair, renumbering (pure functions)
│   │   ├── schemas.py           #   pydantic models for every LLM JSON reply
│   │   ├── llm.py · prompts.py  #   Groq calls; Jinja2 prompt rendering
│   │   └── config.py · db.py · auth.py
│   ├── prompts/                 # generic prompt templates (Jinja2, *.j2)
│   ├── corpora/                 # corpus configs (jfk.toml): everything collection-specific
│   ├── tests/                   # offline pytest suite, one file per module
│   ├── frontend/                # React UI
│   ├── backfill_embeddings_bge.py
│   ├── prompt_check.py          # live prompt consistency check
│   └── Dockerfile
├── admin/                       # architecture diagram, screenshot; database/ and todo.md are local-only
├── docker-compose.yml · Makefile · pyproject.toml · .pre-commit-config.yaml
└── .env.example
```

`admin/database/` (OCR ingestion scripts) is kept locally but not tracked. The old thesis evaluation harness (`eval/`, GEPA prompt-optimisation runs) and the archived prompt versions (`rag/prompts/archive/`) were removed in favour of `rag/eval/`; both are in the git history (harness up to commit `42b6291`).

See [admin/rag-architecture-diagram.html](admin/rag-architecture-diagram.html) for the architecture diagram.

## License

© 2026 Furkan Demir · KU Leuven · All rights reserved · For academic research purposes only.

Academic project, KU Leuven. The JFK records are public domain (NARA, 2025 release). See [LICENSE](LICENSE) for details.
