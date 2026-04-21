# RAG Evaluation Framework

Tests the JFK-files RAG pipeline against a 30-question ground-truth set authored from the corpus itself. Designed for criminologist-grade use: every reference answer traces back to specific pages.

## Question set

`questions.yaml` — 30 questions across 5 categories (6 each):

| Category | Tests |
|---|---|
| `factual` | single-fact verifiability (dates, names, identifiers) |
| `biographical` | "who is X?" summaries — exercises the retrieval fix that surfaces introductory chunks |
| `analytical` | multi-source synthesis, research mode |
| `partial_evidence` | corpus covers the question only partially; answer must hedge |
| `out_of_scope` | unrelated to the archive; system must emit the refusal line |

Each entry has `reference_evidence` (filename + page tuples pulled from actual corpus) and a `reference_answer` authored **only** from the text of those pages.

## Metrics

**Deterministic (no LLM):**
- `evidence_recall@20` — fraction of reference-evidence pages that appear in the system's reranked top-20
- `evidence_precision` — fraction of system sources that overlap reference evidence
- `structure_ok` — simple queries follow bold+bullets shape; research queries have the two required headings
- `has_citations`, `no_bulk_cite` — citation discipline

**LLM-as-judge (gpt-5.4-mini):**
- `faithfulness` (0–1) — are cited claims actually supported by the named sources?
- `completeness` (0–1) — coverage of reference key facts
- `hallucination` (bool) — any substantive claim not traceable to retrieved sources
- `over_commits` (bool) — states contested things as settled fact without hedging
- `clarity` (1–5) — readability for a criminologist reader

**Refusal (deterministic):**
- `correct_refusal` — for `out_of_scope`, did the system emit the exact refusal line?

## Run

```bash
# 1. Start the RAG server in another terminal
cd ../rag && python app.py

# 2. Run every question and collect answers
cd ../eval
python run.py                   # writes results.json
# or: python run.py --limit 5 --only factual   for a smoke test

# 3. Score results (needs OPENAI_API_KEY)
python score.py                 # writes scores.json and report.md
```

## Outputs

- `results.json` — raw system output per question (answer + sources + timings)
- `scores.json` — per-question metric values
- `report.md` — human-readable summary table + per-question detail

## Re-building ground truth

`_build_evidence.py` runs targeted FTS queries and saves candidate chunks to `_evidence_dump.json`. Used when first authoring or extending the question set — not needed to run the eval.

## Notes on reproducibility

- Ground truth is corpus-grounded: every claim in a reference answer appears verbatim or paraphrased from a cited page.
- LLM judge is a different model family (GPT) from the system under test (Groq + llama), avoiding self-judging bias.
- Out-of-scope refusals are checked with exact string match — no judge needed.
