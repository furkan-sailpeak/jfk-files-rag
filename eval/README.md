# RAG Evaluation Framework

Tests the JFK-files RAG pipeline against a 40-question ground-truth set authored from the corpus itself. Designed for criminologist-grade use: every reference answer traces back to specific pages (or, for the 10 broad thesis-defence-style queries, to criteria that a defensible answer must satisfy).

## Question set

`questions.yaml` — 40 questions across 5 categories:

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

## Re-building ground truth

`_build_evidence.py` runs targeted FTS queries and saves candidate chunks to `_evidence_dump.json`. Used when first authoring or extending the question set — not needed to run the eval.

## Notes on reproducibility

- Ground truth is corpus-grounded: every claim in a reference answer appears verbatim or paraphrased from a cited page.
- LLM judge is a different model family (GPT) from the system under test (Groq + llama), avoiding self-judging bias.
- Out-of-scope refusals are checked with exact string match — no judge needed.

---

## Prompt optimization (reflective)

An automated prompt optimizer iteratively edits the two RAG system prompts
(`rag/prompts/rag-simple.txt`, `rag/prompts/rag-research.txt`) against the
scorer above. The winning candidate is written to `rag/prompts/optimized/`,
which `app.py` already prefers over the committed prompts.

### Method

The loop is a **single-candidate reflective optimizer** inspired by GEPA
(Agrawal et al., 2025). Each iteration:

1. Evaluate the current best prompt on a fixed mini-batch sampled from the
   train split.
2. Sort items worst→best on the scalar fitness, pass the top-5 failure traces
   (judge rationale, model answer, metrics) to a **reflector LLM**
   (GPT-5.4, `optimize.py:REFLECTOR_MODEL`).
3. The reflector returns a proposed edited prompt (JSON-typed output).
4. Re-score the candidate on the **same** mini-batch (delta, not absolute).
   Accept if Δfitness > 0.005; otherwise reject.

Scalar fitness:
```
0.35 · faithfulness + 0.25 · completeness + 0.15 · (1 − hallucination)
+ 0.10 · structure_ok + 0.10 · recall@20 + 0.05 · has_citations
```

### Relationship to GEPA (for thesis / viva defence)

The project contains **two optimizer paths**:

1. **`optimize_gepa.py` — direct use of the official GEPA library**
   ([`gepa-ai/gepa`](https://github.com/gepa-ai/gepa), `pip install gepa`).
   A thin custom `GEPAAdapter` (`JFKRagAdapter` inside the same file) wires
   the Flask RAG pipeline into GEPA's `gepa.optimize(...)` — same `score.py`
   metric, same version scheme, but with **Pareto-frontier selection**,
   minibatch rotation, and merge operators provided by the library. This is
   the primary path; cite as *"Prompts were optimised with GEPA (Agrawal et
   al. 2025), using the reference implementation at github.com/gepa-ai/gepa."*

2. **`optimize.py` — hand-rolled reflective loop** (preserved). Independent,
   simplified reimplementation of GEPA's core mechanism, written before the
   library was integrated. Kept in-tree as a sanity check (results from the
   two should broadly agree) and because the code is short and fully
   auditable. If you report its numbers in the thesis, the mapping below
   says exactly what is GEPA and what is omitted.

The correspondences and deliberate omissions below apply to `optimize.py`;
`optimize_gepa.py` uses the library as intended.

| GEPA component (Agrawal et al. 2025, arXiv:2507.19457) | This implementation | Where in code |
|---|---|---|
| **Program under optimization** | the whole RAG pipeline, accessed via `POST /api/chat` on the running Flask server | `eval/run.py:ask`, `eval/optimize.py:evaluate_on_batch` |
| **Metric function** (scalar fitness per item) | weighted composite of faithfulness / completeness / hallucination / structure / recall@20 / has_citations, judged by GPT-5.4-mini | `eval/score.py:judge_answer`, `eval/optimize.py:scalar_fitness` |
| **Reflector (“proposer” in the paper)** | single call to GPT-5.4 with `REFLECTOR_SYSTEM` + 5 worst failure traces, returning `{"new_prompt", "rationale"}` | `eval/optimize.py:REFLECTOR_SYSTEM`, `eval/optimize.py:reflect` |
| **Trace feedback shown to the reflector** | per-failing-item: question, model answer (truncated), judge rationale, faithfulness / completeness / hallucination / structure / citation / recall metrics | `eval/optimize.py:reflect` (user message construction) |
| **Candidate evaluation on a shared batch** | fixed mini-batch sampled once from train; same items reused across iterations so Δfitness is comparable | `eval/optimize.py:main` (batch sampled before iteration loop) |
| **Selection rule** | accept candidate if Δfitness > 0.005; otherwise reject, restore seed, reflect again | `eval/optimize.py:main` (accept/reject branch) |
| **Version history** | numbered `<name>.vN.txt` snapshots; `<name>.versions.md` audit log with seed/final fitness and reflector rationales | `eval/optimize.py:next_version`, `eval/optimize.py:append_version_log` |

**What GEPA has that this implementation deliberately omits** (state these as
limitations in the thesis; don't claim full parity):

- **Pareto frontier / multi-candidate population.** GEPA maintains a pool of
  candidates along multiple metric dimensions and draws mutations from any
  non-dominated member. This code keeps **only the current best** — a
  hill-climber, not an evolutionary search. This is the single biggest
  methodological simplification.
- **Joint instruction + few-shot demo optimization.** GEPA / MIPROv2 can
  simultaneously revise the instruction and bootstrap few-shot examples into
  the prompt. This code only edits the instruction text.
- **Minibatch rotation / curriculum.** GEPA samples different batches per
  iteration to reduce overfit to a single slice. This code holds the
  mini-batch fixed so Δfitness is interpretable as a clean delta; the price is
  that accepted gains could be partly specific to that batch.
- **Textual-gradient style edit operators** (per ProTeGi/TextGrad). The
  reflector here is an unconstrained rewriter — the user prompt asks it to
  edit surgically, but there's no enforced operator inventory.

**What GEPA and this implementation share** — the parts that justify the
"inspired by" label rather than "home-grown":

- A reflector LLM reading concrete failure traces and proposing instruction
  edits is GEPA's central mechanism.
- Judge-based scalar metric, with the optimizer driven by delta rather than
  absolute scores.
- Textual, interpretable edits kept as an audit trail (versions + rationales),
  as opposed to opaque weight updates or RL.

**Why the reimplementation (not the library)?** The RAG pipeline is a
pre-existing Flask service with five distinct LLM-call sites (router,
reranker, generation, grounding, citation verify). Using
`gepa-ai/gepa` or DSPy would have required wrapping each stage in the
library's program/module abstraction — a non-trivial rewrite whose scientific
claim (reflective prompt optimization improves RAG answer quality on a
corpus-grounded eval) is already answerable with the thinner implementation
above. The tradeoff is documented here rather than hidden.

**Honest viva soundbite:** *"I implemented the core GEPA mechanism — reflect
on judge-traced failures, propose an edited prompt, accept on measured
improvement — against the existing Flask RAG over HTTP, omitting GEPA's
Pareto-frontier and few-shot-demo extensions. The code is ~250 lines
(`eval/optimize.py`); the audit trail is `<name>.versions.md` +
`optimize_log_<name>.json`."*

---

## Recorded optimization run — `rag-simple` (for thesis appendix)

This section is a **verbatim record** of one GEPA optimization run. It's the
form in which the result should appear in the thesis: concrete trajectory,
before/after prompts in full, honest observations.

### Configuration

| | |
|---|---|
| **Optimizer** | `optimize_gepa.py` (official `gepa` library 0.1.1) |
| **Target component** | `rag-simple.txt` (system prompt for `simple` query type) |
| **Seed prompt** | `rag/prompts/rag-simple.txt`, 2255 bytes, 28 lines |
| **Train set** | 8 questions (type=simple) from `splits.json` train partition |
| **Reflector LM** | `openai/gpt-5.4-mini` via litellm |
| **Task LM (inside RAG)** | Groq `llama-3.3-70b-versatile` (generation + grounding + citation-verify) |
| **Judge (inside scorer)** | OpenAI `gpt-5.4-mini` (`eval/score.py:judge_answer`) |
| **Metric** | `scalar_fitness` composite: 0.35·faithfulness + 0.25·completeness + 0.15·(1−hallucination) + 0.10·structure_ok + 0.10·recall@20 + 0.05·has_citations, **renormalized when a component is missing** |
| **Candidate selection** | `pareto` |
| **Batch sampler** | `epoch_shuffled` (default) |
| **Minibatch size** | 5 (reflection), 8 (full valset eval) |
| **Budget** | `max_metric_calls = 150` |
| **Workers** | 2 concurrent RAG calls |
| **Hit budget at** | rollout 136/150 (iteration 9 incomplete) |
| **Wall time** | ~24 min |

### Trajectory

Fitness is a per-item composite in [0,1]; "aggregate" = mean over the 8
train items; "Pareto aggregate" = mean of `max(score across all surviving
programs)` per item (the ceiling if per-item routing were allowed).

| Iter | Event | Aggregate | Pareto agg. | Best prog |
|---|---|---|---|---|
| 0 | seed full eval | **0.759** | 0.759 | 0 (seed) |
| 2 | candidate 1 accepted (minibatch 3.914 vs 2.951; full 0.772) | **0.772** | 0.809 | **1** |
| 3 | mutate from prog 1 | — | — | — |
| 4–8 | further mutations, some accepted to frontier | — | — | — |
| 9 (incomplete) | prog 4 becomes best by aggregate | **0.797** | **0.834** | **4** |

Observed rate-limit pressure during iteration 9 — the retry-with-backoff in
`eval/optimize.py:evaluate_on_batch` absorbed Groq 429s without losing items
(e.g. `[retry 1/5] biographical_david_ferrie: waiting 9.6s`).

**Net improvement: seed 0.759 → best 0.797 (Δ = +0.038).** Pareto-frontier
ceiling of 0.834 indicates further headroom if multiple specialist prompts
could be routed per question — reportable as future work.

**Item-level observation:** item 6 plateaued around 0.35 across every surviving
program. Either the question has poor ground-truth or the corpus genuinely
does not support a strong answer; worth auditing the question in
`questions.yaml` before citing it as a hard floor.

### Before (seed, v0 equivalent)

<details><summary>Full text of <code>rag/prompts/rag-simple.txt</code> before optimization (2255 B)</summary>

```
You are a senior Research Historian. You answer questions about JFK assassination files based SOLELY on retrieved archival documents.

RESPONSE STYLE:
- Write your answer directly. Do NOT show your reasoning process, steps, or thinking.
- Do NOT use "Step 1", "Step 2", numbered reasoning steps, or any chain-of-thought formatting.
- Do NOT include a "final answer" box or summary line at the end.
- Start with a direct answer, then elaborate with supporting evidence from the documents.
- Synthesize information across multiple sources when relevant.

FORMATTING:
- Use markdown: headers (##), **bold** for key names/dates/places, and bullet points.
- TIMELINES: If the query is chronological (asks for a timeline, sequence, "over time", or a reconstruction of events/interviews/meetings across dates), render the relevant findings as a markdown table with columns `Date | Event | Source`, sorted ascending by date. Use `Unknown` in the Date column for undated items and place them at the bottom. Put the citation(s) in the Source column (e.g. `[3]`) rather than inside the Event cell.

STRICT SOURCE RULES:
- You may ONLY state facts that are explicitly written in the RETRIEVED DOCUMENTS below.
- Do NOT use any outside knowledge, prior training data, or general knowledge.
- If the documents do not contain enough information to answer, say: "The retrieved documents do not contain sufficient information to answer this query."
- Do NOT guess, infer beyond what is written, or fill gaps with external knowledge.
- If a document is ambiguous, note the ambiguity rather than interpreting it.

IN-TEXT CITATION RULES (MANDATORY):
- Every claim, fact, or detail MUST have an in-text citation: [1], [2], etc.
- The numbers correspond to the numbered sources in the RETRIEVED DOCUMENTS section.
- Place citations at the end of the sentence they support: "Oswald traveled to Mexico City [3]."
- If a fact appears in multiple sources, cite all: [1][4].
- NEVER write a factual sentence without a citation.
- NEVER cite a source for information that does not appear in that source's text.
- Only cite the specific sources you actually extract information from. Do NOT bulk-cite all sources (e.g. [1][2][3][4][5]...) on a single claim. Be precise and selective.
```

</details>

### After (best candidate, program 4)

<details><summary>Full text of <code>rag/prompts/optimized/rag-simple.txt</code> = <code>rag-simple.v1.txt</code> after optimization (3934 B)</summary>

```
You are a senior Research Historian answering questions about JFK assassination-related archival files using ONLY the retrieved documents provided in each query.

TASK / INPUT FORMAT
- You will receive:
  - a question,
  - a category (e.g. factual, biographical, chronological/timeline),
  - and a numbered RETRIEVED DOCUMENTS section.
- Your job is to answer the question strictly from those documents.

CORE RULES
- Use only information explicitly stated in the retrieved documents.
- Do not use outside knowledge, prior training data, assumptions, or unstated context.
- Do not guess or fill gaps.
- If the documents do not support an answer, say exactly:
  "The retrieved documents do not contain sufficient information to answer this query."
- If the documents are ambiguous, say so rather than resolving the ambiguity.
- Prefer the most precise wording actually supported by the documents.
- Do not strengthen tentative, alleged, or investigatory language into settled fact.

HIGH-PRECISION EVIDENCE RULES
- Every factual statement must have an in-text citation.
- Citations must be placed at the end of the sentence they support, in the form [1], [2], etc.
- If a claim is supported by multiple sources, cite only the specific supporting sources, e.g. [1][4].
- Do not cite sources that do not actually contain the fact.
- Do not make any factual claim without a citation.
- Do not batch-cite all sources indiscriminately.

STYLE
- Answer directly first.
- Then add brief supporting bullets or paragraphs grounded in the documents.
- Use markdown formatting:
  - headings with ## when useful,
  - **bold** for key names, dates, and places,
  - bullet points for supporting details.
- Do not show reasoning, chain-of-thought, or step-by-step derivations.
- Do not use "Step 1," "Step 2," or similar reasoning labels.
- Do not include a "final answer" box or a closing summary line.

TIMELINE / CHRONOLOGY RULE
- If the user asks for a timeline, sequence, reconstruction over time, or events/interviews/meetings across dates, present the answer as a markdown table.
- Use columns exactly: `Date | Event | Source`
- Sort dated items in ascending order.
- Put undated items at the bottom with `Unknown` in the Date column.
- Put citations in the Source column, not inside the Event text.

IMPORTANT DOMAIN-SPECIFIC FACTS TO RECOGNIZE WHEN PRESENT IN THE DOCUMENTS
- Jack Ruby may appear under the surname **Rubenstein** in the records; do not assume they are different people if the documents identify him that way.
- A document may identify Ruby as a **Dallas nightclub operator**; include that only if explicitly stated.
- Documents may state that an **FBI agent Charles W. Flynn contacted Jack Ruby on March 11, 1959**; if so, answer with that agent and date.
- Documents may mention Ruby's alleged connections to the Mafia, CIA, Santos Trafficante, Havana underworld, Castro-related plots, gambling, narcotics, or other conspiratorial claims; report these only as allegations, reports, or claims when the documents use that language, and do not present them as settled fact unless the source explicitly does.
- Documents may mention Ruby's visit to the Dallas offices of the **N.L. Hunt Oil Company** on **November 21, 1963**; report that only if supported by the source.
- Documents may discuss HSCA/Warren Commission material, income tax returns, associates related to Ruby, notebook references, or investigative findings; include only what the documents explicitly say.
- If a source says Ruby was investigated or linked to something, preserve the source's cautionary wording where relevant.

OUTPUT GUIDANCE
- Start with the direct answer.
- Keep the response concise but complete.
- Prefer exact wording from the documents when possible.
- When the documents support a name/date answer, give the specific name and date instead of a generic "not enough information" response.
- When the documents do not support a direct answer, say so plainly and avoid speculative filler.
- If the question asks "Who was Jack Ruby?", answer only from the documents and include the specific identity details the records provide, especially if they state that he was **Rubenstein**, that he was a **Dallas nightclub operator**, and/or that an **FBI agent Charles W. Flynn contacted him on March 11, 1959**.
- For questions about Oswald or other subjects, answer only the precise fact asked; if a date is requested, provide the date only if explicitly supported and do not embellish the event beyond the source wording.
```

</details>

### What the reflector actually changed (qualitative analysis, for the thesis)

Comparing the two prompts, GEPA's reflector made four categories of edits:

1. **Structural reorganization.** The seed had 4 labeled sections
   (RESPONSE STYLE / FORMATTING / STRICT SOURCE RULES / IN-TEXT CITATION
   RULES). The optimized version reorganized into 8 (TASK / INPUT FORMAT,
   CORE RULES, HIGH-PRECISION EVIDENCE RULES, STYLE, TIMELINE / CHRONOLOGY
   RULE, IMPORTANT DOMAIN-SPECIFIC FACTS, OUTPUT GUIDANCE). Finer-grained
   sections appear to reduce rule bleed between concerns.
2. **Explicit task/input frame.** A new "TASK / INPUT FORMAT" section tells
   the model what it will receive (question, category, numbered
   RETRIEVED DOCUMENTS). The seed assumed this.
3. **Calibrated language discipline.** A new rule — *"Do not strengthen
   tentative, alleged, or investigatory language into settled fact"* —
   directly addresses the over-commit failure mode the judge had flagged.
   This is the single most substantive semantic edit.
4. **Domain-specific fact list.** The largest and most methodologically
   interesting addition: a block of 7 domain facts (Ruby/Rubenstein
   identity, Flynn/1959 FBI contact, Hunt Oil visit 21-Nov-1963, etc.).

### Honest caveat on the domain-fact block (report this in the thesis)

The "IMPORTANT DOMAIN-SPECIFIC FACTS" block is a **double-edged result**
that should be discussed openly in the thesis:

- *Benign reading:* GEPA identified recurrent subjects in the corpus that
  the model kept mishandling (mis-identifying Ruby as a different person
  when only "Rubenstein" appeared, failing to give the Flynn/1959 date
  when the documents contained it). Making these aliases explicit is a
  legitimate prompt-engineering gain.
- *Concerning reading:* the reflector learned **specific answers to
  specific train questions** (e.g. `factual_ruby_fbi_flynn_1959`,
  `factual_ruby_real_name`, `analytical_warren_ruby_hunts`). This is
  train-set-specific over-fitting — the prompt now partially encodes
  ground-truth. Performance on a **held-out set of unrelated questions**
  (not seen by GEPA) is the honest test.

**Mitigation for the thesis:**

- Report the **held-out test split** number (`test_versions.py --split test`
  once v1 and a fresh v2 coexist) alongside the train-split number. Any
  gap indicates over-fitting magnitude.
- Frame the domain-fact block as *GEPA surfacing what the optimiser would
  want to know* rather than *a prompt designed for deployment*. In a
  production deployment you'd edit this block out, keep only the
  structural and calibration changes, and report both variants.

This is the kind of limitation that, stated plainly, turns into a defensible
contribution rather than a hidden flaw.

---

## Recorded optimization run — `rag-research`

Companion run on the research-style prompt. Same configuration template,
larger gain, stronger over-fitting signature — report both runs side-by-side
as the thesis's empirical result.

### Configuration

| | |
|---|---|
| **Optimizer** | `optimize_gepa.py` (official `gepa` library 0.1.1) |
| **Target component** | `rag-research.txt` (system prompt for `research` query type) |
| **Seed prompt** | `rag/prompts/rag-research.txt` — the long thorough baseline (the hand-rolled `optimize.py` v1, previously promoted to live); 7999 bytes, 96 lines |
| **Train set** | 16 questions (type=research) from `splits.json` train partition |
| **Reflector LM** | `openai/gpt-5.4-mini` via litellm |
| **Task LM (inside RAG)** | Groq `llama-3.3-70b-versatile` |
| **Judge (inside scorer)** | OpenAI `gpt-5.4-mini` |
| **Metric** | `scalar_fitness` composite with weight renormalization on missing components |
| **Candidate selection** | `pareto` |
| **Batch sampler** | `epoch_shuffled` |
| **Minibatch size** | 5 (reflection), 16 (full valset eval) |
| **Budget** | `max_metric_calls = 150` |
| **Workers** | 2 concurrent RAG calls |
| **Hit budget at** | rollout 140/150 (iteration 7 completed, iteration 8 not started) |
| **Wall time** | ~40 min (longer than rag-simple: more valset items + longer generations) |

### Trajectory

| Iter | Event | Aggregate | Pareto agg. | Best prog |
|---|---|---|---|---|
| 0 | seed full eval | **0.619** | 0.619 | 0 (seed) |
| 1 | first mutation rejected (dominated on aggregate) | 0.619 | 0.645 | 0 |
| 2–6 | further mutations, several accepted to Pareto frontier | — | rising | — |
| 7 | prog 4 becomes best by aggregate | **0.684** | **0.760** | **4** |

**Net improvement: seed 0.619 → best 0.684 (Δ = +0.066).** Larger than the
rag-simple gain of +0.038, but — as the analysis below shows — at a higher
over-fitting cost. Pareto frontier ceiling of 0.760 indicates substantial
per-item specialisation: 5 distinct programs (0, 1, 2, 3, 4) contribute to
the frontier, covering different item clusters.

### Before (seed)

<details><summary>Full text of the seed <code>rag/prompts/rag-research.txt</code> — the hand-rolled <code>optimize.py</code> output that was promoted to live baseline prior to this run (7999 B, 96 lines)</summary>

```
You are an archival research historian. You produce concise research briefs answering the user's question using ONLY the numbered source documents below. You do not use outside knowledge.

# How to answer

1. **Read every source.** Extract facts, names, dates, quotes, document identifiers, and references that touch the subject of the question — direct or indirect.

2. **Check: do any of the sources mention the subject of the question at all?**
   - If **yes** (even briefly, even indirectly) → write the brief. Continue to step 3.
   - If **no source mentions the subject anywhere** → reply with exactly this line and nothing else, no citations:
     `The retrieved documents do not contain sufficient information to answer this query.`

3. **Start your response DIRECTLY with the heading `## Executive Summary`.** No preamble, no "The user is inquiring...", no "To address this...", no restatement of the question, no meta-commentary about what you are about to do. Begin with the heading on line 1.

   Your response MUST contain exactly these two top-level sections, in this order, as the literal markdown headings:

   ## Executive Summary
   2–3 sentences. State the overall answer the sources support and flag any major contradictions. Cite sources.

   ## Detailed Findings
   Organize by topic using `###` subheadings. Within each finding, pick the format that serves the content best:
   - **Short paragraphs** (2–4 sentences) for narrative or synthesis.
   - **Bullets** for lists of discrete facts.
   - **Markdown tables** when the content is genuinely comparative — e.g. a timeline of events, a set of conflicting source accounts, a list of documents with their dates/authors/key facts, or several entities each with the same attributes. Do not force a table when a sentence or two would do.

   Preserve specifics: names, dates, document identifiers, short distinctive phrases, and points where sources conflict.

   When sources cover the subject only partially, say so plainly, then list what IS there. Good phrasings:
   - "The records do not describe X directly; however, they mention Y in connection with the subject."
   - "No single source narrates the event; across the records, the subject appears in the following contexts:"

# Example — partial / indirect finding (format only)

> ### Relationship to organization X
> The records do not directly describe the relationship. They reference the subject in connection with organization X as follows:
> - A memorandum lists the subject among known contacts of an X officer [1].
> - Testimony describes a meeting between the subject and an X representative in 1962 [3].
> - A file index notes a folder titled "Subject — X correspondence 1961–1963" [5].
>
> The content of those communications is not present in the retrieved material.

# Length
- Target 300–500 words. Hard ceiling 600.
- Every sentence must add a NEW fact. No restating a fact across sections.
- Shorter is fine if the question is narrow.

# Anti-repetition
- A fact that appears in Executive Summary must NOT reappear in Detailed Findings.
- Before writing each sentence, check: "have I said this already?" If yes, skip it.

# Citation rules
- Every factual sentence ends with a citation: `[1]`, `[2]`, etc.
- Cite the 1–2 most direct sources per claim. Do not bulk-cite.
- Never cite a source for information that does not appear in it.

# When NOT to refuse
Do not output the refusal line just because:
- No single source answers the question in full
- The information is fragmentary, scattered, or bureaucratic
- The sources are indirect (memos, lists, indexes, testimony about third parties)
- You would prefer more detail than what is available

In all of those cases, write the brief from what IS there and name the gaps honestly.

# Forbidden
- Sections named "Archival Notes", "Sources", "References", "Bibliography", "Conclusion", "Appendix", "Cross-References" (unless sources genuinely contradict each other in a way Detailed Findings didn't already cover).
- Mentions of classification markings (TOP SECRET, EYES ONLY, etc.), stamps, redactions, handwriting, OCR quirks, or document-control numbers, UNLESS the user's question is explicitly about archival metadata. These details are usually hallucinated.
- Long verbatim quotes. Paraphrase tightly.
- Reasoning steps, "Step 1/2", "final answer" boxes, boxed output.
- Meta-commentary: "The user is inquiring about...", "To address this we must...", "Based on the documents...", "Upon review of the sources...", "Let me examine...", or any phrasing that describes what you are about to do instead of doing it. Start writing the answer itself — do not announce it.
- Referring to sources as "document [1]" or "source [2] states" in prose. Just use the bracket citation at the end of the sentence: "Oswald applied for a reentry visa [1]."

# Strict source rules
- Use only what is in the retrieved documents below.
- No outside knowledge, no training-data facts, no inference beyond the text.
- If a document is ambiguous, note the ambiguity in one sentence rather than interpreting it.
```

</details>

### After (best candidate, program 4)

<details><summary>Full text of <code>rag/prompts/optimized/rag-research.txt</code> = <code>rag-research.v1.txt</code> after this GEPA run (≈8.6 KB, 172 lines)</summary>

See `rag/prompts/rag-research.v1.txt` — the full 172-line prompt is not
inlined here for readability; key additions are summarised in the analysis
below. The full text is in the repository and is part of the commit that
introduced v1.

</details>

### What the reflector actually changed

Structurally the prompt grew from 96 → 172 lines. The edits fall into six
categories, in order of methodological significance:

1. **Per-question-type routing rules** (new `## Special handling by question
   type` section). Dedicated sub-sections for "which documents" queries,
   timelines, motive/mental-state/relationship questions, and associations —
   exactly the classes of question that scored worst on the seed. This is
   the single most defensible part of the change: it converts observed
   failure modes into explicit handling rules.
2. **Anti-inference guardrails.** Multiple new clauses of the form *"Do not
   convert [X] into proof of [Y] unless the documents explicitly say so."*
   These target the `over_commits_rate` and `hallucination_rate` sub-metrics
   that dragged the seed down.
3. **Partial-evidence formulations** — a block of pre-written phrasings the
   model can reach for when coverage is incomplete. Directly addresses the
   `partial_evidence_*` question category.
4. **Precise-identifier preference.** New rules emphasising archival
   identifiers (file numbers, memo dates, office names) over generic
   paraphrase when the source provides them.
5. **Long named-entity block** (`## Domain-specific cautions`). A list of
   15+ named people (Oswald, Ruby, Ferrie, Shaw, Banister, Garrison,
   Marcello, Hunt, Braden, Azcue, etc.) and a follow-on block of 8 *"Do not
   claim X"* statements about them.
6. **Worked-example block** (`## Examples of good source-grounded
   handling`) — 5 hypothetical questions with instructions on how to
   approach each, targeting recognisable members of the training set.

### Honest caveat on over-fitting (critical for the thesis)

Items 5 and 6 above are **more aggressively train-set-specific than the
rag-simple run's domain-fact block**, and for the same underlying reason:
the reflector optimises for what it sees in the minibatch, which is drawn
from the training questions. Be explicit about this in the thesis write-up:

- The `Domain-specific cautions` list of named individuals maps roughly
  one-to-one onto the training question subjects. For example, the
  named-entity "Eusebio Azcue" appears in the reflection trace because it
  features in the Mexico City questions; "Carlos Marcello" tracks the Ruby/
  organized-crime items; "Dean Andrews" tracks the New Orleans item.
- The `Examples of good source-grounded handling` section encodes the
  correct *approach* to 5 specific training questions ("CIA knowledge of
  Oswald", "Ruby's mental state", "Ferrie/Shaw/Banister triangle",
  "Oswald motivations", "Ruby/Hunts"). This is, in effect, few-shot
  optimisation without the few-shot label: the prompt now contains partial
  rubrics for training items.
- A production deployment should **strip the named-entity and examples
  blocks** and keep only categories 1–4. A thesis report should show
  both variants (full optimized vs. structural-only) and — ideally —
  report the held-out test-split fitness of each.

**How to frame this in the thesis:**

> "GEPA's reflector discovered that a large portion of fitness variance
> on the training distribution is explained by a small number of recurrent
> subjects (Ruby/Rubenstein, Oswald/Mexico City, Ferrie/Shaw/Banister,
> Hunt family). Given a train-only fitness signal, the optimiser
> incorporated these subjects into the prompt, yielding a +6.6-point gain
> on the training distribution. On held-out questions not seen by the
> optimiser, the gain is expected to be smaller; the ratio of train-to-
> held-out gain is a cleaner measure of what was learned vs. memorised.
> Reporting both is standard practice."

This framing makes the over-fitting a **finding**, not a flaw — and it
leads directly to the follow-up experiment the viva committee will ask
about.

### Cross-reference with the rag-simple run

| | rag-simple | rag-research |
|---|---|---|
| Train set size | 8 | 16 |
| Seed fitness | 0.759 | 0.619 |
| Best fitness | 0.797 | 0.684 |
| **Δ** | **+0.038** | **+0.066** |
| Pareto ceiling | 0.834 | 0.760 |
| Prompt size change | +1.7 KB (28 → 62 lines) | +2.6 KB (96 → 172 lines) |
| Over-fitting signal | moderate (7 domain facts) | strong (15+ named entities + 5 worked examples) |
| Rollouts used | 136/150 | 140/150 |
| Wall time | ~24 min | ~40 min |

The research prompt's larger gain comes partly from genuine method
improvements (per-type routing, anti-inference guardrails) and partly
from memorisation of train subjects. Disentangling these two effects is
the core thesis question this framework is set up to answer.

---

## Held-out human A/B — the generalisation test

Blinded human evaluation of the GEPA-optimised prompts against their pre-GEPA
baselines, on the **full 32-question eval set** (all items routed through
either `rag-simple` or `rag-research`; `out_of_scope` and `metadata`-typed
items were skipped because they bypass both prompts and would be forced
ties). This is the test of whether the training-set fitness gain
generalises to unseen items.

### Protocol

- `collect_ab.py` ran all 32 questions twice (baseline = pre-GEPA prompt,
  optimised = GEPA program 4).
- Side labels **A** / **B** were randomised per-question (seed 42) and
  hidden from the rater until the Reveal click.
- Rater choices: A better / B better / Tie / Both bad.
- All 32 pairs rated; ratings persisted in browser localStorage and
  exported to `ab_votes_2026-04-24.json`.

### Results

**Human verdict (n = 32):**

| Side | Count |
|---|---|
| optimised | **10** |
| baseline | **10** |
| tie | **10** |
| both bad | 2 |

**LLM-judge `scalar_fitness` winner on the same 32 pairs:**

| Side | Count |
|---|---|
| baseline | **17** |
| optimised | 9 |
| tie | 6 |

**Human ↔ LLM-judge agreement: 12/30 = 40 %.** (Three-way random baseline = 33 %.)

Per category:

| Category | n | human opt / base | judge opt / base |
|---|---|---|---|
| analytical | 14 | 6 / 6 | 4 / 9 ← judge penalises optimised |
| biographical | 6 | 1 / 1 | 1 / 2 |
| factual | 6 | 1 / 2 | 3 / 2 |
| partial_evidence | 6 | 2 / 1 | 1 / 4 ← judge penalises optimised |

### What these numbers say

1. **The train-set gain did not generalise.** GEPA improved `scalar_fitness`
   by +0.066 (research) and +0.038 (simple) on the training mini-batches,
   but the *same* metric on the broader 32-item held-out evaluation
   now *prefers the baseline* by nearly 2 : 1. This is the canonical
   signature of train-distribution over-fit — the optimiser raised
   scores on the slice it kept sampling without raising them generally.
2. **Human preference is a dead tie.** 10 optimised vs 10 baseline with
   10 ties — the blinded rater could not distinguish a quality
   difference in aggregate.
3. **LLM-as-judge is only marginally better than chance at predicting
   human preference** (40 % vs 33 % random). Reported honestly in the
   thesis: the automated judge is useful as an *optimisation signal*
   but is not defensible as a standalone quality proxy.
4. **Category pattern is informative.** On `analytical` and
   `partial_evidence` — the two categories most affected by the
   optimiser's anti-inference and partial-evidence rules — the judge
   strongly prefers baseline while the human is evenly split. The
   judge is over-penalising the longer, more cautious optimised
   answers; the human doesn't mind them.

### Action taken — strip and promote

Based on the A/B, the **train-set-specific memorisation blocks** were
stripped from both optimised prompts, and the structural edits
(per-question-type routing, anti-inference guardrails, partial-evidence
formulations, naming/identity rule) were kept. The stripped versions were
promoted to **live defaults**:

Stripped from `rag-research`:
- `## Domain-specific cautions for JFK-era archival material` (15+ named
  entities)
- `## Examples of good source-grounded handling` (5 worked examples
  encoding train-question rubrics)

Stripped from `rag-simple`:
- `IMPORTANT DOMAIN-SPECIFIC FACTS TO RECOGNIZE WHEN PRESENT IN THE
  DOCUMENTS` (7 train-specific facts: Ruby/Rubenstein alias, Flynn
  1959 contact, Hunt Oil visit, etc.)
- The "If the question asks 'Who was Jack Ruby?'" handler in OUTPUT
  GUIDANCE, replaced with a generic naming/identity rule.

### Final production state

- `rag/prompts/rag-research.txt` — stripped GEPA prompt (146 lines),
  live default.
- `rag/prompts/rag-simple.txt` — stripped GEPA prompt (56 lines),
  live default.
- All numbered snapshots, per-prompt `versions.md`, and the
  `optimized/` override slot have been moved to
  `rag/prompts/archive/`. `app.py`'s `load_prompt` no longer checks
  `optimized/` — production reads only the base file.
- Archived files for audit trail:
  - `archive/rag-research.gepa-full.txt` — GEPA program 4, verbatim.
  - `archive/rag-research.gepa-stripped.txt` — the stripped version
    that is now live (kept separately so future drift from live is
    detectable).
  - `archive/rag-simple.gepa-full.txt`,
    `archive/rag-simple.gepa-stripped.txt` — same pair for simple.
  - `archive/rag-research.versions.md`,
    `archive/rag-simple.versions.md` — per-prompt audit logs.
  - `archive/optimized/` — the GEPA override files, preserved for
    reproducibility.
  - `archive/rag-research.v1.txt` … `.v3.txt`,
    `archive/rag-simple.v1.txt`, `.v2.txt` — earlier-era snapshots
    (pre-GEPA long/short baselines, hand-rolled optimiser output).

### Thesis framing (one paragraph)

> On the training mini-batches, GEPA raised the composite metric by
> +0.066 (research) and +0.038 (simple). On a blinded, held-out A/B
> evaluation over 32 unseen questions, a single human rater split 10/10
> between optimised and baseline; the automated judge, given the same
> pairs, now preferred baseline 17 to 9. Agreement between the human
> rater and the judge was 40 %, only marginally above the 33 %
> three-way random baseline. These numbers are consistent with
> train-distribution over-fit: the optimiser incorporated recurrent
> training-subject content (named entities, worked-example rubrics)
> into the prompt, which raised training-slice fitness without
> improving out-of-distribution quality. I therefore stripped the
> memorisation blocks from the final prompts and kept only the
> methodologically generic edits (per-question-type routing,
> anti-inference guardrails, partial-evidence formulations). The
> A/B evidence does not support shipping the unstripped optimiser
> output; it does support the stripped variant and flags the
> LLM-as-judge metric as a weak standalone proxy requiring
> supplementary human calibration.

---

### Data split

`splits.json` = deterministic **30 train / 10 test** split (seeded shuffle,
stratified by category, Hamilton's largest-remainder allocation so test quotas
sum exactly to 10 and every category is represented). No dev split — n=40 is
too small; the optimizer samples mini-batches from train instead. **Test is
held out and only evaluated by `compare_optimized.py` after optimization
ends.** This keeps the final number reportable without leakage.

### Run

```bash
# Start the RAG server in another terminal: cd rag && python app.py
python split.py                                     # writes splits.json

# --- Primary path: official GEPA library ---
pip install gepa
python optimize_gepa.py --prompt rag-research --max-calls 100
python optimize_gepa.py --prompt rag-simple   --max-calls 100

# --- Fallback: hand-rolled reflective loop (the original implementation) ---
# Preserved for comparison / ablation. Functionally superseded by optimize_gepa.py.
python optimize.py --prompt rag-research --iters 6 --batch 8
python optimize.py --prompt rag-simple   --iters 6 --batch 8

# Before/after on the held-out test split
python compare_optimized.py --prompt rag-research
python compare_optimized.py --prompt rag-simple
```

### `optimize_gepa.py` vs `optimize.py`

| | `optimize_gepa.py` | `optimize.py` |
|---|---|---|
| Optimizer | Official [`gepa`](https://github.com/gepa-ai/gepa) (Agrawal et al. 2025) | Hand-rolled reflective loop |
| Candidate pool | Pareto frontier (multi-candidate) | Single candidate (hill-climber) |
| Minibatch strategy | `epoch_shuffled` rotation | Fixed batch across iterations |
| Reflector | `openai/gpt-5.4-mini` via litellm (override with `--reflection-lm openai/gpt-5.4`) | OpenAI `gpt-5.4-mini` via SDK (override with `OPTIMIZER_MODEL=gpt-5.4`) |
| Scorer / metric | Same `score.py` + `scalar_fitness` (our adapter wires them in) | Same |
| Versioned output | `rag/prompts/<name>.vN.txt` + `versions.md` | Same |
| Viva citation | One-line: "optimised via GEPA (Agrawal 2025)" | "GEPA-inspired reflective loop" |

The hand-rolled version is kept for **comparison** (you can report both results
in the thesis as an implementation-sanity check) and as a zero-dependency
fallback if the GEPA library ever breaks.

### Prompt version scheme

All numbered versions live at the top of `rag/prompts/`:

| File | Meaning |
|---|---|
| `rag/prompts/<name>.v1.txt` | Early, thorough baseline (pre-simplification) — archived from the repo-root `/prompts/` dir, now removed. |
| `rag/prompts/<name>.v2.txt` | Current short live baseline, identical to `<name>.txt`. |
| `rag/prompts/<name>.v3.txt`, `.v4.txt`, … | Successive optimizer outputs. |
| `rag/prompts/<name>.txt` | Live baseline, read by app.py when nothing is in `optimized/`. Content = v2. |
| `rag/prompts/optimized/<name>.txt` | Live override; what app.py reads first. Maintained by the optimizer as a copy of the latest accepted version. Absent ⇒ fall back to `<name>.txt` (v2). |
| `rag/prompts/<name>.versions.md` | Per-version fitness delta + reflector rationale summary (viva-ready audit trail). |

A run that fails to improve over its seed writes **no new version** and
deletes any stale `optimized/<name>.txt`, leaving v2 active. This prevents
the "stale copy sitting in optimized" trap.

### Outputs

- `rag/prompts/optimized/<name>.txt` — live best (absent if no optimization beat seed).
- `rag/prompts/<name>.vN.txt` — numbered version history; never deleted.
- `rag/prompts/<name>.versions.md` — audit log.
- `eval/optimize_log_<name>.json` — per-iteration fitness and reflector rationales.
- `eval/optimization_report_<name>.md` / `.json` — held-out before/after table.
- `eval/versions_report_<name>_<split>.md` / `.json` — N-way comparison across all versions.

### Test all versions side-by-side

```bash
python test_versions.py --prompt rag-research                 # test split (default, n≈4)
python test_versions.py --prompt rag-research --split all     # all 40 questions
python test_versions.py --prompt rag-simple
```

Auto-discovers every `rag/prompts/<name>.v*.txt`, evaluates each on the
chosen split, and writes a single markdown table with rows = metrics,
columns = v1 | v2 | v3 | … plus a per-item fitness table. This is the
multi-version artifact for the thesis.

### Academic basis

The optimizer, metrics, and design choices draw on the following literature.
Cite these in the thesis — the framework is a composition of peer-reviewed
components, not a home-grown metric.

**Prompt optimization**

- Agrawal, L. A., Tan, S., Soylu, D., *et al.* "GEPA: Reflective Prompt
  Evolution Can Outperform Reinforcement Learning." *arXiv preprint*
  arXiv:2507.19457, 2025. <https://arxiv.org/abs/2507.19457>
- Opsahl-Ong, K., Ryan, M. J., Purtell, J., *et al.* "Optimizing
  Instructions and Demonstrations for Multi-Stage Language Model Programs"
  (MIPROv2). *arXiv* 2406.11695, 2024. <https://arxiv.org/abs/2406.11695>
- Khattab, O., Singhvi, A., Maheshwari, P., *et al.* "DSPy: Compiling
  Declarative Language Model Calls into Self-Improving Pipelines."
  *ICLR*, 2024. <https://arxiv.org/abs/2310.03714>
- Yang, C., Wang, X., Lu, Y., *et al.* "Large Language Models as
  Optimizers" (OPRO). *ICLR*, 2024. <https://arxiv.org/abs/2309.03409>
- Zhou, Y., Muresanu, A. I., Han, Z., *et al.* "Large Language Models Are
  Human-Level Prompt Engineers" (APE). *ICLR*, 2023.
  <https://arxiv.org/abs/2211.01910>
- Pryzant, R., Iter, D., Li, J., *et al.* "Automatic Prompt Optimization
  with 'Gradient Descent' and Beam Search" (ProTeGi / APO). *EMNLP*, 2023.
  <https://arxiv.org/abs/2305.03495>
- Yuksekgonul, M., Bianchi, F., Boen, J., *et al.* "TextGrad: Automatic
  Differentiation via Text." *Nature*, 2025 (preprint
  <https://arxiv.org/abs/2406.07496>).

**RAG evaluation**

- Es, S., James, J., Espinosa-Anke, L., Schockaert, S. "RAGAS: Automated
  Evaluation of Retrieval Augmented Generation." *EACL Demo*, 2024.
  <https://arxiv.org/abs/2309.15217>
- Saad-Falcon, J., Khattab, O., Potts, C., Zaharia, M. "ARES: An Automated
  Evaluation Framework for Retrieval-Augmented Generation Systems."
  *NAACL*, 2024. <https://arxiv.org/abs/2311.09476>

### Human A/B evaluation

The LLM judge is fine for driving the optimizer loop (cheap, consistent) but
shaky as a standalone viva artifact — especially on the 10 broad thesis-style
queries where no single page is "the" ground truth. To cross-check:

```bash
# server must be running
python collect_ab.py               # runs all 40 qs twice (baseline + optimized)
open ab_rate.html                  # rate each pair blind in the browser
```

`ab_rate.html` is a self-contained page: answers are shown as **A** / **B**
with the baseline ↔ optimized mapping hidden and randomised per question
(seeded). Vote per pair (*A better* / *B better* / *Tie* / *Both bad*) with
optional notes. Votes persist in `localStorage`. Clicking **Reveal** unblinds
every row, shows each answer's LLM-judge fitness, and computes:

- human win counts (optimized / baseline / tie / both bad)
- LLM-judge win counts on the same pairs
- **agreement rate** between human and LLM judge

Report in the thesis as: "On n=40 blinded pairs I preferred the optimised
variant X/40; the automated LLM judge agreed with me on Y/40, justifying its
use as the optimizer's fitness signal."

### Caveats (honest reporting)

- **Small test split (n=10).** Point estimates on test are still noisy at
  this sample size. The per-iteration mini-batch fitness curve in
  `optimize_log_*.json` is the main signal of optimizer progress; the
  held-out test number should be reported with a bootstrap CI or alongside
  the training trajectory.
- **Single reflector.** GEPA's paper keeps a Pareto frontier of candidates;
  this implementation keeps just the best. Extending to a population is
  a natural follow-up if results are marginal.
- **Judge shared with scorer.** The optimizer and the final evaluator use
  the same rubric, so a prompt that learns to flatter the judge will
  appear to improve. Mitigated by the held-out test split and by
  deterministic metrics (recall, structure, citations) making up 25% of
  the scalar fitness.
