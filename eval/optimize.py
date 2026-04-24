"""
GEPA-inspired reflective prompt optimizer for the two RAG system prompts
(rag-simple.txt, rag-research.txt).

Each iteration:
  1. Evaluate the current best prompt on a random mini-batch from the train split.
  2. Collect worst-scoring items + their judge rationales (the "reflection trace").
  3. Ask a reflector LLM to read the trace and propose an edited prompt.
  4. Evaluate the candidate on the same mini-batch (hold seed fixed for fair
     comparison). Keep it if fitness improves.

At the end the best prompt is written to rag/prompts/optimized/<name>.txt, which
app.py already prefers over rag/prompts/<name>.txt (see load_prompt).

The test split is held out and used only by the final before/after comparison.

Usage:
    # start the RAG server first:  cd rag && python app.py
    python optimize.py --prompt rag-research --iters 8 --batch 10
    python optimize.py --prompt rag-simple --iters 8 --batch 10

Academic basis: GEPA (Agrawal et al. 2025, arXiv:2507.19457) — reflective
evolutionary prompt optimization; more sample-efficient than MIPROv2 and
RL fine-tuning. This file is a lightweight single-candidate variant of that
idea adapted to a Flask-wrapped RAG pipeline.
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")
sys.path.insert(0, str(HERE))

# Reuse the existing scorer — one judge stack for the whole project.
from run import ask as rag_ask  # noqa: E402
from score import score_one  # noqa: E402

QUESTIONS_PATH = HERE / "questions.yaml"
SPLITS_PATH = HERE / "splits.json"
PROMPTS_SRC = HERE.parent / "rag" / "prompts"
PROMPTS_OPTIMIZED = PROMPTS_SRC / "optimized"
PROMPTS_OPTIMIZED.mkdir(parents=True, exist_ok=True)

# Reflector = the LLM that proposes edited prompts. gpt-5.4-mini is ~10x
# cheaper than gpt-5.4 and empirically sufficient for prompt-editing
# reflection. Any OpenAI model that supports JSON mode and ≥8K completion
# tokens works. Override via OPTIMIZER_MODEL if you want gpt-5.4.
REFLECTOR_MODEL = os.getenv("OPTIMIZER_MODEL", "gpt-5.4-mini")
reflector_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Which query_type each prompt file serves. Mini-batches only draw from the
# matching set so we don't score a simple prompt on research questions.
PROMPT_TO_QUERY_TYPE = {
    "rag-simple": "simple",
    "rag-research": "research",
}


# ---------- fitness ----------

def scalar_fitness(scored_item):
    """Weighted scalar from a score.py output, with **weight renormalization**
    when a component is missing rather than substituting 0 or 0.5.

    Why: previously `recall@20` was forced to 0 for any question with empty
    `reference_evidence` (the 10 broad thesis-style queries), and
    `completeness` was forced to 0.5 when the judge didn't return it. Both
    silently biased the composite downward on those items. The fix is to skip
    the missing component and renormalize the remaining weights to sum to 1 —
    so the composite reflects only the metrics that are actually measurable
    for that item.

    Missing-component policy:
    - Hard failure (`error` or `judge_error`): return 0.0 (question couldn't
      be scored at all).
    - Judge sub-metrics missing (None): skip, renormalize.
    - `recall@20` None (no `reference_evidence` on the question): skip,
      renormalize. Valid for the 10 broad queries where there is no single
      ground-truth page set.
    """
    if scored_item.get("error") or scored_item.get("judge_error"):
        return 0.0
    j = scored_item.get("judge") or {}
    ev = scored_item.get("evidence") or {}

    components = []  # list of (weight, value)

    faith = j.get("faithfulness")
    if faith is not None:
        components.append((0.35, float(faith)))

    compl = j.get("completeness")
    if compl is not None:
        components.append((0.25, float(compl)))

    hall = j.get("hallucination")
    if hall is not None:
        hall_pen = 0.0 if hall else 1.0
        components.append((0.15, hall_pen))

    # Deterministic — always present (produced by score.py for every item)
    components.append((0.10, 1.0 if scored_item.get("structure_ok") else 0.0))
    components.append((0.05, 1.0 if scored_item.get("has_citations") else 0.0))

    recall = ev.get("evidence_recall_at_k")
    if recall is not None:
        components.append((0.10, float(recall)))

    if not components:
        return 0.0
    total_weight = sum(w for w, _ in components)
    return sum(w * v for w, v in components) / total_weight


# ---------- eval loop ----------

def write_prompt(prompt_name, text):
    """Write a prompt candidate into the optimized/ dir that app.py reads."""
    (PROMPTS_OPTIMIZED / f"{prompt_name}.txt").write_text(text)


def clear_optimized(prompt_name):
    p = PROMPTS_OPTIMIZED / f"{prompt_name}.txt"
    if p.exists():
        p.unlink()


def next_version(prompt_name):
    """All numbered versions live in rag/prompts/<name>.vN.txt. v1 = original
    (longer) prompt lifted from the repo-root /prompts dir; v2 = the shorter
    live baseline; v3+ = optimizer outputs. Returns the next free N."""
    pat = re.compile(rf"^{re.escape(prompt_name)}\.v(\d+)\.txt$")
    used = []
    for p in PROMPTS_SRC.glob(f"{prompt_name}.v*.txt"):
        m = pat.match(p.name)
        if m:
            used.append(int(m.group(1)))
    return max(used, default=0) + 1


def append_version_log(prompt_name, version, entry):
    """One-line-per-version markdown log for audit/viva trace."""
    log_path = PROMPTS_SRC / f"{prompt_name}.versions.md"
    if not log_path.exists():
        log_path.write_text(
            f"# Prompt version history — `{prompt_name}`\n\n"
            f"- **v1** = early thorough baseline (pre-simplification), archived in `{prompt_name}.v1.txt`.\n"
            f"- **v2** = current short live baseline, archived in `{prompt_name}.v2.txt`; identical to `{prompt_name}.txt`.\n"
            f"- **v3+** = optimizer outputs.\n\n"
            f"The live prompt read by app.py is whichever version sits in `rag/prompts/optimized/{prompt_name}.txt`; if that file is absent, app.py falls back to `rag/prompts/{prompt_name}.txt`.\n\n"
            f"| Version | Timestamp | Seed fitness | Final fitness | Δ | Iters | Notes |\n"
            f"|---|---|---|---|---|---|---|\n"
        )
    row = (
        f"| v{version} | {entry['timestamp']} | {entry['seed_fitness']:.3f} | "
        f"{entry['final_fitness']:.3f} | {entry['delta']:+.3f} | {entry['iters']} | "
        f"{entry['notes']} |\n"
    )
    with log_path.open("a") as f:
        f.write(row)


def evaluate_on_batch(questions, workers=4, max_retries=5, base_delay=2.0):
    """Run RAG on each question, then score. Returns list of scored items and
    the per-item RAG responses so the reflector can see actual outputs.

    Retries transient failures — most importantly Groq 429 (tokens-per-minute
    limit). Without this the optimizer silently scores failed calls as 0,
    which tanks mini-batch fitness and makes iterations reject good candidates.

    app.py re-reads prompt files per-request, so no server restart needed. But
    the caller must ensure the prompt file is stable during the batch.
    """

    def looks_like_rate_limit(msg):
        m = (msg or "").lower()
        return "429" in m or "rate_limit" in m or "too many requests" in m or "tokens per minute" in m

    def run_one(q):
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = rag_ask(q["question"])
            except Exception as e:
                resp = None
                last_err = str(e)
            if resp is not None and "_error" not in resp:
                return q, {
                    "id": q["id"],
                    "question": q["question"],
                    "category": q["category"],
                    "query_type_seen": resp.get("query_type"),
                    "answer": resp.get("answer", ""),
                    "sources": resp.get("sources", []),
                    "timings": resp.get("timings", {}),
                    "elapsed_s": 0,
                }
            # Extract a reason, prefer the SSE error message over the transport error.
            if resp is not None and "_error" in resp:
                last_err = resp["_error"]
            last_err = last_err or "no_done_event"
            # Rate-limit errors get longer waits (Groq TPM windows reset at ~60s);
            # other transient errors use shorter exponential backoff.
            if looks_like_rate_limit(last_err):
                delay = min(60, 8.0 * (2 ** attempt)) + random.uniform(0, 3.0)
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1.5)
            if attempt < max_retries - 1:
                print(f"  [retry {attempt+1}/{max_retries}] {q['id']}: "
                      f"waiting {delay:.1f}s ({last_err[:100]})")
                time.sleep(delay)
        return q, {"id": q["id"], "error": f"retries_exhausted: {last_err}", "elapsed_s": 0}

    pairs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one, q) for q in questions]
        for fut in as_completed(futures):
            pairs.append(fut.result())

    # restore ordering for readability
    order = {q["id"]: i for i, q in enumerate(questions)}
    pairs.sort(key=lambda p: order[p[0]["id"]])

    scored = []
    for q, result in pairs:
        try:
            scored.append(score_one(q, result))
        except Exception as e:
            scored.append({"id": q["id"], "error": f"score_failed: {e}"})
    return scored, [p[1] for p in pairs]


def batch_fitness(scored):
    if not scored:
        return 0.0
    return sum(scalar_fitness(s) for s in scored) / len(scored)


# ---------- reflector ----------

REFLECTOR_SYSTEM = """You are a prompt engineer tuning the SYSTEM PROMPT of a retrieval-augmented generation (RAG) model for archival JFK-assassination documents.

You will receive:
  1. The CURRENT SYSTEM PROMPT.
  2. A set of FAILURE TRACES — questions on which the RAG system under this prompt scored poorly, each annotated with judge rationale, model answer, and metrics.

Your job: propose an IMPROVED system prompt that addresses the failures WITHOUT losing strengths. Keep the same overall structure (headers, sections, citation rules) — edit surgically. Never remove strict-source / citation rules. Never add instructions that could cause the model to hallucinate or pull in outside knowledge.

Return STRICT JSON ONLY with exactly these two keys:
{"new_prompt": "<full revised prompt text, as a single JSON string>", "rationale": "<one sentence describing the change>"}

JSON FORMATTING RULES (violating any of these makes the response invalid):
- The value of `new_prompt` MUST be a single JSON string delimited by double quotes only: "...".
- NEVER use Python-style triple-quote strings (\"\"\" or ''') — JSON does not support them.
- Inside `new_prompt`, escape every newline as \\n, every double-quote as \\", and every backslash as \\\\.
- Emit no prose, no code fences, no trailing commas — only the JSON object."""


def reflect(current_prompt, failures, prompt_name):
    """Ask the reflector to propose an edited prompt given failures."""
    trace = []
    for s in failures[:5]:
        j = s.get("judge") or {}
        rationale = j.get("rationale", "—")
        faith = j.get("faithfulness")
        compl = j.get("completeness")
        hall = j.get("hallucination")
        trace.append(
            f"[{s['id']}] ({s['category']}) fitness={scalar_fitness(s):.2f}\n"
            f"Q: {s.get('question','')}\n"
            f"Model answer (truncated): {(s.get('answer','') or '')[:600]}\n"
            f"Judge: faithfulness={faith} completeness={compl} hallucination={hall}\n"
            f"Judge rationale: {rationale}\n"
            f"Structure OK: {s.get('structure_ok')}  Cites: {s.get('has_citations')}  "
            f"Evidence recall: {(s.get('evidence') or {}).get('evidence_recall_at_k')}"
        )

    user = (
        f"CURRENT SYSTEM PROMPT ({prompt_name}.txt):\n"
        f'"""\n{current_prompt}\n"""\n\n'
        f"FAILURE TRACES (worst {len(failures[:5])} items in the last evaluation):\n\n"
        + "\n\n---\n\n".join(trace)
    )

    # GPT-5.4 family uses `max_completion_tokens`, not `max_tokens` (renamed in the
    # GPT-5 series). Bumped high because the reflector regenerates a full ~4KB
    # prompt as a single JSON string value; under-sized caps truncate mid-string.
    resp = reflector_client.chat.completions.create(
        model=REFLECTOR_MODEL,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=8000,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    # Defensive parse: if the model still emitted triple-quote Python strings
    # or other JSON-invalid forms, try to rescue by regex extraction before
    # giving up.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        import re as _re
        m = _re.search(r'"new_prompt"\s*:\s*"""(.*?)"""\s*[,}]', raw, _re.DOTALL)
        if not m:
            raise
        new_prompt = m.group(1).strip()
        rm = _re.search(r'"rationale"\s*:\s*"([^"]*)"', raw)
        rationale = rm.group(1) if rm else "(recovered from malformed JSON)"
        return new_prompt, rationale
    return data.get("new_prompt", "").strip(), data.get("rationale", "")


# ---------- main ----------

def load_pool(prompt_name):
    """Prefer any existing optimized/<name>.txt as the seed so reruns build on
    prior progress; otherwise fall back to the committed rag/prompts/<name>.txt."""
    optimized = PROMPTS_OPTIMIZED / f"{prompt_name}.txt"
    base = PROMPTS_SRC / f"{prompt_name}.txt"
    seed_path = optimized if optimized.exists() else base
    return seed_path.read_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, choices=list(PROMPT_TO_QUERY_TYPE.keys()))
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--reset", action="store_true",
                    help="ignore any existing optimized/ candidate; start from rag/prompts/")
    args = ap.parse_args()

    random.seed(args.seed)

    target_type = PROMPT_TO_QUERY_TYPE[args.prompt]
    splits = json.loads(SPLITS_PATH.read_text())
    questions = {q["id"]: q for q in yaml.safe_load(QUESTIONS_PATH.read_text())}

    # Only questions of the matching query_type; out_of_scope never goes through
    # either RAG prompt, so exclude it from prompt-level tuning.
    train_pool = [
        questions[qid] for qid in splits["train"]
        if questions[qid].get("query_type_expected") == target_type
    ]
    if len(train_pool) < args.batch:
        print(f"WARN: only {len(train_pool)} train items match type={target_type}; "
              f"dropping batch size to {len(train_pool)}")
        args.batch = max(1, len(train_pool))

    if args.reset:
        clear_optimized(args.prompt)
    seed_text = load_pool(args.prompt)
    best_text = seed_text
    write_prompt(args.prompt, best_text)

    # Baseline eval on a fixed mini-batch (so later candidates are compared
    # on the same items — fitness deltas are the signal we trust).
    batch = random.sample(train_pool, args.batch)
    print(f"[seed] evaluating baseline on {len(batch)} items...")
    t0 = time.time()
    seed_scored, _ = evaluate_on_batch(batch, workers=args.workers)
    seed_fit = batch_fitness(seed_scored)
    print(f"[seed] fitness={seed_fit:.3f}  ({time.time()-t0:.0f}s)")

    best_fit = seed_fit
    history = [{"iter": 0, "fitness": seed_fit, "rationale": "(seed)", "prompt_len": len(best_text)}]

    for it in range(1, args.iters + 1):
        # Order failures worst → best for the reflector
        failures = sorted(seed_scored, key=scalar_fitness)
        try:
            new_text, rationale = reflect(best_text, failures, args.prompt)
        except Exception as e:
            print(f"[iter {it}] reflector failed: {e}; skipping")
            continue
        if not new_text or len(new_text) < 100:
            print(f"[iter {it}] reflector returned empty/short prompt; skipping")
            continue

        print(f"[iter {it}] candidate rationale: {rationale[:150]}")
        write_prompt(args.prompt, new_text)
        t0 = time.time()
        cand_scored, _ = evaluate_on_batch(batch, workers=args.workers)
        cand_fit = batch_fitness(cand_scored)
        print(f"[iter {it}] candidate fitness={cand_fit:.3f}  ({time.time()-t0:.0f}s)")

        if cand_fit > best_fit + 0.005:
            best_fit = cand_fit
            best_text = new_text
            seed_scored = cand_scored  # drive the next reflection off the new failures
            print(f"[iter {it}] ACCEPTED (Δ={cand_fit-seed_fit:+.3f})")
        else:
            write_prompt(args.prompt, best_text)  # restore
            print(f"[iter {it}] rejected")
        history.append({"iter": it, "fitness": cand_fit, "accepted": cand_fit > best_fit - 1e-9,
                        "rationale": rationale, "prompt_len": len(new_text)})

    # Save live + versioned copy ONLY if fitness actually moved. If the
    # optimizer never accepted a candidate, don't pollute the version history
    # with a redundant copy of v1.
    improved = best_fit > seed_fit + 1e-9
    if improved:
        write_prompt(args.prompt, best_text)
        version = next_version(args.prompt)
        versioned_path = PROMPTS_SRC / f"{args.prompt}.v{version}.txt"
        versioned_path.write_text(best_text)
        accepted_rationales = [
            h["rationale"] for h in history
            if h.get("accepted") and h.get("rationale") and h["rationale"] != "(seed)"
        ]
        notes = " · ".join(r.replace("|", "/").replace("\n", " ")[:120]
                           for r in accepted_rationales) or "—"
        append_version_log(args.prompt, version, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed_fitness": seed_fit,
            "final_fitness": best_fit,
            "delta": best_fit - seed_fit,
            "iters": args.iters,
            "notes": notes,
        })
    else:
        # Don't leave a stale copy of v1 sitting in optimized/. That's exactly
        # the trap the user hit earlier — collect_ab.py sees the file and
        # assumes it's a real optimization.
        clear_optimized(args.prompt)

    log_path = HERE / f"optimize_log_{args.prompt}.json"
    log_path.write_text(json.dumps({
        "prompt": args.prompt,
        "seed_fitness": seed_fit,
        "final_fitness": best_fit,
        "iters": args.iters,
        "batch_ids": [q["id"] for q in batch],
        "history": history,
        "improved": improved,
        "version_written": (f"v{version}" if improved else None),
    }, indent=2))
    print(f"\n[done] seed={seed_fit:.3f} → final={best_fit:.3f}  (Δ={best_fit-seed_fit:+.3f})")
    if improved:
        print(f"[done] optimized prompt written: {PROMPTS_OPTIMIZED / (args.prompt + '.txt')}")
        print(f"[done] versioned copy: {versioned_path}")
    else:
        print(f"[done] no improvement over seed; optimized/ cleared (v1 remains the active prompt)")
    print(f"[done] log: {log_path}")


if __name__ == "__main__":
    main()
