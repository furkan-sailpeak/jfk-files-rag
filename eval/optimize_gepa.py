"""
Prompt optimizer using the OFFICIAL GEPA library (gepa-ai/gepa).

This is the primary optimizer. It plugs the existing Flask RAG pipeline into
GEPA via a custom adapter, giving us:
  - Pareto-frontier candidate selection (vs. our hand-rolled hill-climber).
  - Minibatch rotation ("epoch_shuffled") to reduce overfit to a single slice.
  - Well-tested reflective mutation + optional merge operators.

Comparison to the hand-rolled `optimize.py` in the same dir:
  - Same RAG black box, same scorer, same .vN.txt version scheme.
  - GEPA manages the population and candidate selection. We only supply the
    adapter: `evaluate(batch, candidate)` → scores + trajectories, and
    `make_reflective_dataset(...)` → feedback records for the reflector LM.

Academic basis: GEPA (Agrawal et al. 2025, arXiv:2507.19457).

Usage:
    # server must be running: cd rag && python app.py
    python optimize_gepa.py --prompt rag-research --max-calls 100
    python optimize_gepa.py --prompt rag-simple   --max-calls 100
    # Env: OPENAI_API_KEY (for gpt-5.4 reflector), plus the rag/ .env keys.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")
sys.path.insert(0, str(HERE))

import gepa  # noqa: E402
from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: E402

# Reuse utilities already built for the hand-rolled optimizer.
from optimize import (  # noqa: E402
    PROMPT_TO_QUERY_TYPE,
    PROMPTS_OPTIMIZED,
    PROMPTS_SRC,
    append_version_log,
    evaluate_on_batch,
    next_version,
    scalar_fitness,
    write_prompt,
)

QUESTIONS_PATH = HERE / "questions.yaml"
SPLITS_PATH = HERE / "splits.json"


class JFKRagAdapter(GEPAAdapter):
    """
    GEPA adapter that plugs a candidate system prompt into the live Flask RAG
    pipeline, evaluates via the same scorer used by test_versions.py, and
    exposes per-item judge feedback for the reflector LM.

    Single optimized component: "system_prompt".

    `evaluate()` writes the candidate into rag/prompts/optimized/<name>.txt
    (which app.py reads first), runs every question in the batch through
    /api/chat concurrently, scores with score.py, and returns per-item
    scalar_fitness in [0,1]. On failure (network / rate-limit / empty
    response) the item gets the existing retries_exhausted → score 0.0 path.
    """

    def __init__(self, prompt_name, workers=2):
        self.prompt_name = prompt_name
        self.workers = workers

    def _apply_candidate(self, candidate):
        """Install the candidate's system prompt into optimized/<name>.txt so
        the running Flask server picks it up on the next request. Returns a
        restore callback so the caller can roll back to whatever was there."""
        target = PROMPTS_OPTIMIZED / f"{self.prompt_name}.txt"
        backup = target.read_text() if target.exists() else None
        target.write_text(candidate["system_prompt"])

        def restore():
            if backup is not None:
                target.write_text(backup)
            else:
                target.unlink(missing_ok=True)
        return restore

    def evaluate(self, batch, candidate, capture_traces=False):
        restore = self._apply_candidate(candidate)
        try:
            scored, _ = evaluate_on_batch(list(batch), workers=self.workers)
        finally:
            restore()

        # Align scored with batch by id (evaluate_on_batch preserves order, but
        # be defensive).
        by_id = {s["id"]: s for s in scored}
        ordered = [by_id.get(q["id"], {"id": q["id"], "error": "missing"}) for q in batch]

        scores = [scalar_fitness(s) for s in ordered]
        outputs = [{"answer": s.get("answer", ""), "score": sc} for s, sc in zip(ordered, scores)]
        trajectories = ordered if capture_traces else None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, _candidate, eval_batch, components_to_update):
        """
        For each item in the evaluated batch, produce a reflection record the
        teacher LM can reason over. Following GEPA's recommended schema:
        {Inputs, Generated Outputs, Feedback}.

        Feedback carries: scalar fitness, per-sub-metric values (faithfulness,
        completeness, hallucination, structure, citations, recall), and the
        judge's one-sentence rationale.
        """
        records = []
        for i, s in enumerate(eval_batch.trajectories or []):
            j = s.get("judge") or {}
            ev = s.get("evidence") or {}
            fitness = eval_batch.scores[i]
            feedback_parts = [
                f"scalar_fitness={fitness:.3f}",
                f"faithfulness={j.get('faithfulness')}",
                f"completeness={j.get('completeness')}",
                f"hallucination={j.get('hallucination')}",
                f"over_commits={j.get('over_commits')}",
                f"clarity={j.get('clarity')}",
                f"structure_ok={s.get('structure_ok')}",
                f"has_citations={s.get('has_citations')}",
                f"evidence_recall@20={ev.get('evidence_recall_at_k')}",
            ]
            if j.get("rationale"):
                feedback_parts.append(f"judge_rationale: {j['rationale']}")
            if s.get("error"):
                feedback_parts.append(f"error: {s['error']}")

            records.append({
                "Inputs": {
                    "question": s.get("question", ""),
                    "category": s.get("category", ""),
                },
                "Generated Outputs": (s.get("answer", "") or "")[:1800],
                "Feedback": " | ".join(feedback_parts),
            })
        # Same records fed for any component GEPA wants to update (we have one).
        return {comp: records for comp in components_to_update}


def load_split(prompt_name):
    target_type = PROMPT_TO_QUERY_TYPE[prompt_name]
    questions = yaml.safe_load(QUESTIONS_PATH.read_text())
    splits = json.loads(SPLITS_PATH.read_text())
    by_id = {q["id"]: q for q in questions}
    train = [by_id[i] for i in splits["train"]
             if by_id[i].get("query_type_expected") == target_type]
    test = [by_id[i] for i in splits["test"]
            if by_id[i].get("query_type_expected") == target_type]
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, choices=list(PROMPT_TO_QUERY_TYPE.keys()))
    ap.add_argument("--max-calls", type=int, default=100,
                    help="Max total evaluations GEPA may perform. ~6-10x this many RAG calls.")
    ap.add_argument("--minibatch", type=int, default=5,
                    help="Reflection minibatch size (GEPA default is 3).")
    ap.add_argument("--workers", type=int, default=2,
                    help="Concurrent RAG calls per batch. Keep ≤2 to avoid Groq 429s.")
    # gpt-5.4-mini is the default: ~10x cheaper than gpt-5.4 and empirically
    # sufficient for prompt-editing reflection. Override via --reflection-lm
    # if you want the full model.
    ap.add_argument("--reflection-lm", default="openai/gpt-5.4-mini")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    # Ensure OPENAI_API_KEY is present for the reflection LM (litellm-routed).
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set; the reflection LM requires it.")
        sys.exit(1)

    prompt_path = PROMPTS_SRC / f"{args.prompt}.txt"
    seed_text = prompt_path.read_text()
    seed = {"system_prompt": seed_text}

    train, _held_out_test = load_split(args.prompt)
    # Use train as both trainset AND valset (GEPA supports valset=None → reuses
    # trainset for Pareto tracking). splits["test"] is NOT passed to GEPA —
    # it stays held-out; run test_versions.py after optimization to report on it.
    print(f"[gepa] seed from {prompt_path}, len={len(seed_text)}")
    print(f"[gepa] train={len(train)} items of type={PROMPT_TO_QUERY_TYPE[args.prompt]}")
    print(f"[gepa] reflection_lm={args.reflection_lm} max_calls={args.max_calls}")

    run_dir = args.run_dir or str(HERE / f"gepa_run_{args.prompt}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    adapter = JFKRagAdapter(prompt_name=args.prompt, workers=args.workers)

    result = gepa.optimize(
        seed_candidate=seed,
        trainset=train,
        adapter=adapter,
        reflection_lm=args.reflection_lm,
        reflection_minibatch_size=args.minibatch,
        max_metric_calls=args.max_calls,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        display_progress_bar=True,
        run_dir=run_dir,
        seed=17,
    )

    best_text = result.best_candidate["system_prompt"]
    # GEPA picks the best_candidate from its own population. Detect improvement
    # cheaply: if the returned text equals the seed text, nothing beat the seed.
    improved = best_text.strip() != seed_text.strip()

    if improved:
        write_prompt(args.prompt, best_text)
        version = next_version(args.prompt)
        versioned_path = PROMPTS_SRC / f"{args.prompt}.v{version}.txt"
        versioned_path.write_text(best_text)
        append_version_log(args.prompt, version, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed_fitness": float(getattr(result, "val_aggregate_scores", [0.0])[0] or 0.0),
            "final_fitness": float(max(getattr(result, "val_aggregate_scores", [0.0]) or [0.0])),
            "delta": 0.0,  # GEPAResult doesn't expose a clean scalar delta; see gepa_result.json
            "iters": args.max_calls,
            "notes": f"GEPA run_dir={run_dir}",
        })
        print(f"\n[gepa] optimized prompt written → {PROMPTS_OPTIMIZED / (args.prompt + '.txt')}")
        print(f"[gepa] versioned copy → {versioned_path}")
    else:
        # Clean up optimized/ — don't leave a stale copy of seed sitting there.
        (PROMPTS_OPTIMIZED / f"{args.prompt}.txt").unlink(missing_ok=True)
        print(f"\n[gepa] no improvement over seed; optimized/ cleared (live baseline unchanged)")

    # Dump full GEPAResult for the viva audit trail.
    result_path = HERE / f"gepa_result_{args.prompt}.json"
    try:
        payload = {
            "best_candidate": result.best_candidate,
            "val_aggregate_scores": getattr(result, "val_aggregate_scores", None),
            "val_subscores": getattr(result, "val_subscores", None),
            "num_candidates": getattr(result, "num_candidates", None),
        }
        result_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[gepa] result dump → {result_path}")
    except Exception as e:
        print(f"[gepa] result dump skipped: {e}")


if __name__ == "__main__":
    main()
