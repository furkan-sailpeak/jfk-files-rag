"""
Before/after comparison on the held-out test split.

Runs the RAG system twice over the test questions — once with the committed
`rag/prompts/<name>.txt` (baseline) and once with the optimizer's output
`rag/prompts/optimized/<name>.txt`. Scores both runs with score.py and writes a
side-by-side report.

Usage:
    # server must be running:  cd rag && python app.py
    python compare_optimized.py --prompt rag-research
    python compare_optimized.py --prompt rag-simple
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")
sys.path.insert(0, str(HERE))

from optimize import (  # noqa: E402
    PROMPT_TO_QUERY_TYPE,
    PROMPTS_OPTIMIZED,
    PROMPTS_SRC,
    batch_fitness,
    evaluate_on_batch,
    scalar_fitness,
)

QUESTIONS_PATH = HERE / "questions.yaml"
SPLITS_PATH = HERE / "splits.json"


def mean_of(items, key_path):
    """Average a nested field (e.g. ('judge','faithfulness')) across scored items, skipping Nones."""
    vals = []
    for s in items:
        node = s
        for k in key_path:
            node = (node or {}).get(k) if isinstance(node, dict) else None
        if isinstance(node, bool):
            vals.append(1.0 if node else 0.0)
        elif isinstance(node, (int, float)):
            vals.append(float(node))
    return None if not vals else sum(vals) / len(vals)


def evaluate_with_prompt(prompt_name, prompt_text, test_qs, workers=4):
    """Swap the optimized/ prompt into place, evaluate, then restore. We always
    use the optimized/ slot for both A and B so we go through the exact same
    load_prompt path in app.py."""
    target = PROMPTS_OPTIMIZED / f"{prompt_name}.txt"
    backup = None
    if target.exists():
        backup = target.read_text()
    target.write_text(prompt_text)
    try:
        scored, _ = evaluate_on_batch(test_qs, workers=workers)
    finally:
        if backup is not None:
            target.write_text(backup)
        else:
            target.unlink(missing_ok=True)
    return scored


def summarize(scored):
    return {
        "n": len(scored),
        "fitness": batch_fitness(scored),
        "faithfulness": mean_of(scored, ("judge", "faithfulness")),
        "completeness": mean_of(scored, ("judge", "completeness")),
        "clarity": mean_of(scored, ("judge", "clarity")),
        "hallucination_rate": mean_of(scored, ("judge", "hallucination")),
        "over_commits_rate": mean_of(scored, ("judge", "over_commits")),
        "structure_ok_rate": mean_of(scored, ("structure_ok",)),
        "has_citations_rate": mean_of(scored, ("has_citations",)),
        "recall_at_20": mean_of(scored, ("evidence", "evidence_recall_at_k")),
    }


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, choices=list(PROMPT_TO_QUERY_TYPE.keys()))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    target_type = PROMPT_TO_QUERY_TYPE[args.prompt]
    splits = json.loads(SPLITS_PATH.read_text())
    questions = {q["id"]: q for q in yaml.safe_load(QUESTIONS_PATH.read_text())}
    test_qs = [
        questions[qid] for qid in splits["test"]
        if questions[qid].get("query_type_expected") == target_type
    ]
    if not test_qs:
        print(f"No held-out test items match type={target_type}.")
        return

    seed_path = PROMPTS_SRC / f"{args.prompt}.txt"
    opt_path = PROMPTS_OPTIMIZED / f"{args.prompt}.txt"
    if not opt_path.exists():
        print(f"Missing {opt_path}. Run optimize.py first.")
        return
    seed_text = seed_path.read_text()
    opt_text = opt_path.read_text()

    print(f"[compare] {len(test_qs)} held-out test items of type={target_type}")
    print(f"[compare] evaluating BASELINE (rag/prompts/{args.prompt}.txt)...")
    t0 = time.time()
    seed_scored = evaluate_with_prompt(args.prompt, seed_text, test_qs, args.workers)
    print(f"[compare]   done in {time.time()-t0:.0f}s")

    print(f"[compare] evaluating OPTIMIZED (rag/prompts/optimized/{args.prompt}.txt)...")
    t0 = time.time()
    opt_scored = evaluate_with_prompt(args.prompt, opt_text, test_qs, args.workers)
    print(f"[compare]   done in {time.time()-t0:.0f}s")

    # After evaluation, reinstate optimized slot so the live app stays on the
    # optimized prompt (evaluate_with_prompt restored whatever was present; we
    # explicitly want optimized/ to remain active).
    opt_path.write_text(opt_text)

    seed_sum = summarize(seed_scored)
    opt_sum = summarize(opt_scored)

    report_lines = [
        f"# Prompt optimization — before/after ({args.prompt})\n",
        f"Held-out test items: **{len(test_qs)}** (type={target_type}, split=test, "
        "never seen by the optimizer).\n",
        "| Metric | Baseline | Optimized | Δ |",
        "|---|---|---|---|",
    ]
    for key in ["fitness", "faithfulness", "completeness", "clarity",
                "hallucination_rate", "over_commits_rate",
                "structure_ok_rate", "has_citations_rate", "recall_at_20"]:
        a, b = seed_sum[key], opt_sum[key]
        delta = "—" if (a is None or b is None) else f"{(b-a):+.3f}"
        report_lines.append(f"| {key} | {fmt(a)} | {fmt(b)} | {delta} |")

    report_lines.append("\n## Per-item scalar fitness\n")
    report_lines.append("| id | category | baseline | optimized | Δ |")
    report_lines.append("|---|---|---|---|---|")
    by_id = {s["id"]: s for s in opt_scored}
    for a in seed_scored:
        b = by_id.get(a["id"], {})
        fa = scalar_fitness(a)
        fb = scalar_fitness(b)
        report_lines.append(
            f"| {a['id']} | {a.get('category','')} | {fa:.3f} | {fb:.3f} | {fb-fa:+.3f} |"
        )

    report_path = HERE / f"optimization_report_{args.prompt}.md"
    report_path.write_text("\n".join(report_lines))

    data_path = HERE / f"optimization_report_{args.prompt}.json"
    data_path.write_text(json.dumps({
        "prompt": args.prompt,
        "n": len(test_qs),
        "baseline": seed_sum,
        "optimized": opt_sum,
        "scored_baseline": seed_scored,
        "scored_optimized": opt_scored,
    }, indent=2, ensure_ascii=False))

    print(f"\n[compare] wrote {report_path}")
    print(f"[compare] wrote {data_path}")
    print(f"[compare] fitness: baseline={seed_sum['fitness']:.3f}  "
          f"optimized={opt_sum['fitness']:.3f}  "
          f"Δ={opt_sum['fitness']-seed_sum['fitness']:+.3f}")


if __name__ == "__main__":
    main()
