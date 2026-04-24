"""
Evaluate every versioned prompt (v1, v2, v3, ...) against the eval set and
emit a side-by-side metrics table.

Versions are discovered automatically as `rag/prompts/<name>.v*.txt`. Each
one is temporarily swapped into `rag/prompts/optimized/<name>.txt` (what
app.py reads) for the duration of its run, then restored — so whatever was
in optimized/ before the script runs is preserved.

Usage:
    # server must be running: cd rag && python app.py
    python test_versions.py --prompt rag-research                # test split (default)
    python test_versions.py --prompt rag-research --split all    # all 40 questions
    python test_versions.py --prompt rag-research --split train
    python test_versions.py --prompt rag-simple
"""
import argparse
import json
import re
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
    evaluate_on_batch,
    scalar_fitness,
)
from compare_optimized import fmt, mean_of, summarize  # noqa: E402

QUESTIONS_PATH = HERE / "questions.yaml"
SPLITS_PATH = HERE / "splits.json"


def discover_versions(prompt_name):
    """Return [(version_int, path)] sorted ascending."""
    pat = re.compile(rf"^{re.escape(prompt_name)}\.v(\d+)\.txt$")
    found = []
    for p in PROMPTS_SRC.glob(f"{prompt_name}.v*.txt"):
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    return found


def pick_questions(prompt_name, split, questions, splits):
    target_type = PROMPT_TO_QUERY_TYPE[prompt_name]
    if split == "all":
        ids = [q["id"] for q in questions]
    else:
        ids = splits[split]
    by_id = {q["id"]: q for q in questions}
    return [by_id[i] for i in ids
            if by_id[i].get("query_type_expected") == target_type]


def evaluate_version(prompt_name, version_text, test_qs, workers):
    target = PROMPTS_OPTIMIZED / f"{prompt_name}.txt"
    backup = target.read_text() if target.exists() else None
    target.write_text(version_text)
    try:
        scored, _ = evaluate_on_batch(test_qs, workers=workers)
    finally:
        if backup is not None:
            target.write_text(backup)
        else:
            target.unlink(missing_ok=True)
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, choices=list(PROMPT_TO_QUERY_TYPE.keys()))
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    versions = discover_versions(args.prompt)
    if not versions:
        print(f"No versioned prompts found for {args.prompt}. "
              f"Expected rag/prompts/{args.prompt}.v*.txt")
        return
    print(f"[test] versions found: {[f'v{v}' for v, _ in versions]}")

    questions = yaml.safe_load(QUESTIONS_PATH.read_text())
    splits = json.loads(SPLITS_PATH.read_text())
    test_qs = pick_questions(args.prompt, args.split, questions, splits)
    if not test_qs:
        print(f"No items match prompt={args.prompt} in split={args.split}.")
        return
    print(f"[test] {len(test_qs)} items of type={PROMPT_TO_QUERY_TYPE[args.prompt]} "
          f"in split={args.split}")

    results = {}
    for v, path in versions:
        print(f"\n[test] evaluating v{v} from {path.name}...")
        t0 = time.time()
        scored = evaluate_version(args.prompt, path.read_text(), test_qs, args.workers)
        dt = time.time() - t0
        summary = summarize(scored)
        results[f"v{v}"] = {"summary": summary, "scored": scored, "elapsed_s": dt}
        print(f"[test]   done in {dt:.0f}s  fitness={summary['fitness']:.3f}")

    # Markdown report — columns are versions, rows are metrics. When there
    # are ≥2 versions we also show a "Δ vs v1" column per later version so
    # per-sub-metric tradeoffs are visible (otherwise the composite hides
    # them: e.g. v2 wins on faithfulness but loses on completeness).
    version_labels = [f"v{v}" for v, _ in versions]
    show_delta = len(versions) >= 2
    header_cells = ["Metric"] + version_labels
    if show_delta:
        for lbl in version_labels[1:]:
            header_cells.append(f"Δ {lbl} vs {version_labels[0]}")
    lines = [
        f"# Prompt version comparison — `{args.prompt}`\n",
        f"Split: **{args.split}**  ·  items: **{len(test_qs)}**  "
        f"·  type: **{PROMPT_TO_QUERY_TYPE[args.prompt]}**\n",
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join("---" for _ in header_cells) + "|",
    ]
    metric_keys = [
        "fitness", "faithfulness", "completeness", "clarity",
        "hallucination_rate", "over_commits_rate",
        "structure_ok_rate", "has_citations_rate", "recall_at_20",
    ]
    for key in metric_keys:
        row = [key] + [fmt(results[label]["summary"][key]) for label in version_labels]
        if show_delta:
            base = results[version_labels[0]]["summary"][key]
            for lbl in version_labels[1:]:
                later = results[lbl]["summary"][key]
                if base is None or later is None:
                    row.append("—")
                else:
                    row.append(f"{(later - base):+.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # Per-item fitness table
    lines.append("\n## Per-item scalar fitness\n")
    header = ["id", "category"] + [f"v{v}" for v, _ in versions]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    by_id = {f"v{v}": {s["id"]: s for s in results[f"v{v}"]["scored"]} for v, _ in versions}
    for q in test_qs:
        row = [q["id"], q["category"]]
        for v, _ in versions:
            s = by_id[f"v{v}"].get(q["id"], {})
            row.append(f"{scalar_fitness(s):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    report_path = HERE / f"versions_report_{args.prompt}_{args.split}.md"
    data_path = HERE / f"versions_report_{args.prompt}_{args.split}.json"
    report_path.write_text("\n".join(lines))
    data_path.write_text(json.dumps({
        "prompt": args.prompt,
        "split": args.split,
        "n": len(test_qs),
        "versions": {f"v{v}": str(path) for v, path in versions},
        "results": {k: {"summary": r["summary"], "elapsed_s": r["elapsed_s"]}
                    for k, r in results.items()},
        "scored": {k: r["scored"] for k, r in results.items()},
    }, indent=2, ensure_ascii=False))

    print(f"\n[test] wrote {report_path}")
    print(f"[test] wrote {data_path}")
    print("\nFitness summary:")
    for v, _ in versions:
        print(f"  v{v}: {results[f'v{v}']['summary']['fitness']:.3f}")


if __name__ == "__main__":
    main()
