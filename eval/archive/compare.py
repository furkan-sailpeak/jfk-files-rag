"""
Merge RAG (scores.json) and baseline (baseline_scores.json) into a single
comparison.md — the headline artefact for the thesis.

Usage:
    python compare.py
"""
import json
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).parent
RAG_SCORES = HERE / "scores.json"
BASE_SCORES = HERE / "baseline_scores.json"
OUT = HERE / "comparison.md"


def pct(x):
    return "—" if x is None else f"{100*x:.0f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else sum(xs) / len(xs)


def rag_aggregates(scored, cat):
    items = [s for s in scored if s["category"] == cat and not s.get("error")]
    return {
        "n": len(items),
        "recall": mean([i.get("evidence", {}).get("evidence_recall_at_k") for i in items]),
        "faith": mean([i.get("judge", {}).get("faithfulness") for i in items if "judge" in i]),
        "compl": mean([i.get("judge", {}).get("completeness") for i in items if "judge" in i]),
        "clar": mean([i.get("judge", {}).get("clarity") for i in items if "judge" in i]),
        "hall": mean([1.0 if i.get("judge", {}).get("hallucination") else 0.0 for i in items if "judge" in i]),
        "over": mean([1.0 if i.get("judge", {}).get("over_commits") else 0.0 for i in items if "judge" in i]),
        "refusal_ok": mean([1.0 if i.get("correct_refusal") else 0.0 for i in items if "correct_refusal" in i]),
    }


def base_aggregates(scored, cat):
    items = [s for s in scored if s["category"] == cat and not s.get("error")]
    return {
        "n": len(items),
        "compl": mean([i.get("judge", {}).get("completeness") for i in items if "judge" in i]),
        "corr": mean([i.get("judge", {}).get("correctness_vs_corpus") for i in items if "judge" in i]),
        "clar": mean([i.get("judge", {}).get("clarity") for i in items if "judge" in i]),
        "hall": mean([1.0 if i.get("judge", {}).get("hallucination") else 0.0 for i in items if "judge" in i]),
        "over": mean([1.0 if i.get("judge", {}).get("over_commits") else 0.0 for i in items if "judge" in i]),
        "ack_uncert": mean([1.0 if i.get("judge", {}).get("acknowledges_uncertainty") else 0.0 for i in items if "judge" in i]),
        "answered_oos": mean([1.0 if i.get("answered_out_of_scope") else 0.0 for i in items if "answered_out_of_scope" in i]),
    }


def fmt(x, mode="pct"):
    if x is None:
        return "—"
    if mode == "pct":
        return f"{100*x:.0f}%"
    if mode == "score":
        return f"{x:.2f}"
    return str(x)


def main():
    if not RAG_SCORES.exists() or not BASE_SCORES.exists():
        missing = [p.name for p in (RAG_SCORES, BASE_SCORES) if not p.exists()]
        raise SystemExit(f"missing: {missing}. run score.py and baseline_score.py first.")

    rag = json.loads(RAG_SCORES.read_text())
    base = json.loads(BASE_SCORES.read_text())

    categories = ["factual", "biographical", "analytical", "partial_evidence", "out_of_scope"]

    lines = ["# RAG vs. GPT-5.4 Baseline — Comparison\n"]
    lines.append(f"Total questions scored: {len(rag)} RAG, {len(base)} baseline.\n")

    # Headline table
    lines.append("## Headline: completeness, hallucination, over-commitment\n")
    lines.append("| Category | N | Compl (RAG) | Compl (GPT) | Hallucin (RAG) | Hallucin (GPT) | Overcommit (RAG) | Overcommit (GPT) | Clarity (RAG) | Clarity (GPT) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cat in categories:
        if cat == "out_of_scope":
            continue
        r = rag_aggregates(rag, cat)
        b = base_aggregates(base, cat)
        lines.append(
            f"| {cat} | {r['n']} | "
            f"{fmt(r['compl'],'score')} | {fmt(b['compl'],'score')} | "
            f"{fmt(r['hall'])} | {fmt(b['hall'])} | "
            f"{fmt(r['over'])} | {fmt(b['over'])} | "
            f"{fmt(r['clar'],'score')} | {fmt(b['clar'],'score')} |"
        )

    # Corpus correctness (baseline only — RAG is scored on faithfulness instead)
    lines.append("\n## Baseline-specific: does GPT-5.4 contradict the archival record?\n")
    lines.append("| Category | N | Correctness vs. corpus | Acknowledges uncertainty |")
    lines.append("|---|---|---|---|")
    for cat in categories:
        if cat == "out_of_scope":
            continue
        b = base_aggregates(base, cat)
        lines.append(f"| {cat} | {b['n']} | {fmt(b['corr'],'score')} | {fmt(b['ack_uncert'])} |")

    # RAG-specific: faithfulness + retrieval
    lines.append("\n## RAG-specific: faithfulness + retrieval\n")
    lines.append("| Category | N | Recall@20 | Faithfulness |")
    lines.append("|---|---|---|---|")
    for cat in categories:
        if cat == "out_of_scope":
            continue
        r = rag_aggregates(rag, cat)
        lines.append(f"| {cat} | {r['n']} | {fmt(r['recall'])} | {fmt(r['faith'],'score')} |")

    # Out-of-scope is excluded from comparison by design — it's a behavioral
    # split (RAG must refuse, plain GPT is expected to answer). Not comparable.
    lines.append("\n> **Note:** `out_of_scope` questions are intentionally excluded from the comparison. "
                 "The RAG system is designed to refuse; plain GPT is expected to answer. "
                 "It is not a like-for-like axis. See `report.md` for RAG's refusal-accuracy figure.\n")

    # Per-question head-to-head (excludes out_of_scope for the same reason)
    lines.append("\n## Per-question head-to-head\n")
    rag_by_id = {s["id"]: s for s in rag}
    base_by_id = {s["id"]: s for s in base}
    for qid, r in rag_by_id.items():
        if r["category"] == "out_of_scope":
            continue
        b = base_by_id.get(qid)
        if not b:
            continue
        lines.append(f"### {qid} — {r['category']}\n")
        lines.append(f"**Q:** {r['question']}\n")
        if r.get("judge") and b.get("judge"):
            rj, bj = r["judge"], b["judge"]
            lines.append(f"- **RAG:**  compl={rj.get('completeness')}  hallucin={rj.get('hallucination')}  overcommit={rj.get('over_commits')}  faith={rj.get('faithfulness')}  clarity={rj.get('clarity')}")
            lines.append(f"  rationale: {rj.get('rationale')}")
            lines.append(f"- **GPT:**  compl={bj.get('completeness')}  hallucin={bj.get('hallucination')}  overcommit={bj.get('over_commits')}  corr_vs_corpus={bj.get('correctness_vs_corpus')}  clarity={bj.get('clarity')}")
            lines.append(f"  rationale: {bj.get('rationale')}")
        lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
