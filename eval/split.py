"""
Deterministic 30/10 train/test split of questions.yaml.

Uses a seeded shuffle so the counts are *exact* (simple hash-bucketing on n=40
drifts ±5 from the target). Stratified by category so both splits reflect the
overall category mix — critical when categories are unbalanced and the test
split is only ~25% of the data.

Usage:
    python split.py              # writes splits.json
    python split.py --show       # print counts by split x category
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "questions.yaml"
SPLITS_PATH = HERE / "splits.json"

SEED = 17
TRAIN_N = 30
TEST_N = 10


def split(questions):
    """Stratified seeded shuffle → exact 30/10 split.
    Per-category allocation rounds proportionally; any rounding remainder is
    assigned to test (keeps test non-empty for small categories when possible).
    """
    rng = random.Random(SEED)

    # Group by category, shuffle within group for reproducibility.
    by_cat = defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q["id"])
    for qids in by_cat.values():
        rng.shuffle(qids)

    total = TRAIN_N + TEST_N
    if len(questions) != total:
        raise SystemExit(
            f"Expected {total} questions, found {len(questions)}. "
            f"Adjust TRAIN_N/TEST_N in split.py or questions.yaml."
        )

    # Largest-remainder (Hamilton) allocation. Per-category test quota =
    # floor(n_cat * TEST_N / total); distribute the leftover test slots to
    # the categories with the largest fractional remainders. Guarantees
    # sum(test_quotas) == TEST_N exactly AND preserves per-category balance.
    quotas = {}
    remainders = {}
    for cat, qids in by_cat.items():
        raw = len(qids) * TEST_N / total
        quotas[cat] = int(raw)
        remainders[cat] = raw - int(raw)
    deficit = TEST_N - sum(quotas.values())
    # Tie-break on category name for determinism.
    tied = sorted(remainders.items(), key=lambda kv: (-kv[1], kv[0]))
    for cat, _ in tied[:deficit]:
        quotas[cat] += 1

    train_ids, test_ids = [], []
    for cat, qids in by_cat.items():
        n_test = min(quotas[cat], len(qids))
        test_ids.extend(qids[:n_test])
        train_ids.extend(qids[n_test:])

    assert len(train_ids) == TRAIN_N and len(test_ids) == TEST_N, (len(train_ids), len(test_ids))
    return {"train": sorted(train_ids), "test": sorted(test_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    qs = yaml.safe_load(QUESTIONS_PATH.read_text())
    splits = split(qs)
    SPLITS_PATH.write_text(json.dumps(splits, indent=2))
    print(f"wrote {SPLITS_PATH}: train={len(splits['train'])}  test={len(splits['test'])}")

    if args.show:
        by_cat = defaultdict(lambda: {"train": 0, "test": 0})
        train_set = set(splits["train"])
        for q in qs:
            sp = "train" if q["id"] in train_set else "test"
            by_cat[q["category"]][sp] += 1
        print("\nBy category:")
        print(f"{'category':<25} {'train':>6} {'test':>6}")
        for cat, d in sorted(by_cat.items()):
            print(f"{cat:<25} {d['train']:>6} {d['test']:>6}")


if __name__ == "__main__":
    main()
