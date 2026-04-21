"""
Baseline: send every eval question directly to gpt-5.4 with no retrieval /
no archive context. Writes baseline_results.json mirroring results.json shape
(minus sources and timings), so the scorer can treat both runs symmetrically.

Usage:
    export OPENAI_API_KEY=...
    python baseline.py                   # run all
    python baseline.py --limit 5         # smoke
    python baseline.py --only factual
"""
import argparse
import json
import os
import time
from pathlib import Path

import yaml
from openai import OpenAI
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")

QUESTIONS_PATH = HERE / "questions.yaml"
BASELINE_RESULTS_PATH = HERE / "baseline_results.json"
BASELINE_MODEL = os.getenv("BASELINE_MODEL", "gpt-5.4")

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = (
    "You are a helpful assistant answering a question asked by a researcher. "
    "Answer clearly and concisely using your own knowledge. "
    "If you do not know or are uncertain, say so explicitly rather than guessing."
)


def ask(question):
    resp = oai.chat.completions.create(
        model=BASELINE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=str(BASELINE_RESULTS_PATH))
    args = ap.parse_args()

    questions = yaml.safe_load(QUESTIONS_PATH.read_text())
    if args.only:
        questions = [q for q in questions if q["category"] == args.only]
    if args.limit:
        questions = questions[: args.limit]

    results = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        print(f"[{i}/{len(questions)}] {qid}  {q['question']!r}")
        t0 = time.time()
        try:
            answer = ask(q["question"])
            dt = time.time() - t0
            results.append({
                "id": qid,
                "question": q["question"],
                "category": q["category"],
                "answer": answer,
                "sources": [],  # baseline has no retrieval
                "model": BASELINE_MODEL,
                "elapsed_s": dt,
            })
            print(f"  ok ({dt:.1f}s, {len(answer)} chars)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": qid, "error": str(e), "elapsed_s": time.time() - t0})

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
