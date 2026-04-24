"""
Run every question in questions.yaml against the RAG server and store
raw system outputs (answer + sources + timings) per question.

Usage:
    # start the RAG server in another terminal first:
    #   cd rag && python app.py
    python run.py                 # run all questions
    python run.py --limit 5       # quick smoke test
    python run.py --only factual  # only one category
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "questions.yaml"
RESULTS_PATH = HERE / "results.json"
SERVER = os.getenv("RAG_SERVER", "http://localhost:5001")
TIMEOUT = 120


def parse_sse(stream):
    """Parse the SSE stream from /api/chat. Returns the `done` event payload, or
    a dict `{"_error": "<message>"}` if the stream ended with an `error` event
    (e.g. upstream 429 rate limit). Callers can use this to decide whether to
    retry."""
    event = None
    data_lines = []
    final = None
    last_error = None
    for raw in stream.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.rstrip("\r")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line == "":
            if data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                except Exception:
                    payload = None
                if event == "done" and payload is not None:
                    final = payload
                elif event == "error" and payload is not None:
                    last_error = payload.get("message") or str(payload)
            event = None
            data_lines = []
    if final is not None:
        return final
    if last_error is not None:
        return {"_error": last_error}
    return None


def ask(query, history=None):
    r = requests.post(
        f"{SERVER}/api/chat",
        json={"query": query, "history": history or []},
        stream=True,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return parse_sse(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="filter by category")
    ap.add_argument("--out", default=str(RESULTS_PATH))
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
            resp = ask(q["question"])
            dt = time.time() - t0
            if resp is None:
                print(f"  WARN: no done event received")
                results.append({"id": qid, "error": "no_done_event", "elapsed_s": dt})
                continue
            answer = resp.get("answer", "")
            sources = resp.get("sources", [])
            query_type = resp.get("query_type")
            timings = resp.get("timings", {})
            results.append({
                "id": qid,
                "question": q["question"],
                "category": q["category"],
                "query_type_seen": query_type,
                "answer": answer,
                "sources": sources,
                "timings": timings,
                "elapsed_s": dt,
            })
            print(f"  ok ({dt:.1f}s, {len(sources)} sources, {len(answer)} chars)")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": qid, "error": str(e), "elapsed_s": time.time() - t0})

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
