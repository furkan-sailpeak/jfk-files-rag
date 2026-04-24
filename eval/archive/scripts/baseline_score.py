"""
Score baseline (plain-GPT) answers with the same gpt-5.4-mini judge.

The rubric drops faithfulness and evidence metrics (no retrieval → nothing
to be faithful TO), and adds `correctness_vs_corpus`: does the answer agree
with what the corpus-grounded reference says? This is the heart of the
comparison — can plain GPT state as fact something that contradicts the
archival record?

Usage:
    python baseline_score.py
"""
import json
import os
from pathlib import Path

import yaml
from openai import OpenAI
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")

QUESTIONS_PATH = HERE / "questions.yaml"
RESULTS_PATH = HERE / "baseline_results.json"
SCORES_PATH = HERE / "baseline_scores.json"

JUDGE_MODEL = os.getenv("JUDGE_MODEL_EVAL", "gpt-5.4-mini")
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REFUSAL_PHRASES = [
    "do not contain sufficient information",
    "i don't know",
    "i do not know",
    "cannot answer",
    "unable to answer",
    "no information",
]


JUDGE_SYSTEM = """You are an impartial evaluator comparing a plain-LLM answer (no retrieval) against a corpus-grounded reference answer authored from declassified JFK archival documents. Return STRICT JSON only — no prose, no markdown."""

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

REFERENCE ANSWER (ground truth — authored from the archival corpus; this is what a criminologist would accept as correct):
{reference}

REFERENCE KEY FACTS (the answer should cover these if relevant):
{key_facts}

LLM ANSWER (to evaluate — this model had no access to the archive; relied on training knowledge only):
{answer}

Return JSON with these keys EXACTLY:
{{
  "completeness": <float 0.0-1.0>,           // coverage of reference key facts
  "correctness_vs_corpus": <float 0.0-1.0>,  // 1.0 = every claim in the LLM answer is consistent with the reference; 0.0 = multiple claims contradict the archival record
  "hallucination": <bool>,                   // does the LLM invent specifics (names, dates, document references) not supported by the reference or general consensus? true = yes
  "over_commits": <bool>,                    // does it state contested or uncertain things as settled fact without hedging?
  "acknowledges_uncertainty": <bool>,        // does the answer properly signal when it isn't sure, or does it sound overconfident throughout?
  "clarity": <int 1-5>,                      // readability for a criminologist reader
  "rationale": "<one short sentence explaining the main deduction>"
}}

Only JSON. No code fences."""


def judge(q, r):
    user = JUDGE_USER_TEMPLATE.format(
        question=q["question"],
        reference=q["reference_answer"],
        key_facts="\n".join(f"- {f}" for f in q.get("key_facts", [])),
        answer=r.get("answer", ""),
    )
    resp = oai.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def is_refusal(answer):
    if not answer:
        return False
    a = answer.lower()
    return any(p in a for p in REFUSAL_PHRASES)


def score_one(q, r):
    out = {"id": q["id"], "category": q["category"], "question": q["question"]}
    if r.get("error"):
        out["error"] = r["error"]
        return out
    ans = r.get("answer", "")
    out["answer"] = ans
    out["is_refusal_like"] = is_refusal(ans)

    # out_of_scope: we don't judge; we just observe whether the plain model
    # refused (as the RAG system is required to) or answered the trivia.
    if q["category"] == "out_of_scope":
        out["answered_out_of_scope"] = not out["is_refusal_like"]
        return out

    try:
        out["judge"] = judge(q, r)
    except Exception as e:
        out["judge_error"] = str(e)
    return out


def main():
    questions = {q["id"]: q for q in yaml.safe_load(QUESTIONS_PATH.read_text())}
    results = {r["id"]: r for r in json.loads(RESULTS_PATH.read_text())}

    scored = []
    for qid, q in questions.items():
        r = results.get(qid)
        if r is None:
            print(f"skip {qid}: no baseline result")
            continue
        print(f"scoring baseline {qid}...")
        scored.append(score_one(q, r))

    SCORES_PATH.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
    print(f"\nwrote {SCORES_PATH}")


if __name__ == "__main__":
    main()
