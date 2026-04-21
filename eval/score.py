"""
Score every result from run.py against its reference evidence and reference
answer. Produces scores.json and report.md.

Deterministic metrics (no LLM):
    - evidence_recall@N    : fraction of ref-evidence (filename,page) tuples
                             that appear in the system's sources list
    - has_citations        : binary; any [N] markers in the answer
    - structure_compliance : binary; simple=bold+bullets, research=two headings
    - no_bulk_cite         : binary; no more than 2 consecutive [N] markers

LLM-as-judge metrics (gpt-5.4-mini):
    - faithfulness         : 0-1; are cited claims supported by the sources
    - completeness         : 0-1; coverage of reference-answer key facts
    - hallucination        : binary; any claim not traceable to any source
    - over_commits         : binary; states contested things as settled
    - correct_refusal      : binary (for out_of_scope only); emitted refusal line
    - clarity              : 1-5; readability for a criminologist reader

Usage:
    export OPENAI_API_KEY=...
    python score.py                 # score results.json → scores.json + report.md
"""
import json
import os
import re
from pathlib import Path

import psycopg2
import yaml
from openai import OpenAI
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
# Cap per-source content passed to the judge. Enough for a paragraph or two of
# context per page, keeps the prompt under the judge's input budget when a
# result has 15-20 sources.
SOURCE_CONTENT_CAP = 1200

_content_cache = {}


def fetch_source_content(filename, page):
    """Pull the actual chunk text for a (filename, page) from jfk_pages so the
    judge can verify a claim against retrieved content, not just filenames."""
    key = (filename, page)
    if key in _content_cache:
        return _content_cache[key]
    if not DATABASE_URL or filename is None or page is None:
        _content_cache[key] = ""
        return ""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT content FROM jfk_pages WHERE filename = %s AND page_number = %s LIMIT 1",
                (filename, page),
            )
            row = cur.fetchone()
            cur.close()
        finally:
            conn.close()
        content = (row[0] or "") if row else ""
    except Exception as e:
        print(f"  [warn] db fetch failed for {filename} p.{page}: {e}")
        content = ""
    _content_cache[key] = content
    return content

QUESTIONS_PATH = HERE / "questions.yaml"
RESULTS_PATH = HERE / "results.json"
SCORES_PATH = HERE / "scores.json"
REPORT_PATH = HERE / "report.md"

JUDGE_MODEL = os.getenv("JUDGE_MODEL_EVAL", "gpt-5.4-mini")
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REFUSAL_LINE = "The retrieved documents do not contain sufficient information to answer this query."


# ---------- deterministic metrics ----------

def source_key(s):
    return (s.get("filename"), s.get("page"))


def evidence_metrics(sys_sources, ref_evidence, k=20):
    sys_set = {source_key(s) for s in sys_sources[:k]}
    ref_set = {(e["filename"], e["page"]) for e in ref_evidence}
    if not ref_set:
        return {"evidence_recall_at_k": None, "k": k}
    hits = sys_set & ref_set
    # Only recall is reported here. Raw precision = |hits| / k is misleading
    # because k=20 and reference evidence is typically 2–4 pages, capping
    # precision at ~20% even under perfect retrieval.
    return {
        "evidence_recall_at_k": len(hits) / len(ref_set),
        "k": k,
        "hits": sorted([list(h) for h in hits]),
        "missed": sorted([list(m) for m in ref_set - sys_set]),
    }


def structure_ok(answer, query_type):
    if not answer:
        return False
    if query_type == "simple":
        # must have at least one bolded line and at least one bullet
        has_bold_lead = bool(re.search(r"^\*\*[^*].+?\*\*", answer.strip(), re.M))
        has_bullets = len(re.findall(r"^\s*-\s+", answer, re.M)) >= 2
        return has_bold_lead and has_bullets
    if query_type == "research":
        return bool(
            re.search(r"^##\s*Executive Summary", answer, re.M) and
            re.search(r"^##\s*Detailed Findings", answer, re.M)
        )
    return True  # metadata/conversational/out_of_scope — no structure rules here


def has_citations(answer):
    return bool(re.search(r"\[\d+\]", answer))


def no_bulk_cite(answer):
    # violation = 3+ consecutive [N] markers
    return not re.search(r"(\[\d+\]\s*){3,}", answer)


# ---------- LLM judge ----------

JUDGE_SYSTEM = """You are an impartial evaluator of a RAG system used by criminologists researching archival documents. You return STRICT JSON only — no prose, no markdown, no preamble.

Your job is to score a single system answer against:
  - the reference answer (authored from the same archival corpus)
  - the sources the system cited

Be rigorous. A criminologist relies on this system; hallucinations and overstatements are unacceptable."""

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

REFERENCE ANSWER (ground truth, derived from the corpus):
{reference}

REFERENCE KEY FACTS (the answer should cover these if relevant):
{key_facts}

SYSTEM ANSWER (to evaluate):
{answer}

SYSTEM SOURCES (what the system actually retrieved — each source shows filename, page, and the chunk of text the system had access to. A claim is "grounded" if it appears in ANY of these, even if the answer cites a different [N] or no [N] at all):
{sources_block}

Return JSON with these keys EXACTLY:
{{
  "faithfulness": <float 0.0-1.0>,        // are the cited claims actually supported by the named sources? 1.0 = every [N] is justified by that source; 0.0 = citations are fabricated.
  "completeness": <float 0.0-1.0>,        // how many of the reference key facts does the answer cover? 1.0 = all; 0.0 = none.
  "hallucination": <bool>,                // CRITICAL DEFINITION: true ONLY if the answer makes a substantive factual claim that does NOT appear, in substance, in ANY of the retrieved source texts above. Do NOT flag as hallucination just because a claim is missing from the reference answer — the retrieved sources are the source of truth here, NOT the reference. A claim that is supported by any retrieved source (even one the answer didn't cite) is grounded. Only flag inventions with no support anywhere in the sources.
  "over_commits": <bool>,                 // does the answer state contested or uncertain things as settled fact, without hedging? true = yes (a problem).
  "clarity": <int 1-5>,                   // readability for a criminologist: 5 = crisp, well-structured, precise; 1 = confusing or poorly organized.
  "rationale": "<one short sentence explaining the key deductions, specifically citing which claim (if any) was unsupported>"
}}

Only JSON. No code fences."""


def judge_answer(q, result):
    source_parts = []
    for i, s in enumerate(result.get("sources", []), 1):
        fn, pg = s.get("filename"), s.get("page")
        content = fetch_source_content(fn, pg)[:SOURCE_CONTENT_CAP]
        if not content:
            source_parts.append(f"[{i}] {fn} p.{pg}\n(content unavailable)")
        else:
            source_parts.append(f"[{i}] {fn} p.{pg}\n{content}")
    sources_block = "\n\n".join(source_parts) or "(no sources)"
    user = JUDGE_USER_TEMPLATE.format(
        question=q["question"],
        reference=q["reference_answer"],
        key_facts="\n".join(f"- {f}" for f in q.get("key_facts", [])),
        answer=result.get("answer", ""),
        sources_block=sources_block,
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


# ---------- refusal category (scored without judge) ----------

def refusal_ok(answer):
    if not answer:
        return False
    return REFUSAL_LINE in answer.strip()


# ---------- main ----------

def score_one(q, r):
    out = {"id": q["id"], "category": q["category"], "question": q["question"]}
    if r.get("error"):
        out["error"] = r["error"]
        return out

    ans = r.get("answer", "")
    out["answer"] = ans
    out["query_type_seen"] = r.get("query_type_seen")

    # deterministic
    out["has_citations"] = has_citations(ans)
    out["no_bulk_cite"] = no_bulk_cite(ans)
    out["structure_ok"] = structure_ok(ans, r.get("query_type_seen"))
    out["evidence"] = evidence_metrics(r.get("sources", []), q.get("reference_evidence", []))

    # refusal category is scored separately — don't waste judge calls on it
    if q["category"] == "out_of_scope":
        out["correct_refusal"] = refusal_ok(ans)
        return out

    # judge
    try:
        judged = judge_answer(q, r)
        out["judge"] = judged
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
            print(f"skip {qid}: no result")
            continue
        print(f"scoring {qid}...")
        scored.append(score_one(q, r))

    SCORES_PATH.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
    print(f"\nwrote {SCORES_PATH}")

    write_report(scored)
    print(f"wrote {REPORT_PATH}")


# ---------- report ----------

def pct(x):
    return "—" if x is None else f"{100*x:.0f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else sum(xs) / len(xs)


def write_report(scored):
    by_cat = {}
    for s in scored:
        by_cat.setdefault(s["category"], []).append(s)

    lines = ["# RAG Evaluation Report\n"]
    lines.append(f"Scored {len(scored)} questions across {len(by_cat)} categories.\n")

    # per-category summary
    lines.append("## Summary by category\n")
    lines.append("| Category | N | recall@20 | struct_ok | cites_ok | no_bulk | faith | compl | clarity | hallucin | overcommit |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cat, items in by_cat.items():
        n = len(items)
        rec = mean([i.get("evidence", {}).get("evidence_recall_at_k") for i in items])
        struct = mean([1.0 if i.get("structure_ok") else 0.0 for i in items])
        cites = mean([1.0 if i.get("has_citations") else 0.0 for i in items])
        nb = mean([1.0 if i.get("no_bulk_cite") else 0.0 for i in items])
        faith = mean([i.get("judge", {}).get("faithfulness") for i in items if "judge" in i])
        compl = mean([i.get("judge", {}).get("completeness") for i in items if "judge" in i])
        clar = mean([i.get("judge", {}).get("clarity") for i in items if "judge" in i])
        hall = mean([1.0 if i.get("judge", {}).get("hallucination") else 0.0 for i in items if "judge" in i])
        over = mean([1.0 if i.get("judge", {}).get("over_commits") else 0.0 for i in items if "judge" in i])
        lines.append(f"| {cat} | {n} | {pct(rec)} | {pct(struct)} | {pct(cites)} | {pct(nb)} | "
                     f"{'—' if faith is None else f'{faith:.2f}'} | "
                     f"{'—' if compl is None else f'{compl:.2f}'} | "
                     f"{'—' if clar is None else f'{clar:.2f}'} | "
                     f"{pct(hall)} | {pct(over)} |")

    # refusal subsection
    refusals = by_cat.get("out_of_scope", [])
    if refusals:
        ok = sum(1 for r in refusals if r.get("correct_refusal"))
        lines.append(f"\n**Refusal accuracy:** {ok}/{len(refusals)} out-of-scope queries correctly refused.\n")

    # per-question detail
    lines.append("\n## Per-question detail\n")
    for s in scored:
        lines.append(f"### {s['id']} — {s['category']}\n")
        lines.append(f"**Q:** {s['question']}\n")
        if s.get("error"):
            lines.append(f"- ERROR: {s['error']}")
            continue
        ev = s.get("evidence", {})
        lines.append(f"- recall@20: {pct(ev.get('evidence_recall_at_k'))}   "
                     f"structure_ok: {s.get('structure_ok')}   "
                     f"no_bulk_cite: {s.get('no_bulk_cite')}")
        if s.get("judge"):
            j = s["judge"]
            lines.append(f"- faithfulness: {j.get('faithfulness')}   "
                         f"completeness: {j.get('completeness')}   "
                         f"clarity: {j.get('clarity')}   "
                         f"hallucin: {j.get('hallucination')}   "
                         f"overcommit: {j.get('over_commits')}")
            lines.append(f"- judge rationale: {j.get('rationale', '')}")
        if s.get("correct_refusal") is not None:
            lines.append(f"- correct_refusal: {s['correct_refusal']}")
        if ev.get("missed"):
            miss = ", ".join(f"{m[0]} p.{m[1]}" for m in ev["missed"])
            lines.append(f"- missed reference evidence: {miss}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
