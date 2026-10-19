"""
Prompt consistency check: runs every LLM stage of the pipeline several times on
a fixed question set and reports (a) whether each output honours its prompt's
contract and (b) whether repeated runs agree with each other.

Not the thesis eval -- no reference answers, no OpenAI judge. It answers "do
the prompts behave the same way every time", which the eval cannot.

Run inside the app container (needs the app's env and DB):
    docker compose exec -T app python - < rag/prompt_check.py
Env: RUNS (default 3), WORKERS (default 3), OUT (json path, default stdout only).
"""
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from app import config, generation, llm, postprocess, prompts, retrieval, schemas, verification

RUNS = int(os.getenv("RUNS", "3"))
WORKERS = int(os.getenv("WORKERS", "3"))
REFUSAL = config.REFUSAL_LINE

RUBY_HIST = [
    {"role": "user", "content": "Who was Jack Ruby?"},
    {"role": "assistant", "content": "**Jack Ruby was a Dallas nightclub owner who shot Lee Harvey Oswald on November 24, 1963.** [1]"},
]
OSWALD_HIST = [
    {"role": "user", "content": "Who was Lee Harvey Oswald?"},
    {"role": "assistant", "content": "**Lee Harvey Oswald was a former U.S. Marine who defected to the Soviet Union in 1959.** [1]"},
]

# (id, query, history, acceptable router types, extra expectations)
CASES = [
    ("s-kostikov", "Who was Valeriy Kostikov?", [], {"simple"}, {}),
    ("s-mexico-arrival", "When did Oswald arrive in Mexico City?", [], {"simple"}, {}),
    ("s-ruby-club", "What was the name of Jack Ruby's nightclub?", [], {"simple"}, {}),
    ("s-cos-mexico", "Who was the CIA Chief of Station in Mexico City in 1963?", [], {"simple"}, {}),
    ("r-cia-oswald", "What did the CIA know about Oswald before the assassination?", [], {"research"}, {}),
    ("r-fbi-wc", "Analyze the relationship between the FBI and the Warren Commission.", [], {"research"}, {}),
    ("r-exiles", "What do the files say about Cuban exile groups and Oswald?", [], {"research"}, {}),
    ("r-photos", "Summarize the evidence about the Mexico City surveillance photos.", [], {"research"}, {}),
    ("t-defection", "Give a timeline of Oswald's defection to the Soviet Union.", [], {"simple", "research"}, {"timeline": True}),
    ("t-new-orleans", "List the key dates of Oswald's time in New Orleans in 1963 in chronological order.", [], {"simple", "research"}, {"timeline": True}),
    ("f-ruby-crime", "What about his connections to organized crime?", RUBY_HIST, {"simple", "research"}, {"rewrite_contains": "ruby"}),
    ("f-wife", "Did his wife testify?", OSWALD_HIST, {"simple", "research"}, {"rewrite_contains": ("oswald", "marina")}),
    ("n-icecream", "What was Oswald's favorite ice cream flavor according to the files?", [], {"simple", "research"}, {"no_answer": True}),
    ("m-handwriting", "How many pages have handwriting?", [], {"metadata"}, {"meta_col": "includes_handwriting"}),
    ("m-redacted", "Show me documents with redactions.", [], {"metadata"}, {"meta_col": "has_redactions"}),
    ("c-hi", "Hi there!", [], {"conversational"}, {}),
    ("c-capabilities", "What can you do?", [], {"conversational"}, {}),
    ("o-latvia", "What is the capital of Latvia?", [], {"out_of_scope"}, {}),
    ("o-poem", "Write a poem about the ocean.", [], {"out_of_scope"}, {}),
]

# --- Count API failures: app stages swallow exceptions and fall back silently,
# which would otherwise look like (in)consistent model behaviour.
API_ERRORS = Counter()


def _wrap(client, label):
    orig = client.chat.completions.create

    def create(*a, **kw):
        for attempt in range(4):
            try:
                return orig(*a, **kw)
            except Exception as e:
                if "rate" in str(e).lower() and attempt < 3:
                    time.sleep(2 ** attempt * 3)
                    continue
                API_ERRORS[f"{label}: {type(e).__name__}"] += 1
                raise
    client.chat.completions.create = create


_wrap(llm.client, "main")
_wrap(llm.judge_client, "judge")


# --- Output-contract checks -------------------------------------------------
ODD_CITE = re.compile(r"【|［|〔|†")
BULK = re.compile(r"(?:\[\d+\]\s*){3,}")
PREAMBLE = re.compile(r"^\s*(the user|to address|based on the|upon review|user inquiry)", re.I)


def gen_checks(kind, raw, timeline=False):
    t = postprocess.strip_artifacts(raw)
    lines = [ln for ln in t.splitlines() if ln.strip()]
    refusal = t.strip().startswith(REFUSAL[:40])
    c = {
        "refusal": refusal,
        "odd_cite_syntax": len(ODD_CITE.findall(raw)),
        "bulk_cites": len(BULK.findall(t)),
        "preamble": bool(PREAMBLE.search(t)),
        "reasoning_leak": bool(re.search(r"(?mi)^\s*(#+\s*)?step \d|final answer is|\\boxed", raw)),
        "cited": bool(re.search(r"\[\d+\]", t)),
        "words": len(t.split()),
        "has_table": bool(re.search(r"(?m)^\s*\|.*\|\s*$", t)),
    }
    if refusal:
        c["format_ok"] = t.strip() == REFUSAL
        c["shape"] = "refusal"
        return c
    if kind == "simple":
        bullets = [ln for ln in lines if re.match(r"\s*[-*] ", ln)]
        c["bullets"] = len(bullets)
        c["format_ok"] = (
            bool(lines) and lines[0].lstrip().startswith("**")
            and not re.search(r"(?m)^#", t)
            and (2 <= len(bullets) <= 5 or (bool(timeline) and c["has_table"]))
        )
        c["shape"] = "simple-table" if c["has_table"] else "simple-bullets"
    else:
        heads = [h.strip() for h in re.findall(r"(?m)^## (.+)$", t)]
        c["format_ok"] = (
            t.startswith("## Executive Summary")
            and heads[:2] == ["Executive Summary", "Detailed Findings"]
            and len(heads) == 2
        )
        c["shape"] = "research"
        c["within_length"] = c["words"] <= 650
    c["contract_ok"] = c["format_ok"] and c["cited"] and not (c["preamble"] or c["reasoning_leak"] or c["bulk_cites"])
    return c


def jaccard(sets):
    sets = [set(s) for s in sets]
    if not sets or not any(sets):
        return 1.0
    inter = set.intersection(*sets)
    union = set.union(*sets)
    return round(len(inter) / len(union), 2) if union else 1.0


def raw_generate(query, picked, kind, bg, prior=""):
    sys_p = generation.build_rag_system_prompt(kind)
    user_p = generation.build_rag_user_prompt(query, generation.build_context(picked), prior, kind, bg)
    leftovers = sorted(set(re.findall(r"\{[a-z_]+\}", sys_p + user_p)))
    text = llm.complete(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        temperature=config.GEN_TEMPERATURE,
    )
    return text, leftovers


def run_case(case):
    cid, query, hist, want_types, exp = case
    out = {"id": cid, "query": query}

    # Router x RUNS
    routes = [generation.route_query(query, hist).model_dump() for _ in range(RUNS)]
    types = [r["query_type"] for r in routes]
    out["router"] = {
        "types": types,
        "type_ok": all(t in want_types for t in types),
        "type_stable": len(set(types)) == 1,
        "keywords": [r["search_terms"] for r in routes],
        "keyword_jaccard": jaccard([[k.lower() for k in r["search_terms"]] for r in routes]),
        "rewrites": [r["rewritten_query"] for r in routes],
    }
    if "rewrite_contains" in exp:
        needles = exp["rewrite_contains"]
        needles = (needles,) if isinstance(needles, str) else needles
        out["router"]["rewrite_ok"] = all(any(n in r["rewritten_query"].lower() for n in needles) for r in routes)
    if "meta_col" in exp:
        cols = [(r.get("metadata_filter") or {}).get("column") for r in routes]
        out["router"]["meta_cols"] = cols
        out["router"]["meta_ok"] = all(c == exp["meta_col"] for c in cols)

    kind = types[0]
    if kind not in ("simple", "research"):
        return out

    r0 = routes[0]
    terms = [t for t in r0["search_terms"] if t and t.strip()] or [query]
    ts = " ".join(terms)
    cands = retrieval.hybrid_search(ts, r0["rewritten_query"], retrieval.tokenize_terms(ts), or_terms=terms)
    limit = 15 if kind == "simple" else 20
    out["retrieved"] = len(cands)

    # Reranker x RUNS on identical candidates
    picks = [retrieval.rerank(cands, r0["rewritten_query"], limit, terms) for _ in range(RUNS)]
    out["rerank_jaccard"] = jaccard([[(p[1], p[2]) for p in pk] for pk in picks])
    picked = picks[0]

    # Entity briefs x RUNS (cache cleared so each run really calls the model)
    briefs = []
    for _ in range(RUNS):
        generation._entity_brief_cache.clear()
        briefs.append(generation.build_background_block(terms))
    out["brief_present"] = [bool(b) for b in briefs]
    out["brief_stable"] = len(set(out["brief_present"])) == 1
    bg = briefs[0]

    # Generation x RUNS on identical sources
    prior = generation.summarize_last_answer(hist)
    gens, leftovers = [], set()
    for _ in range(RUNS):
        raw, left = raw_generate(query, picked, kind, bg, prior)
        leftovers |= set(left)
        gens.append(raw)
    checks = [gen_checks(kind, g, exp.get("timeline")) for g in gens]
    out["generation"] = {
        "checks": checks,
        "unfilled_placeholders": sorted(leftovers),
        "shape_stable": len({c["shape"] for c in checks}) == 1,
        "refusal_stable": len({c["refusal"] for c in checks}) == 1,
        "all_contract_ok": all(c.get("contract_ok", c["format_ok"]) for c in checks),
        "word_range": [min(c["words"] for c in checks), max(c["words"] for c in checks)],
    }
    if exp.get("timeline"):
        out["generation"]["timeline_table"] = [c["has_table"] for c in checks]

    answer = postprocess.strip_artifacts(gens[0])

    # Grounding judge x RUNS on the identical answer
    verdicts = [verification.check_answer_grounded(answer, picked, r0["rewritten_query"], kind)[0] for _ in range(RUNS)]
    out["grounding"] = {"verdicts": verdicts, "stable": len(set(verdicts)) == 1}

    # Citation verifier x RUNS on the identical answer
    flags = [sorted(verification.verify_citations(answer, picked), key=str) for _ in range(RUNS)]
    n_cites = len(re.findall(r"\[\d+\]", answer))
    out["cite_verify"] = {
        "flagged_counts": [len(f) for f in flags],
        "total_citations": n_cites,
        "jaccard": jaccard(flags) if any(flags) else 1.0,
    }

    # Expansion: contract only (3 queries)
    try:
        reply = llm.complete_json(
            prompts.render("expansion.txt", rewritten_query=r0["rewritten_query"], reason="the answer did not address the subject"),
            schemas.ExpansionReply, temperature=0.4,
        )
        out["expansion_ok"] = len(reply.queries) == 3 and all(q.strip() for q in reply.queries)
    except Exception as e:
        out["expansion_ok"] = f"error: {e}"
    return out




def summarize(results):
    print("\n=== ROUTER ===")
    for r in results:
        ro = r["router"]
        flags = []
        if not ro["type_ok"]:
            flags.append(f"WRONG TYPE {ro['types']}")
        elif not ro["type_stable"]:
            flags.append(f"UNSTABLE {ro['types']}")
        if ro.get("rewrite_ok") is False:
            flags.append(f"REWRITE MISSED {ro['rewrites']}")
        if ro.get("meta_ok") is False:
            flags.append(f"META COL {ro['meta_cols']}")
        if ro["types"][0] in ("simple", "research") and ro["keyword_jaccard"] < 1:
            flags.append(f"keywords vary (J={ro['keyword_jaccard']}): {ro['keywords']}")
        print(f" {r['id']:16} {ro['types'][0]:14} {'OK' if not flags else '; '.join(flags)}")

    print("\n=== GENERATION / JUDGES (RAG cases) ===")
    agg = Counter()
    for r in results:
        g = r.get("generation")
        if not g:
            continue
        cs = g["checks"]
        agg["answers"] += len(cs)
        agg["contract_ok"] += sum(c.get("contract_ok", c["format_ok"]) for c in cs)
        agg["odd_cite_syntax"] += sum(c["odd_cite_syntax"] for c in cs)
        agg["bulk"] += sum(c["bulk_cites"] for c in cs)
        agg["preamble"] += sum(c["preamble"] for c in cs)
        agg["leak"] += sum(c["reasoning_leak"] for c in cs)
        agg["over_length"] += sum(c.get("within_length") is False for c in cs)
        agg["simple_bullets_out_of_range"] += sum(
            c["shape"] == "simple-bullets" and not 2 <= c.get("bullets", 0) <= 5 for c in cs)
        agg["ground_unstable"] += not r["grounding"]["stable"]
        agg["shape_unstable"] += not g["shape_stable"]
        agg["refusal_unstable"] += not g["refusal_stable"]
        bad = [i for i, c in enumerate(cs) if not c.get("contract_ok", c["format_ok"])]
        print(
            f" {r['id']:16} shapes={[c['shape'] for c in cs]} words={g['word_range']} "
            f"contract_fail_runs={bad or '-'} odd_cites={[c['odd_cite_syntax'] for c in cs]} "
            f"rerankJ={r['rerank_jaccard']} briefs={r['brief_present']} "
            f"ground={r['grounding']['verdicts']} citeflags={r['cite_verify']['flagged_counts']}/{r['cite_verify']['total_citations']}"
            + (f" timeline_table={g['timeline_table']}" if "timeline_table" in g else "")
            + (f" UNFILLED={g['unfilled_placeholders']}" if g["unfilled_placeholders"] else "")
            + ("" if r.get("expansion_ok") is True else f" expansion={r.get('expansion_ok')}")
        )
    print("\n=== TOTALS ===")
    for k, v in agg.items():
        print(f" {k}: {v}")
    print(f" api_errors: {dict(API_ERRORS) or 0}")


if __name__ == "__main__":
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(run_case, CASES))
    if os.getenv("OUT"):
        with open(os.getenv("OUT"), "w") as f:
            json.dump(results, f, indent=1, default=str)
    summarize(results)
    print(f"\n{len(CASES)} cases x {RUNS} runs in {time.time() - t0:.0f}s", file=sys.stderr)
