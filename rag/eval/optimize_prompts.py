"""Textual gradient descent on the answer prompts (rag-simple.j2, rag-research.j2).

Each step:
  1. forward  -- answer the TRAIN questions with the current prompts
  2. loss     -- judge verdicts (yes/no vs. the reference) + cited-source recall,
                 with the judge's reason for every failure
  3. gradient -- an LLM reads the failures and writes what in the prompt caused them
  4. step     -- the LLM edits the prompt to fix that, keeping the contracts
  5. accept   -- the candidate is scored on VALIDATION questions and kept only if it
                 beats the best score by at least MIN_GAIN (one question's worth)
The TEST questions are never used here; score the final prompts on them with
    python rag/eval/langsmith_eval.py run --split test

Retrieval does not depend on the answer prompt, so router + search + rerank run
once per question and are cached; each candidate only re-runs answer writing
and citation checks on the same pages (fast, and without retrieval noise). Not
covered here: the grounding-check -> search-again path, which the final test
eval runs.

    python rag/eval/optimize_prompts.py --steps 3                # pilot
    python rag/eval/optimize_prompts.py --steps 10 --resume RUN  # continue a run

Output: rag/eval/optimization/<run>/ with every candidate prompt, its scores, the
critiques and a summary. Nothing is written to rag/prompts/; copy the winner by
hand after reviewing it (or --apply RUN).
"""
import argparse
import json
import os
import random
import re
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
load_dotenv(HERE.parent.parent / ".env")
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(HERE / ".models"))

import langsmith_eval as le  # noqa: E402
import store  # noqa: E402
from jinja2 import ChoiceLoader, DictLoader, meta  # noqa: E402

from app import config, generation, llm, prompts, retrieval, verification  # noqa: E402
from app.postprocess import (  # noqa: E402
    cite_table_source_columns,
    drop_bulk_citations,
    remap_citations,
    strip_unsupported_citations,
)

DOC_CATEGORIES = ["fact", "person", "explanation", "timeline"]
TARGETS = {"simple": "rag-simple.j2", "research": "rag-research.j2"}
OUT = HERE / "optimization"
MIN_GAIN = 0.024          # = one question on a 21-question validation set (objective is the mean of two 0-1 scores)
WORKERS = 2   # Groq's tokens-per-minute limit: 4 parallel 20-page prompts hit it
# Rate limits: retry (the SDK waits for Groq's retry-after) instead of letting the app's
# router/reranker fall back silently, which would quietly degrade cache and scores.
llm.client = llm.client.with_options(max_retries=8)
llm.judge_client = llm.judge_client.with_options(max_retries=8)

# The prompts are corpus-generic; an edit that names JFK content is overfitting.
DOMAIN_TERMS = re.compile(r"jfk|kennedy|oswald|ruby|warren|\bcia\b|\bfbi\b|cold war|assassinat|declassif|mexico|soviet|kgb|castro|cuba|hsca|nara", re.I)


# --- Split --------------------------------------------------------------------------
def split_path(dataset):
    return dataset.with_name(dataset.stem.replace("dataset_", "split_") + ".json")


def make_split(dataset, rows, seed=7, sizes=(40, 21, 21)):
    """Stratified by category; saved next to the dataset so every run uses the same split."""
    path = split_path(dataset)
    if path.exists():
        return json.loads(path.read_text())
    rng = random.Random(seed)
    parts = {"train": [], "val": [], "test": []}
    docs = [r for r in rows if r["category"] in DOC_CATEGORIES]
    total = len(docs)
    for cat in DOC_CATEGORIES:
        ids = sorted(r["id"] for r in docs if r["category"] == cat)
        rng.shuffle(ids)
        n_val = round(len(ids) * sizes[1] / total)
        n_test = round(len(ids) * sizes[2] / total)
        parts["val"] += ids[:n_val]
        parts["test"] += ids[n_val:n_val + n_test]
        parts["train"] += ids[n_val + n_test:]
    parts["other"] = sorted(r["id"] for r in rows if r["category"] not in DOC_CATEGORIES)
    parts = {k: sorted(v) for k, v in parts.items()}
    parts["note"] = f"stratified by category, seed {seed}; test is only for the final score"
    path.write_text(json.dumps(parts, indent=1) + "\n")
    return parts


# --- Retrieval cache ----------------------------------------------------------------------
def build_cache(rows, path):
    """Router + search + rerank (+ background briefs) once per question."""
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [r for r in rows if r["id"] not in cache]

    def one(row):
        q = row["question"]
        route = generation.route_query(q, [])
        terms = [t for t in route.search_terms if t and t.strip()] or [q]
        limit = config.CONTEXT_LIMIT_SIMPLE if route.query_type == "simple" else config.CONTEXT_LIMIT_RESEARCH
        ts = " ".join(terms).strip()
        cands = retrieval.hybrid_search(ts, route.rewritten_query, retrieval.tokenize_terms(ts), or_terms=terms)
        picked = retrieval.rerank(cands, route.rewritten_query, limit, terms)
        return row["id"], {"query_type": route.query_type, "rewritten": route.rewritten_query, "terms": terms,
                           "picked": [list(p) for p in picked], "background": generation.build_background_block(terms)}

    if todo:
        print(f"[cache] retrieving for {len(todo)} question(s)...")
        with ThreadPoolExecutor(WORKERS) as pool:
            for rid, entry in pool.map(one, todo):
                cache[rid] = entry
        path.write_text(json.dumps(cache))
    return cache


# --- Candidate prompts ------------------------------------------------------------------------
_base_loader = prompts._env.loader


def use_prompts(overrides):
    """Make prompts.load()/render() see `overrides` ({name: template text})."""
    prompts._env.loader = ChoiceLoader([DictLoader(overrides), _base_loader]) if overrides else _base_loader
    if prompts._env.cache is not None:
        prompts._env.cache.clear()


def contract_problems(name, text, original):
    """Why a candidate may not be used (empty list = OK)."""
    problems = []
    try:
        ast = prompts._env.parse(text)
    except Exception as e:
        return [f"template does not parse: {e}"]
    want = sorted(set(re.findall(r"corpus\.[a-z_.]+", original)))
    have = sorted(set(re.findall(r"corpus\.[a-z_.]+", text)))
    if want != have:
        problems.append(f"corpus placeholders changed: {want} -> {have}")
    if meta.find_undeclared_variables(ast) - set(prompts._env.globals):
        problems.append("uses variables the code does not pass")
    if config.REFUSAL_LINE not in text:
        problems.append("lost the exact refusal line")
    if "Date | Event | Source" not in text:
        problems.append("lost the timeline table format 'Date | Event | Source'")
    if not re.search(r"\[1\]|\[N\]", text):
        problems.append("lost the [N] citation format")
    added = set(m.lower() for m in DOMAIN_TERMS.findall(text)) - set(m.lower() for m in DOMAIN_TERMS.findall(original))
    if added:
        problems.append(f"added collection-specific terms {sorted(added)} (overfitting; prompts must stay generic)")
    if len(text) > 1.6 * len(original):
        problems.append("grew by more than 60% (keep edits targeted)")
    return problems


# --- Forward + loss ---------------------------------------------------------------------------
def answer(row, entry):
    """Answer writing + citation checks on the cached pages, as in pipeline.answer_rag."""
    picked = [tuple(p) for p in entry["picked"]]
    qt = entry["query_type"] if entry["query_type"] in TARGETS else "research"
    system_prompt = generation.build_rag_system_prompt(qt)
    text = generation.generate_answer(row["question"], picked, system_prompt, "", qt, entry["background"])
    text = drop_bulk_citations(cite_table_source_columns(text, len(picked)))
    unsupported = verification.verify_citations(text, picked)
    if unsupported:
        text = strip_unsupported_citations(text, unsupported)
    refs = [{"filename": p[1], "page": p[2]} for p in picked]
    text, cited = remap_citations(text, refs)
    if not re.search(r"\[\d+\]", text):
        cited = []
    return text, cited, refs


def score(row, entry):
    ex = le._example(row)
    try:
        text, cited, refs = answer(row, entry)
    except Exception as e:
        # Left out of the averages (an API error is not a wrong answer), but reported.
        return {"id": row["id"], "category": row["category"], "query_type": entry["query_type"], "error": str(e)[:200]}
    md = types.SimpleNamespace(metadata=ex["metadata"])
    judge = le.correctness(ex["inputs"], {"answer": text}, ex["outputs"], md)
    recall = le.source_recall({"cited_sources": cited}, ex["outputs"])
    retrieved = le.retrieval_recall({"sources": refs}, ex["outputs"])
    return {"id": row["id"], "category": row["category"], "query_type": entry["query_type"],
            "question": row["question"], "reference": row["ground_truth_answer"], "answer": text,
            "correct": float(judge["score"]), "judge": judge["comment"],
            "cited_recall": recall["score"] if recall["score"] is not None else 0.0, "cited_comment": recall["comment"],
            "retrieved_recall": retrieved["score"] or 0.0}


def evaluate(overrides, ids, rows, cache):
    use_prompts(overrides)
    by_id = {r["id"]: r for r in rows}
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda i: score(by_id[i], cache[i]), ids))
    use_prompts(None)
    ok = [r for r in results if "error" not in r]
    if len(ok) < len(results):
        print(f"  ! {len(results) - len(ok)} question(s) failed with API errors and are left out of the score")
    correct = sum(r["correct"] for r in ok) / max(len(ok), 1)
    cited = sum(r["cited_recall"] for r in ok) / max(len(ok), 1)
    return {"objective": (correct + cited) / 2, "correct": correct, "cited_recall": cited,
            "errors": len(results) - len(ok), "results": results}


# --- Gradient + step ------------------------------------------------------------------------------
GRADIENT_PROMPT = """You are optimizing a system prompt for a retrieval-augmented assistant that answers questions
using ONLY numbered source pages and must cite every claim as [N].

Below is the current prompt, then answers it produced that FAILED, each with the question, the reference
answer, the assistant's answer, why a judge rejected it, and whether the expected source pages were cited.

Diagnose what in the PROMPT caused these failures (not the individual facts). Look for patterns: answers that
refuse although the pages contain the answer; facts taken from the wrong page or the wrong person; claims left
uncited or cited to the wrong page; the main point buried; timelines with wrong dates; missing the core of the
question. Be concrete: name the instruction that is missing, unclear, or counterproductive.

Reply with 3-6 short numbered findings, each ending with the prompt change it suggests.

=== CURRENT PROMPT ===
{prompt}

=== FAILURES ===
{failures}"""

STEP_PROMPT = """Rewrite the system prompt below to fix these findings.

Findings:
{critique}

Rules for the rewrite:
- Make targeted edits; keep everything that is not implicated. Do not make it more than ~30% longer.
- The prompt must stay GENERIC: it is used for any document collection. Never add names, places, topics or
  facts from the failures; describe the behaviour instead.
- Keep exactly, character for character: every {{{{ ... }}}} and {{% ... %}} template tag, the refusal line
  "{refusal}", the citation format [N], and the timeline table header "Date | Event | Source".
- Do not add any new template variables.

Return ONLY the full new prompt between <prompt> and </prompt>.

=== CURRENT PROMPT ===
{prompt}"""


def failures_text(results, k=8):
    bad = [r for r in results if "error" not in r and (r["correct"] < 1 or r["cited_recall"] < 1)]
    bad.sort(key=lambda r: (r["correct"], r["cited_recall"]))
    out = []
    for r in bad[:k]:
        out.append(f"--- [{r['category']}] Q: {r['question']}\nREFERENCE: {r['reference'][:500]}\n"
                   f"ANSWER: {r['answer'][:900]}\nJUDGE: {r['judge'][:400]}\nCITATIONS: {r['cited_comment']}")
    return "\n\n".join(out), len(bad)


def propose(name, current, results):
    text, n_bad = failures_text(results)
    if not n_bad:
        return None, "no failures"
    critique = llm.complete([{"role": "user", "content": GRADIENT_PROMPT.format(prompt=current, failures=text)}], temperature=0.3)
    reply = llm.complete([{"role": "user", "content": STEP_PROMPT.format(
        critique=critique, refusal=config.REFUSAL_LINE, prompt=current)}], temperature=0.3)
    m = re.search(r"<prompt>\s*(.*?)\s*</prompt>", reply, re.S)
    candidate = (m.group(1) if m else "").strip() + "\n"
    return candidate, critique


# --- Run ---------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", help="file name or stem in rag/eval/datasets/ (default: newest)")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--resume", help="run folder name to continue from its best prompts")
    p.add_argument("--apply", help="copy the best prompts of this run into rag/prompts/ and exit")
    args = p.parse_args()

    if args.apply:
        best = OUT / args.apply / "best"
        for f in best.glob("*.j2"):
            (Path(prompts.PROMPTS_DIR) / f.name).write_text(f.read_text())
            print(f"applied {f.name}")
        return

    dataset = store.resolve(args.dataset)
    rows = store.load(dataset)
    split = make_split(dataset, rows)
    print(f"split: train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])} (test untouched)")
    by_id = {r["id"]: r for r in rows}
    OUT.mkdir(exist_ok=True)
    cache = build_cache([by_id[i] for i in split["train"] + split["val"]], OUT / f"cache_{dataset.stem}.json")

    run = args.resume or datetime.now().strftime("%Y%m%d-%H%M%S")
    rdir = OUT / run
    (rdir / "best").mkdir(parents=True, exist_ok=True)
    originals = {n: prompts.source(n) for n in TARGETS.values()}
    best = {n: ((rdir / "best" / n).read_text() if (rdir / "best" / n).exists() else originals[n]) for n in TARGETS.values()}
    log = json.loads((rdir / "log.json").read_text()) if (rdir / "log.json").exists() else []

    def save():
        for n, t in best.items():
            (rdir / "best" / n).write_text(t)
        (rdir / "log.json").write_text(json.dumps(log, indent=1))

    t0 = time.time()
    overrides = {n: t for n, t in best.items() if t != originals[n]}
    val_best = evaluate(overrides, split["val"], rows, cache)
    train = evaluate(overrides, split["train"], rows, cache)
    print(f"[start] val objective {val_best['objective']:.3f} (correct {val_best['correct']:.2f}, cited {val_best['cited_recall']:.2f}) | "
          f"train objective {train['objective']:.3f}")
    if not log:
        log.append({"step": 0, "kind": "baseline", "val": {k: v for k, v in val_best.items() if k != "results"},
                    "train": {k: v for k, v in train.items() if k != "results"}})
        (rdir / "step0_val.json").write_text(json.dumps(val_best["results"], indent=1))
        (rdir / "step0_train.json").write_text(json.dumps(train["results"], indent=1))
    save()

    start = max(e["step"] for e in log) + 1
    for step in range(start, start + args.steps):
        # Alternate the prompt being edited, choosing the one whose questions fail most on train.
        fails = {}
        for r in (r for r in train["results"] if "error" not in r):
            n = TARGETS.get(r["query_type"], "rag-research.j2")
            fails.setdefault(n, []).append(r)
        name = max(fails, key=lambda n: sum(1 - (r["correct"] + r["cited_recall"]) / 2 for r in fails[n]))
        print(f"\n[step {step}] editing {name} ({len(fails[name])} train questions use it)")
        candidate, critique = propose(name, best[name], fails[name])
        entry = {"step": step, "prompt": name, "critique": critique}
        if not candidate:
            entry["result"] = "no candidate"
            log.append(entry)
            save()
            continue
        (rdir / f"step{step}_{name}").write_text(candidate)
        problems = contract_problems(name, candidate, originals[name])
        if problems:
            print(f"  rejected before scoring: {'; '.join(problems)}")
            entry.update(result="rejected (contract)", problems=problems)
            log.append(entry)
            save()
            continue
        trial = {**{n: t for n, t in best.items() if t != originals[n]}, name: candidate}
        val = evaluate(trial, split["val"], rows, cache)
        gain = val["objective"] - val_best["objective"]
        accepted = gain >= MIN_GAIN
        print(f"  val objective {val['objective']:.3f} (correct {val['correct']:.2f}, cited {val['cited_recall']:.2f}), "
              f"gain {gain:+.3f} -> {'ACCEPTED' if accepted else 'rejected'}")
        entry.update(result="accepted" if accepted else "rejected (no gain)", gain=gain,
                     val={k: v for k, v in val.items() if k != "results"})
        (rdir / f"step{step}_val.json").write_text(json.dumps(val["results"], indent=1))
        if accepted:
            best[name] = candidate
            val_best = val
            train = evaluate({n: t for n, t in best.items() if t != originals[n]}, split["train"], rows, cache)
            entry["train"] = {k: v for k, v in train.items() if k != "results"}
        log.append(entry)
        save()

    changed = [n for n, t in best.items() if t != originals[n]]
    summary = [f"# Prompt optimization {run}", "",
               f"Dataset `{dataset.name}`; split train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}.",
               f"Objective = mean(correct, cited-source recall) on VALIDATION; accept if gain >= {MIN_GAIN}.", "",
               "| step | prompt | result | val objective | correct | cited |", "|---|---|---|---|---|---|"]
    for e in log:
        v = e.get("val", {})
        summary.append(f"| {e['step']} | {e.get('prompt', '-')} | {e.get('result', e.get('kind'))} | "
                       f"{v.get('objective', float('nan')):.3f} | {v.get('correct', float('nan')):.2f} | {v.get('cited_recall', float('nan')):.2f} |")
    summary += ["", f"Changed prompts: {', '.join(changed) or 'none'} (in best/). "
                "Score them on the untouched TEST split before adopting: "
                f"python rag/eval/langsmith_eval.py run --split test (after --apply {run} on a branch)."]
    (rdir / "summary.md").write_text("\n".join(summary) + "\n")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results: {rdir}")
    print("\n".join(summary[5:]))


if __name__ == "__main__":
    main()
