"""
Collect answers for blinded A/B human evaluation.

Runs every question in questions.yaml through the RAG system twice:
  1. BASELINE:  `rag/prompts/optimized/` cleared → app.py falls back to committed prompts
  2. OPTIMIZED: `rag/prompts/optimized/` restored → app.py uses optimizer's output

Per-question, the two answers are randomly assigned to labels "A" / "B" (seeded
for reproducibility). The mapping is embedded in the HTML but hidden from the
UI until the user clicks "Reveal". Output is a single self-contained file
`eval/ab_rate.html` — open it directly in a browser, no server needed.

Also scores each answer with score.py so "agreement with LLM judge" can be
computed after human rating.

Usage:
    # server must be running:  cd rag && python app.py
    python collect_ab.py                  # all 40 questions
    python collect_ab.py --limit 5        # smoke test
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")
sys.path.insert(0, str(HERE))

from optimize import PROMPTS_OPTIMIZED, evaluate_on_batch, scalar_fitness  # noqa: E402

QUESTIONS_PATH = HERE / "questions.yaml"
OUTPUT_HTML = HERE / "ab_rate.html"
OUTPUT_DATA = HERE / "ab_pairs.json"
SEED = 42


def hide_optimized():
    """Rename only the LIVE optimized prompt files (rag-research.txt,
    rag-simple.txt) to *.txt.__hidden__ so app.py's load_prompt falls back
    to the committed baseline. Numbered version history (.vN.txt) is left
    alone — app.py never reads those files. Returns restore pairs."""
    hidden = []
    if not PROMPTS_OPTIMIZED.exists():
        return hidden
    for name in ("rag-research.txt", "rag-simple.txt"):
        p = PROMPTS_OPTIMIZED / name
        if p.exists():
            tgt = p.with_suffix(p.suffix + ".__hidden__")
            p.rename(tgt)
            hidden.append((tgt, p))
    return hidden


def restore_optimized(hidden):
    for tgt, orig in hidden:
        if tgt.exists():
            tgt.rename(orig)


def run_pass(questions, workers, label):
    print(f"[{label}] evaluating {len(questions)} items...")
    t0 = time.time()
    scored, _ = evaluate_on_batch(questions, workers=workers)
    print(f"[{label}] done in {time.time()-t0:.0f}s")
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    # Default 2 to stay well under Groq's 300K TPM on llama-70b. collect_ab
    # runs 80 RAG calls in a row (40 qs × 2 passes); at workers=4 we routinely
    # hit 429s on iteration boundaries. Retries in optimize.evaluate_on_batch
    # absorb them but add ~10-30s per affected item. Bump to 4 if your quota
    # allows.
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--include-trivial", action="store_true",
                    help="Include out_of_scope and metadata-typed questions. "
                         "These bypass the rag-simple/rag-research prompts "
                         "(canned refusal or document-agent path), so they "
                         "produce identical output in both passes and waste "
                         "RAG calls. Default: skip them.")
    args = ap.parse_args()

    questions = yaml.safe_load(QUESTIONS_PATH.read_text())
    total_before_filter = len(questions)
    if not args.include_trivial:
        # Only rate questions that actually route through an optimisable prompt.
        # query_type_expected of "simple" or "research" → goes through
        # rag-simple.txt / rag-research.txt. Other types (out_of_scope,
        # metadata, conversational, document) take code paths that don't
        # read either optimised prompt, so baseline and optimised passes
        # produce identical output for them.
        questions = [q for q in questions
                     if q.get("query_type_expected") in ("simple", "research")]
        skipped = total_before_filter - len(questions)
        if skipped:
            print(f"[collect] skipped {skipped} trivial items "
                  f"(out_of_scope / metadata / etc.); {len(questions)} remain. "
                  f"Use --include-trivial to keep them.")
    if args.limit:
        questions = questions[: args.limit]

    # Sanity: warn if no optimized prompts exist — without them baseline==optimized.
    opt_files = sorted(PROMPTS_OPTIMIZED.glob("*.txt")) if PROMPTS_OPTIMIZED.exists() else []
    if not opt_files:
        print(f"WARN: no optimized prompts found in {PROMPTS_OPTIMIZED}. "
              "Baseline and optimized runs will be identical. "
              "Run optimize.py first.")
    else:
        print(f"[collect] optimized prompts present: {[p.name for p in opt_files]}")

    # -- Pass 1: baseline (optimized/ hidden) --
    hidden = hide_optimized()
    try:
        baseline_scored = run_pass(questions, args.workers, "baseline")
    finally:
        restore_optimized(hidden)

    # -- Pass 2: optimized (optimized/ visible) --
    optimized_scored = run_pass(questions, args.workers, "optimized")

    by_id_b = {s["id"]: s for s in baseline_scored}
    by_id_o = {s["id"]: s for s in optimized_scored}

    # -- Build A/B pairs with seeded per-question randomization --
    pairs = []
    for q in questions:
        qid = q["id"]
        b, o = by_id_b.get(qid), by_id_o.get(qid)
        if not b or not o:
            continue
        # per-question deterministic coin flip
        rnd = random.Random(f"{SEED}:{qid}")
        if rnd.random() < 0.5:
            A_src, B_src = "baseline", "optimized"
            a_item, b_item = b, o
        else:
            A_src, B_src = "optimized", "baseline"
            a_item, b_item = o, b

        def strip(item):
            # keep only what the rater sees + what the reveal step needs
            return {
                "answer": item.get("answer", "") or "",
                "sources": [
                    {"filename": s.get("filename"), "page": s.get("page")}
                    for s in (item.get("_sources") or [])
                ] if item.get("_sources") else [],
                "judge": item.get("judge") or {},
                "structure_ok": item.get("structure_ok"),
                "has_citations": item.get("has_citations"),
                "fitness": scalar_fitness(item),
            }

        pairs.append({
            "id": qid,
            "question": q["question"],
            "category": q["category"],
            "query_type_expected": q.get("query_type_expected"),
            "reference_answer": q.get("reference_answer", "").strip(),
            "mapping": {"A": A_src, "B": B_src},
            "A": strip(a_item),
            "B": strip(b_item),
        })

    OUTPUT_DATA.write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
    print(f"[collect] wrote {OUTPUT_DATA} ({len(pairs)} pairs)")

    write_html(pairs)
    print(f"[collect] wrote {OUTPUT_HTML} — open it in a browser to rate.")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Blinded A/B — JFK RAG prompt evaluation</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {
    --bg:#f7f5ef; --fg:#2b2a27; --muted:#6e6a60; --card:#fffdf6; --line:#d8d4c8;
    --accent:#7a5a24; --ok:#2f7a3a; --bad:#a32b2b; --tie:#6e6a60; --note:#4a4737;
  }
  * { box-sizing: border-box; }
  body { font-family: Georgia, 'Times New Roman', serif; margin: 0; background: var(--bg); color: var(--fg); }
  header { padding: 1.2rem 2rem 0.6rem; border-bottom: 1px solid var(--line); background:#ece8dc; }
  header h1 { margin: 0 0 0.3rem; font-size: 1.2rem; font-weight: 600; letter-spacing: 0.02em; }
  header p { margin: 0.2rem 0; color: var(--muted); font-size: 0.85rem; max-width: 70ch; }
  .progress { font-size: 0.85rem; color: var(--accent); }
  main { padding: 1.2rem 2rem 4rem; max-width: 1400px; margin: 0 auto; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: 1.2rem; }
  .qhdr { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin-bottom: 0.4rem; }
  .qhdr .q { font-weight: 600; flex: 1; }
  .qhdr .meta { color: var(--muted); font-size: 0.8rem; font-variant: small-caps; letter-spacing: 0.05em; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.8rem; }
  .ans { border: 1px solid var(--line); background: #fffef9; border-radius: 4px; padding: 0.7rem 0.9rem; font-size: 0.92rem; line-height: 1.5; max-height: 500px; overflow-y: auto; }
  .ans h2, .ans h3 { font-size: 1em; margin-top: 0.7em; margin-bottom: 0.3em; }
  .ans p { margin: 0.4em 0; }
  .ans table { border-collapse: collapse; margin: 0.5em 0; font-size: 0.88em; }
  .ans td, .ans th { border: 1px solid var(--line); padding: 0.2em 0.5em; }
  .ans .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.4rem; }
  .vote { margin-top: 0.8rem; display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
  .vote button { font-family: inherit; font-size: 0.85rem; padding: 0.4rem 0.9rem; border: 1px solid var(--line); background: #fffdf6; color: var(--fg); border-radius: 4px; cursor: pointer; }
  .vote button:hover { background: #f0ecdd; }
  .vote button.selected[data-v="A"] { background: var(--ok); color: #fff; border-color: var(--ok); }
  .vote button.selected[data-v="B"] { background: var(--ok); color: #fff; border-color: var(--ok); }
  .vote button.selected[data-v="tie"] { background: var(--tie); color: #fff; border-color: var(--tie); }
  .vote button.selected[data-v="both_bad"] { background: var(--bad); color: #fff; border-color: var(--bad); }
  textarea { font-family: inherit; font-size: 0.85rem; width: 100%; margin-top: 0.5rem; padding: 0.5rem; border: 1px solid var(--line); border-radius: 4px; background: #fffef9; color: var(--note); resize: vertical; min-height: 2.4rem; }
  .reveal { font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; display: none; }
  .reveal.shown { display: block; }
  .reveal .baseline { color: var(--bad); }
  .reveal .optimized { color: var(--ok); }
  footer { position: fixed; bottom: 0; left: 0; right: 0; background: #ece8dc; border-top: 1px solid var(--line); padding: 0.6rem 2rem; display: flex; gap: 0.7rem; align-items: center; font-size: 0.85rem; }
  footer button { font-family: inherit; font-size: 0.85rem; padding: 0.4rem 0.9rem; border: 1px solid var(--line); background: #fffdf6; color: var(--fg); border-radius: 4px; cursor: pointer; }
  footer button:hover { background: #f0ecdd; }
  .stats { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--muted); }
  .ref { font-size: 0.85rem; color: var(--muted); font-style: italic; margin-top: 0.4rem; max-height: 80px; overflow-y: auto; background: #f6f2e6; padding: 0.4rem 0.7rem; border-left: 2px solid var(--line); border-radius: 2px; }
  .hide-ref .ref { display: none; }
</style>
</head>
<body>
<header>
  <h1>Blinded A/B prompt evaluation — JFK files RAG</h1>
  <p>Rate each pair. Labels <b>A</b> and <b>B</b> are randomised per-question; you do not know which is the baseline system prompt and which is the optimised one until you click <b>Reveal</b>. Votes save automatically to this browser.</p>
  <div class="progress"><span id="done">0</span> / <span id="total">0</span> rated</div>
</header>
<main id="main"></main>
<footer>
  <button id="reveal-btn">Reveal &amp; compute stats</button>
  <button id="export-btn">Export JSON</button>
  <button id="clear-btn">Clear votes</button>
  <label style="cursor:pointer;"><input type="checkbox" id="toggle-ref"> show reference answers</label>
  <div class="stats" id="stats"></div>
</footer>
<script>
const DATA = __PAIRS_JSON__;
const STORAGE_KEY = "ab_votes_v1";
const main = document.getElementById("main");
const doneEl = document.getElementById("done");
const totalEl = document.getElementById("total");
const statsEl = document.getElementById("stats");
totalEl.textContent = DATA.length;

let votes = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
let revealed = false;

function save() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(votes));
  const done = Object.values(votes).filter(v => v && v.choice).length;
  doneEl.textContent = done;
  if (revealed) computeStats();
}

function render() {
  main.innerHTML = "";
  DATA.forEach((p, i) => {
    const v = votes[p.id] || {};
    const card = document.createElement("div");
    card.className = "card hide-ref";
    card.id = "q-" + p.id;
    const aMd = marked.parse(p.A.answer || "(empty)");
    const bMd = marked.parse(p.B.answer || "(empty)");
    card.innerHTML = `
      <div class="qhdr">
        <div class="q">${i+1}. ${escapeHtml(p.question)}</div>
        <div class="meta">${p.category} · ${p.query_type_expected || ""}</div>
      </div>
      ${p.reference_answer ? `<div class="ref"><b>Reference:</b> ${escapeHtml(p.reference_answer)}</div>` : ""}
      <div class="row">
        <div class="ans"><div class="label">Answer A</div>${aMd}</div>
        <div class="ans"><div class="label">Answer B</div>${bMd}</div>
      </div>
      <div class="vote" data-qid="${p.id}">
        <button data-v="A">A better</button>
        <button data-v="B">B better</button>
        <button data-v="tie">Tie</button>
        <button data-v="both_bad">Both bad</button>
        <span style="color:var(--muted); font-size:0.8rem;">notes:</span>
      </div>
      <textarea data-qid="${p.id}" placeholder="optional note">${escapeHtml(v.note || "")}</textarea>
      <div class="reveal" data-qid="${p.id}"></div>
    `;
    main.appendChild(card);
    const btns = card.querySelectorAll(".vote button");
    btns.forEach(b => {
      if (v.choice === b.dataset.v) b.classList.add("selected");
      b.addEventListener("click", () => {
        votes[p.id] = votes[p.id] || {};
        votes[p.id].choice = b.dataset.v;
        btns.forEach(x => x.classList.remove("selected"));
        b.classList.add("selected");
        save();
      });
    });
    card.querySelector("textarea").addEventListener("input", e => {
      votes[p.id] = votes[p.id] || {};
      votes[p.id].note = e.target.value;
      save();
    });
  });
  save();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

function computeStats() {
  let humanOpt = 0, humanBase = 0, tie = 0, bothBad = 0, rated = 0;
  let judgeOpt = 0, judgeBase = 0, judgeTie = 0;
  let agree = 0, compared = 0;
  DATA.forEach(p => {
    const v = votes[p.id]; if (!v || !v.choice) return;
    rated++;
    const chosenSide = v.choice === "A" ? p.mapping.A : v.choice === "B" ? p.mapping.B : null;
    if (chosenSide === "optimized") humanOpt++;
    else if (chosenSide === "baseline") humanBase++;
    else if (v.choice === "tie") tie++;
    else bothBad++;

    // LLM judge comparison via scalar fitness
    const fA = p.A.fitness, fB = p.B.fitness;
    let judgeWinner = null;
    if (Math.abs(fA - fB) < 0.02) judgeWinner = "tie";
    else judgeWinner = fA > fB ? p.mapping.A : p.mapping.B;
    if (judgeWinner === "optimized") judgeOpt++;
    else if (judgeWinner === "baseline") judgeBase++;
    else judgeTie++;

    if (chosenSide && judgeWinner && chosenSide !== "tie") {
      compared++;
      if ((chosenSide === "optimized" && judgeWinner === "optimized") ||
          (chosenSide === "baseline" && judgeWinner === "baseline") ||
          (v.choice === "tie" && judgeWinner === "tie")) agree++;
    }
  });
  statsEl.innerHTML =
    `<b>Human:</b> opt ${humanOpt} · base ${humanBase} · tie ${tie} · both bad ${bothBad} ` +
    `&nbsp;&nbsp; <b>Judge:</b> opt ${judgeOpt} · base ${judgeBase} · tie ${judgeTie} ` +
    `&nbsp;&nbsp; <b>Agreement:</b> ${compared ? (100*agree/compared).toFixed(0) : "—"}% (${agree}/${compared})`;
}

document.getElementById("reveal-btn").addEventListener("click", () => {
  revealed = true;
  DATA.forEach(p => {
    const el = document.querySelector(`.reveal[data-qid="${p.id}"]`);
    if (!el) return;
    el.classList.add("shown");
    const cls = s => `<span class="${s}">${s}</span>`;
    el.innerHTML = `A = ${cls(p.mapping.A)} (fitness ${p.A.fitness.toFixed(3)}) &nbsp;·&nbsp; B = ${cls(p.mapping.B)} (fitness ${p.B.fitness.toFixed(3)})`;
  });
  computeStats();
});

document.getElementById("export-btn").addEventListener("click", () => {
  const payload = DATA.map(p => ({
    id: p.id,
    question: p.question,
    category: p.category,
    mapping: p.mapping,
    vote: votes[p.id] || null,
    fitness: { A: p.A.fitness, B: p.B.fitness },
  }));
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `ab_votes_${new Date().toISOString().slice(0,10)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

document.getElementById("clear-btn").addEventListener("click", () => {
  if (!confirm("Clear all votes for this rater?")) return;
  votes = {};
  localStorage.removeItem(STORAGE_KEY);
  render();
});

document.getElementById("toggle-ref").addEventListener("change", e => {
  document.querySelectorAll(".card").forEach(c => c.classList.toggle("hide-ref", !e.target.checked));
});

render();
</script>
</body>
</html>
"""


def write_html(pairs):
    # Marked library handles markdown; we embed the data as a JS constant.
    payload = json.dumps(pairs, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__PAIRS_JSON__", payload)
    OUTPUT_HTML.write_text(html)


if __name__ == "__main__":
    main()
