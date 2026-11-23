"""Sync an eval dataset (rag/eval/datasets/*.json) to LangSmith and run the RAG
pipeline against it.

    python rag/eval/langsmith_eval.py sync                 # push the newest dataset
    python rag/eval/langsmith_eval.py run                  # sync, then run an experiment
    python rag/eval/langsmith_eval.py run --dataset dataset_jfk_2026-11-22 --validated-only --category fact

Each dataset file becomes its own LangSmith dataset, named "{project}-{date}".

The pipeline runs in-process (no server, no rate limiter), so it needs the
app's .env: DATABASE_URL, GROQ_API_KEY. Plus:
    LANGSMITH_API_KEY      required
    EVAL_JUDGE_PROVIDER    groq (default, uses GROQ_API_KEY), ollama (local) or openai (uses OPENAI_API_KEY)
    EVAL_JUDGE_MODEL       default: see JUDGE_DEFAULTS
    OLLAMA_URL             default http://localhost:11434 (start Ollama, then `ollama pull <model>`)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # rag/, so `import app` finds the backend
load_dotenv(HERE.parent.parent / ".env")
# The app's default embedding cache is the Docker path /app/models; outside
# Docker that is read-only and the vector leg of retrieval silently turns off.
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(HERE / ".models"))

import store  # noqa: E402
from langsmith import Client, evaluate  # noqa: E402
from openai import omit  # noqa: E402
from openai.types.shared_params import ResponseFormatJSONSchema  # noqa: E402

# EVAL_* names, not JUDGE_MODEL: the app already reads that for its own judge.
# groq default is the 20B, not the 120B that writes the answers, so the judge is not grading itself.
# ollama: gemma3:1b passed made-up answers in a 10-case check; use a 4B+ model there.
JUDGE_DEFAULTS = {"groq": "openai/gpt-oss-20b", "ollama": "gemma3:4b", "openai": "gpt-5.4-mini"}
JUDGE_PROVIDER = os.getenv("EVAL_JUDGE_PROVIDER", "groq")
if JUDGE_PROVIDER not in JUDGE_DEFAULTS:
    sys.exit(f"EVAL_JUDGE_PROVIDER must be one of {', '.join(JUDGE_DEFAULTS)}")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", JUDGE_DEFAULTS[JUDGE_PROVIDER])
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
CITATION = re.compile(r"\[\d+\]")
CITATION_NUM = re.compile(r"\[(\d+)\]")
CITED_CATEGORIES = {"fact", "person", "explanation", "timeline"}


# --- Sync ---------------------------------------------------------------------
def _example(row):
    return {
        "inputs": {"question": row["question"]},
        "outputs": {"answer": row["ground_truth_answer"], "expected_sources": row["expected_sources"]},
        "metadata": {"id": row["id"], "category": row["category"], "validated": row["validated"]},
    }


def langsmith_name(path):
    return "-".join(store.parse_name(path))


def sync(client, path):
    """Make the LangSmith dataset match the file, matched on metadata.id:
    create new rows, update changed ones, delete rows removed from the file."""
    name = langsmith_name(path)
    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
    else:
        ds = client.create_dataset(name, description=f"RAG eval, synced from rag/eval/datasets/{path.name}")
    remote = {(ex.metadata or {}).get("id"): ex for ex in client.list_examples(dataset_id=ds.id)}
    local = {r["id"]: r for r in store.load(path)}

    create, update = [], []
    for rid, row in local.items():
        want = _example(row)
        ex = remote.get(rid)
        if ex is None:
            create.append(want)
        elif (ex.inputs, ex.outputs, {k: (ex.metadata or {}).get(k) for k in want["metadata"]}) != (
            want["inputs"], want["outputs"], want["metadata"]
        ):
            update.append({"id": ex.id, **want})
    delete = [ex.id for rid, ex in remote.items() if rid not in local]

    if create:
        client.create_examples(dataset_id=ds.id, examples=create)
    if update:
        client.update_examples(dataset_id=ds.id, updates=update)
    if delete:
        client.delete_examples(delete)
    print(f"[sync] {path.name} -> {name}: {len(create)} created, {len(update)} updated, {len(delete)} deleted, {len(local)} total")
    return ds


# --- Target -------------------------------------------------------------------
def _parse_sse(chunks):
    """pipeline.run() yields SSE strings; keep the last 'done' or 'error'."""
    result = None
    for chunk in chunks:
        m = re.match(r"event: (\w+)\ndata: (.*)\n\n", chunk, re.S)
        if m and m.group(1) in ("done", "error"):
            result = (m.group(1), json.loads(m.group(2)))
    return result


def target(inputs):
    from app import pipeline  # imported lazily: loads the DB pool and embedder

    event, body = _parse_sse(pipeline.run(inputs["question"], [])) or ("error", {"message": "no final event"})
    if event == "error":
        raise RuntimeError(body.get("message"))
    sources = body.get("sources", [])
    # The pipeline returns the cited pages first, renumbered [1]..[M], followed
    # by every uncited retrieved page; keep the cited ones separately.
    cited_nums = sorted({int(n) for n in CITATION_NUM.findall(body["answer"])})
    return {
        "answer": body["answer"],
        "query_type": body.get("query_type"),
        "sources": sources,
        "cited_sources": [sources[n - 1] for n in cited_nums if 0 < n <= len(sources)],
        "timings": body.get("timings", {}),
    }


# --- Evaluators -----------------------------------------------------------------
# Generic on purpose: nothing here is specific to one project or domain, so the
# same judge works for any dataset with question / category / reference answer.
JUDGE_PROMPT = """You grade an assistant's answer against a reference answer.

Question category: {category}
Question: {question}

Reference answer (ground truth):
{reference}

Assistant's answer:
{answer}

The rule: an answer is wrong only if (a) it gets the core answer wrong or leaves it out, or
(b) something it says contradicts the reference. Leaving out anything beyond the core answer
NEVER makes it wrong. References are often longer than needed; most of their content is
supporting detail.

Step 1, core_answer: in ONE short sentence, the minimum the question asks for:
- "who is/was X": the person's role or identity (e.g. "a physicist", "the company's CFO").
  What they did, when they died, or other roles they had are supporting detail.
- "when / where / what / which / how many": that date, place, name or number.
- "why / how / what happened" (explanations): the main conclusion only.
- timelines: that the events asked about happened in this order. Each event the answer lists
  must have the right date and order; it does not need to list every event.
- expected behaviour (e.g. "should say the documents contain no information on this",
  "a short friendly reply"): that behaviour.

Step 2, verdict:
- yes: the core answer is there and nothing contradicts the reference. A less precise but
  consistent value is fine ("2019" for "March 3, 2019", "about 40%" for "41.7%").
- no: the core answer is missing or wrong; the answer is about a different person or thing;
  a statement contradicts the reference (a different date, name, place, number or outcome for
  the same fact); or it refuses although the reference has an answer.

Examples: "who was Marie Curie?" -> "a physicist and chemist" is yes, even if the reference
also lists her discoveries and prizes; "who were the Apollo 11 crew?" -> the three astronauts
is yes without the backup crew; "why did sales drop?" -> the main cause is yes without the
other factors the reference lists.

Two statements only contradict if they cannot both be true ("announced in 2018" and
"launched in 2020" can both be true). Ignore citation markers like [1] and formatting."""

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "core_answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["core_answer", "reasoning", "verdict"],
    "additionalProperties": False,
}

RESPONSE_FORMAT: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": {"name": "grade", "strict": True, "schema": GRADE_SCHEMA},
}

def _category(example):
    """The row's category. LangSmith passes evaluators only `run`, `example`,
    `inputs`, `outputs`, `reference_outputs` and `attachments` (other parameter
    names are left at their defaults), so metadata has to come from `example`."""
    return ((getattr(example, "metadata", None) or {}).get("category")) or "unknown"


_judge = None
VERDICT_SCORE = {"yes": 1, "no": 0}


def correctness(inputs, outputs, reference_outputs, example=None):
    global _judge
    if _judge is None:
        from openai import OpenAI
        if JUDGE_PROVIDER == "ollama":  # OpenAI-compatible endpoint; the key is ignored
            _judge = OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama")
        elif JUDGE_PROVIDER == "groq":
            _judge = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
        else:
            _judge = OpenAI()
    prompt = JUDGE_PROMPT.format(
        category=_category(example),
        question=inputs["question"],
        reference=reference_outputs["answer"],
        answer=outputs["answer"],
    )
    # Retry: the API sometimes rate-limits or returns output that fails its own
    # JSON-schema check (Groq: json_validate_failed); one miss would otherwise
    # leave the question without a score.
    for attempt in range(3):
        try:
            resp = _judge.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format=RESPONSE_FORMAT,
                # temperature 0 for repeatable verdicts; OpenAI's gpt-5 models only accept the default
                temperature=omit if JUDGE_PROVIDER == "openai" else 0,
            )
            grade = json.loads(resp.choices[0].message.content or "")
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt * 2)
    return {"key": "correctness", "score": VERDICT_SCORE[grade["verdict"]], "comment": f'{grade["verdict"]}: {grade["reasoning"]} (core: {grade["core_answer"]})'}


def refusal(outputs, example=None):
    """Off-topic questions must get the exact refusal line; questions the files
    answer must not. For not_in_files either can be right ("the documents don't
    contain this" is the refusal line), so the correctness judge decides those."""
    from app.config import REFUSAL_LINE

    category = _category(example)
    refused = REFUSAL_LINE in outputs["answer"]
    if category == "not_in_files":
        return {"key": "refusal_correct", "score": None, "comment": f"not applicable (refused={refused})"}
    should = category == "off_topic"
    return {"key": "refusal_correct", "score": int(refused == should),
            "comment": f"refused={refused}, expected={should}"}


def _recall(key, pages, reference_outputs):
    expected = reference_outputs.get("expected_sources") or []
    if not expected:
        return {"key": key, "score": None, "comment": "no expected sources"}
    got_pages = {(s.get("filename"), s.get("page")) for s in pages or []}
    got_files = {f for f, _ in got_pages}
    files = {s["file"] for s in expected}
    hit_pages = [s for s in expected if (s["file"], s["page"]) in got_pages]
    return {"key": key, "score": len(files & got_files) / len(files),
            "comment": f"files {len(files & got_files)}/{len(files)}, exact pages {len(hit_pages)}/{len(expected)}; "
                       f"missing: {', '.join(sorted(files - got_files)) or 'none'}"}


def source_recall(outputs, reference_outputs):
    """Share of the expected files the answer actually cites with [N]
    (file level; the comment also reports exact page hits)."""
    return _recall("source_recall", outputs.get("cited_sources"), reference_outputs)


def retrieval_recall(outputs, reference_outputs):
    """Share of the expected files among all retrieved pages, cited or not.
    High retrieval_recall with low source_recall = found but not cited."""
    return _recall("retrieval_recall", outputs.get("sources"), reference_outputs)


def cited(outputs, example=None):
    """Answers from the documents must carry [N] citations and sources."""
    if _category(example) not in CITED_CATEGORIES:
        return {"key": "has_citations", "score": None, "comment": "not applicable"}
    ok = bool(CITATION.search(outputs["answer"])) and bool(outputs.get("sources"))
    return {"key": "has_citations", "score": int(ok)}


# --- Run ----------------------------------------------------------------------
def _git_rev():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=HERE) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run(client, path, args):
    ds = sync(client, path)
    examples = [
        ex for ex in client.list_examples(dataset_id=ds.id)
        if (not args.validated_only or ex.metadata.get("validated"))
        and (not args.category or ex.metadata.get("category") in args.category)
    ]
    examples.sort(key=lambda ex: ex.metadata.get("id", ""))
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        sys.exit("No examples match the filters (did you validate any?).")

    from app import config

    rev = _git_rev()
    print(f"[run] {path.name}: {len(examples)} examples · commit {rev} · model {config.MODEL} · judge {JUDGE_PROVIDER}:{JUDGE_MODEL}")
    results = evaluate(
        target,
        data=examples,
        evaluators=[*([] if args.no_judge else [correctness]), refusal, cited, source_recall, retrieval_recall],
        experiment_prefix=args.prefix or f"rag-{rev}",
        description=args.description,
        metadata={"dataset_file": path.name, "commit": rev, "model": config.MODEL, "judge": f"{JUDGE_PROVIDER}:{JUDGE_MODEL}", "validated_only": args.validated_only, "categories": args.category or "all"},
        max_concurrency=args.concurrency,
        client=client,
    )
    _summary(results)


def _summary(results):
    scores = {}
    for r in results:
        for ev in r["evaluation_results"]["results"]:
            if ev.score is not None:
                scores.setdefault(ev.key, []).append(float(ev.score))
    print("\n[summary]")
    for key, vals in sorted(scores.items()):
        print(f"  {key:<16} {sum(vals) / len(vals):.3f}  (n={len(vals)})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["sync", "run"])
    p.add_argument("--dataset", help="file name or stem in rag/eval/datasets/ (default: the newest)")
    p.add_argument("--no-judge", action="store_true",
                   help="skip the correctness LLM judge (e.g. when it runs as a LangSmith UI evaluator)")
    p.add_argument("--validated-only", action="store_true", help="only examples ticked as validated")
    p.add_argument("--category", action="append", choices=store.CATEGORIES, help="repeatable")
    p.add_argument("--limit", type=int, help="first N examples (by id)")
    p.add_argument("--concurrency", type=int, default=2, help="parallel questions (Groq rate limits!)")
    p.add_argument("--prefix", help="experiment name prefix (default rag-<commit>)")
    p.add_argument("--description", help="experiment description, e.g. what you changed")
    args = p.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY is not set (add it to .env).")
    path = store.resolve(args.dataset)
    client = Client()
    sync(client, path) if args.command == "sync" else run(client, path, args)


if __name__ == "__main__":
    main()
