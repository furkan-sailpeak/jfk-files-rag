"""Eval datasets: rag/eval/datasets/dataset_{project}_{YYYY-MM-DD}.json.

Every dataset is a JSON list of rows with the same fields:

    id                    q001, q002, ... (assigned on save)
    question              what a user would type
    category              one of CATEGORIES
    ground_truth_answer   the reference answer (or the expected behaviour)
    expected_sources      pages the answer should cite: [{"file": "104-10015-10001.pdf", "page": 12}];
                          empty when no citation is expected (off_topic, chit_chat, ...)
    validated             a human has checked this row
"""
import json
import os
import re
import tempfile
import tomllib
from datetime import date
from pathlib import Path

DATASETS_DIR = Path(__file__).with_name("datasets")
NAME = re.compile(r"^dataset_(?P<project>[a-z0-9][a-z0-9-]*)_(?P<date>\d{4}-\d{2}-\d{2})\.json$")
FIELDS = ["id", "question", "category", "ground_truth_answer", "expected_sources", "validated"]
# Source links come from the corpus config with the dataset's project name
# (rag/corpora/<project>.toml, documents.source_url), so a dataset for another
# corpus links to that corpus's documents.
CORPORA_DIR = Path(__file__).resolve().parent.parent / "corpora"
CATEGORIES = [
    "fact", "person", "explanation", "timeline",
    "not_in_files", "archive_stats", "off_topic", "chit_chat",
]


# --- Finding datasets ---------------------------------------------------------------
def list_datasets():
    """Dataset files, newest date first."""
    files = [p for p in DATASETS_DIR.glob("dataset_*.json") if NAME.match(p.name)]
    return sorted(files, key=lambda p: (parse_name(p)[1], p.name), reverse=True)


def parse_name(path):
    """(project, date) from a dataset file name."""
    m = NAME.match(Path(path).name)
    return (m["project"], m["date"]) if m else (Path(path).stem, "")


def dataset_path(project, day=None):
    project = re.sub(r"[^a-z0-9-]+", "-", project.strip().lower()).strip("-")
    if not project:
        raise ValueError("Project name must contain letters or digits.")
    return DATASETS_DIR / f"dataset_{project}_{(day or date.today()).isoformat()}.json"


def resolve(name=None):
    """A path, file name or stem in datasets/; default: the newest dataset."""
    if not name:
        found = list_datasets()
        if not found:
            raise FileNotFoundError(f"No datasets in {DATASETS_DIR}")
        return found[0]
    for cand in (Path(name), DATASETS_DIR / name, DATASETS_DIR / f"{name}.json"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Dataset not found: {name}")


# --- Reading and writing --------------------------------------------------------------
SOURCE_TEXT = re.compile(r"^\s*(?P<file>[^\s,#]+?)(?:\.pdf)?\s*(?:#page=|p\.?\s*|page\s*|\s)\s*(?P<page>\d+)\s*$", re.I)


def source_text(src):
    """{"file": "x.pdf", "page": 3} -> "x.pdf p.3" (the editable form in the table)."""
    return f"{src['file']} p.{src['page']}"


def parse_source(value):
    """A source dict, or text like "104-10015-10001.pdf p.12" / "104-10015-10001 12"."""
    if isinstance(value, dict):
        return {"file": _pdf(str(value["file"])), "page": int(value["page"])}
    m = SOURCE_TEXT.match(str(value))
    if not m:
        raise ValueError(f"Not a source: {value!r} (expected e.g. '104-10015-10001.pdf p.12')")
    return {"file": _pdf(m["file"]), "page": int(m["page"])}


def _pdf(name):
    name = name.strip()
    return name if name.lower().endswith(".pdf") else name + ".pdf"


def source_url(src, project):
    """Direct link to the page in the source PDF, or None when the project has
    no corpus config to take the link format from."""
    path = CORPORA_DIR / f"{project}.toml"
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        template = tomllib.load(f)["documents"]["source_url"]
    return template.format(file_id=src["file"][:-4]) + f"#page={src['page']}"


def _sources(value):
    if isinstance(value, str):
        value = [v for v in re.split(r"[,;\n]", value) if v.strip()]
    out = []
    for v in value or []:
        src = parse_source(v)
        if src not in out:
            out.append(src)
    return out


def normalize(row):
    return {
        "id": str(row.get("id") or "").strip(),
        "question": str(row.get("question") or "").strip(),
        "category": str(row.get("category") or "").strip(),
        "ground_truth_answer": str(row.get("ground_truth_answer") or "").strip(),
        "expected_sources": _sources(row.get("expected_sources")),
        "validated": bool(row.get("validated")),
    }


def load(path):
    with open(path, encoding="utf-8") as f:
        return [normalize(r) for r in json.load(f)]


def save(rows, path):
    """Drop rows without a question, give new rows an id, write atomically
    (a reader never sees half a file). Returns the rows as written."""
    rows = [normalize(r) for r in rows]
    n = max((int(r["id"][1:]) for r in rows if r["id"][1:].isdigit()), default=0)
    clean = []
    for r in rows:
        if not r["question"]:
            continue
        if not r["id"]:
            n += 1
            r["id"] = f"q{n:03d}"
        clean.append(r)
    fd, tmp = tempfile.mkstemp(dir=Path(path).parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return clean


def create(project, day=None, rows=None):
    """New dataset file, empty or with a copy of `rows`."""
    path = dataset_path(project, day)
    if path.exists():
        raise FileExistsError(f"{path.name} already exists.")
    DATASETS_DIR.mkdir(exist_ok=True)
    save(rows or [], path)
    return path
