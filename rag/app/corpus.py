"""Corpus config: rag/corpora/<name>.toml holds everything specific to one
document collection (prompt wording, examples, scope, table, document IDs,
source URLs). The prompts are generic templates that read it as {{ corpus.* }}.

load() validates the file up front, so a missing or malformed value fails at
startup instead of producing a prompt with a hole in it or bad SQL."""
import os
import re
import tomllib

CORPORA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpora")

# section -> {key: type}. Every key is required.
SCHEMA = {
    "database": {"table": str, "metadata_columns": list},
    "documents": {"id_pattern": str, "id_example": str, "source_url": str},
    "assistant": {"name": str, "persona": str, "persona_title": str, "research_persona": str},
    "collection": {
        "short_name": str, "description": str, "archive_name": str, "answer_topic": str,
        "research_topic": str, "search_description": str, "vocabulary_sources": str,
        "document_noun": str, "boilerplate": str, "boilerplate_example": str,
    },
    "scope": {
        "subject": str, "out_of_scope_subject": str, "related_terms": str, "generous": str,
        "metadata_properties": str, "metadata_examples": list, "metadata_filter_examples": list,
    },
    "examples": {
        "history_subject": str, "history_followup": str, "history_followup_keyword": str,
        "rewrite_question": str, "rewrite_topic": str, "rewrite_result": str,
        "multiword_terms": list, "keywords": list,
    },
    "entities": {"context": str, "source": str, "scope": str, "examples": list},
}

# Interpolated into SQL, so only plain identifiers are allowed.
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class CorpusError(ValueError):
    pass


def available():
    return sorted(f[:-5] for f in os.listdir(CORPORA_DIR) if f.endswith(".toml"))


def load(name):
    path = os.path.join(CORPORA_DIR, f"{name}.toml")
    if not os.path.isfile(path):
        raise CorpusError(f"Unknown corpus {name!r}; available: {', '.join(available())}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    _validate(name, data)
    data["id"] = name
    return data


def _validate(name, data):
    problems = []
    for section, keys in SCHEMA.items():
        values = data.get(section)
        if not isinstance(values, dict):
            problems.append(f"missing section [{section}]")
            continue
        for key, typ in keys.items():
            if key not in values:
                problems.append(f"missing {section}.{key}")
            elif not isinstance(values[key], typ) or (typ is str and not values[key].strip()):
                problems.append(f"{section}.{key} must be a non-empty {typ.__name__}")
    if problems:
        raise CorpusError(f"corpora/{name}.toml: " + "; ".join(problems))

    db, docs = data["database"], data["documents"]
    for ident in [db["table"], *db["metadata_columns"]]:
        if not isinstance(ident, str) or not _IDENTIFIER.match(ident):
            problems.append(f"not a plain SQL identifier: {ident!r}")
    try:
        if re.compile(docs["id_pattern"]).groups != 1:
            problems.append("documents.id_pattern needs exactly one capture group")
    except re.error as e:
        problems.append(f"documents.id_pattern is not a valid regex: {e}")
    if "{file_id}" not in docs["source_url"]:
        problems.append("documents.source_url must contain {file_id}")
    for ex in data["examples"]["keywords"]:
        if not (isinstance(ex, dict) and isinstance(ex.get("question"), str) and isinstance(ex.get("terms"), list)):
            problems.append("examples.keywords entries need question (string) and terms (list)")
            break
    if problems:
        raise CorpusError(f"corpora/{name}.toml: " + "; ".join(problems))
