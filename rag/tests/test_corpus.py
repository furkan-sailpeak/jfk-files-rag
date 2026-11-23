"""Corpus config: every corpus file loads, bad files fail loudly, and the
prompts are generic (no collection-specific text outside corpora/*.toml)."""
import copy
import os
import re

import pytest
from jinja2 import meta

from app import config, corpus, prompts

PROMPT_FILES = sorted(f for f in os.listdir(prompts.PROMPTS_DIR) if f.endswith(".j2"))

# Words that only belong in the JFK corpus file. If one shows up in a prompt
# rendered with an unrelated corpus, a template still hard-codes JFK content.
JFK_TERMS = re.compile(
    r"jfk|kennedy|oswald|ruby|warren|\bcia\b|\bfbi\b|cold war|assassinat|declassif|"
    r"mexico|soviet|104-10|historian", re.I)


def _fake_corpus():
    """The JFK corpus with every value swapped for neutral placeholder text."""
    fake = copy.deepcopy({k: v for k, v in config.CORPUS.items() if k != "id"})
    for section, values in fake.items():
        for key, value in values.items():
            if isinstance(value, str):
                values[key] = f"<{section}.{key}>"
            elif key == "keywords":
                values[key] = [{"question": "<example question>", "terms": ["<term one>", "<term two>"]}]
            else:
                values[key] = [f"<{section}.{key} {i}>" for i in range(2)]
    fake["database"] = {"table": "acme_pages", "metadata_columns": ["has_tables", "department"]}
    fake["documents"] = {"id_pattern": r"\b(ACME-\d{4})\b", "id_example": "ACME-0001",
                         "source_url": "https://docs.example.com/{file_id}.pdf"}
    return fake


def _render_all():
    out = {}
    for name in PROMPT_FILES:
        variables = meta.find_undeclared_variables(prompts._env.parse(prompts.source(name)))
        out[name] = prompts.render(name, **{v: "x" for v in variables - set(prompts._env.globals)})
    return out


@pytest.mark.parametrize("name", corpus.available())
def test_every_corpus_file_loads(name):
    assert corpus.load(name)["id"] == name


def test_fake_corpus_is_valid():
    corpus._validate("fake", _fake_corpus())


def test_prompts_are_generic(monkeypatch):
    """Rendered with an unrelated corpus, no prompt mentions JFK content."""
    monkeypatch.setitem(prompts._env.globals, "corpus", _fake_corpus())
    leaks = {name: JFK_TERMS.findall(text) for name, text in _render_all().items() if JFK_TERMS.search(text)}
    assert not leaks, f"collection-specific text in templates: {leaks}"


def test_corpus_values_reach_the_prompts(monkeypatch):
    monkeypatch.setitem(prompts._env.globals, "corpus", _fake_corpus())
    rendered = _render_all()
    assert "<assistant.name>" in rendered["conversational.j2"]
    assert "ACME-0001" in rendered["conversational.j2"]
    assert '"<term one>", "<term two>"' in rendered["router.j2"]
    assert '"has_tables", "department"' in rendered["router.j2"]
    assert '- "<entities.examples 1>"' in rendered["entity-brief.j2"]


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["assistant"].pop("name"), "missing assistant.name"),
    (lambda c: c.pop("scope"), "missing section [scope]"),
    (lambda c: c["collection"].update(short_name="  "), "collection.short_name must be a non-empty str"),
    (lambda c: c["database"].update(table="pages; DROP TABLE x"), "not a plain SQL identifier"),
    (lambda c: c["database"].update(metadata_columns=["ok", "bad column"]), "not a plain SQL identifier"),
    (lambda c: c["documents"].update(id_pattern=r"\d+"), "exactly one capture group"),
    (lambda c: c["documents"].update(id_pattern="(unclosed"), "not a valid regex"),
    (lambda c: c["documents"].update(source_url="https://x/{id}.pdf"), "must contain {file_id}"),
    (lambda c: c["examples"].update(keywords=[{"question": "q"}]), "need question (string) and terms (list)"),
])
def test_invalid_corpus_fails_loudly(mutate, message):
    bad = _fake_corpus()
    mutate(bad)
    with pytest.raises(corpus.CorpusError, match=re.escape(message)):
        corpus._validate("bad", bad)


def test_unknown_corpus_lists_the_available_ones():
    with pytest.raises(corpus.CorpusError, match="available: .*jfk"):
        corpus.load("does-not-exist")


def test_backend_settings_come_from_the_corpus():
    assert config.PAGES_TABLE == config.CORPUS["database"]["table"]
    assert config.DOC_ID.search(f"see {config.CORPUS['documents']['id_example']} please")
    assert "{file_id}" in config.SOURCE_URL
