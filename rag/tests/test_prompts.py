"""Cross-prompt consistency: every template the code references exists and
renders, and the prompts agree with each other on the answer contract."""
import glob
import os
import re

import pytest
from jinja2 import UndefinedError, meta

from app import config, prompts

PROMPT_FILES = sorted(f for f in os.listdir(prompts.PROMPTS_DIR) if f.endswith(".txt"))
APP_SOURCE = "".join(open(p).read() for p in glob.glob(os.path.join(os.path.dirname(prompts.__file__), "*.py")))


def _referenced(fn):
    return set(re.findall(rf'prompts\.{fn}\(\s*["\']([^"\']+\.txt)["\']', APP_SOURCE))


def test_every_referenced_prompt_exists():
    referenced = _referenced("render") | _referenced("load")
    assert referenced, "no prompt references found -- regex out of date?"
    assert referenced <= set(PROMPT_FILES), referenced - set(PROMPT_FILES)


def test_every_prompt_file_is_used():
    # Any quoted "<name>.txt" in the package counts: some names are chosen in
    # expressions (simple vs research) or looked up from a dict (/api/analyze).
    mentioned = set(re.findall(r'["\']([\w-]+\.txt)["\']', APP_SOURCE))
    assert not set(PROMPT_FILES) - mentioned, set(PROMPT_FILES) - mentioned


@pytest.mark.parametrize("name", PROMPT_FILES)
def test_templates_parse_and_use_only_jinja_placeholders(name):
    source = prompts.load(name)
    prompts._env.parse(source)  # raises on a syntax error
    assert not re.search(r"(?<!\{)\{[a-z_]+\}(?!\})", source), "old-style {placeholder} left in template"


def test_missing_variable_raises():
    with pytest.raises(UndefinedError):
        prompts.render("router.txt")


@pytest.mark.parametrize("name", PROMPT_FILES)
def test_render_calls_pass_exactly_the_template_variables(name):
    variables = meta.find_undeclared_variables(prompts._env.parse(prompts.load(name)))
    for call in re.findall(rf'prompts\.render\(\s*["\']{re.escape(name)}["\'](.*?)\)\s*(?:\+|$|\n)', APP_SOURCE, re.S):
        passed = set(re.findall(r"(\w+)\s*=", call))
        assert variables <= passed, f"{name}: missing {variables - passed}"


@pytest.mark.parametrize("name", PROMPT_FILES)
def test_prompts_use_the_exact_refusal_line(name):
    for line in re.findall(r"The retrieved documents do not contain[^\n\"]*", prompts.load(name)):
        assert line.rstrip(".") == config.REFUSAL_LINE.rstrip("."), name


@pytest.mark.parametrize("name", PROMPT_FILES)
def test_prompts_do_not_promise_a_category_input(name):
    assert not re.search(r"\bcategory\b", prompts.load(name), re.I)


def test_answer_prompts_agree_on_timeline_table():
    for name in ("rag-simple.txt", "rag-research.txt", "rag-format-simple.txt", "rag-format-research.txt"):
        assert "Date | Event | Source" in prompts.load(name), name
