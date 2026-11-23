"""Prompt files live in rag/prompts/ as Jinja2 templates ({{ var }}). They are
generic: everything about the document collection comes from the corpus config
(corpora/<CORPUS>.toml), available in every template as {{ corpus.* }}.
StrictUndefined makes a missing or misspelled variable raise instead of
silently leaving a placeholder in a prompt."""
import os

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import config

# Prompts sit next to the package (rag/prompts), not inside it.
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    autoescape=False,
)
_env.globals["corpus"] = config.CORPUS


def render(template, **variables):
    """Render rag/prompts/<template> with the given variables."""
    return _env.get_template(template).render(**variables)


def source(name):
    """Raw template text (for tests and tooling)."""
    with open(os.path.join(PROMPTS_DIR, name)) as f:
        return f.read()


def load(name):
    """A prompt that takes no per-request variables (it may still use corpus.*)."""
    return render(name)
