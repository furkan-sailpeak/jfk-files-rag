"""Prompt files live in rag/prompts/ as Jinja2 templates ({{ var }}).
StrictUndefined makes a missing or misspelled variable raise instead of
silently leaving a placeholder in a prompt."""
import os

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Prompts sit next to the package (rag/prompts), not inside it.
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    autoescape=False,
)


def render(template, **variables):
    """Render rag/prompts/<template> with the given variables."""
    return _env.get_template(template).render(**variables)


def load(name):
    """Raw text of a prompt that takes no variables."""
    with open(os.path.join(PROMPTS_DIR, name)) as f:
        return f.read()
