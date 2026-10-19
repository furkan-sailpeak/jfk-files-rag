import json
import os

import pytest

# Blank every external credential BEFORE the app package is imported.
# load_dotenv() does not override variables that are already set, so the real
# .env cannot leak in: no LLM client, no embedding API, no database.
for var in ("GROQ_API_KEY", "OPENAI_API_KEY", "DATABASE_URL", "SUPABASE_URL", "SUPABASE_JWT_SECRET"):
    os.environ[var] = ""
os.environ.setdefault("EMBED_PROVIDER", "local")

from app import db, llm  # noqa: E402  (must follow the env setup above)


@pytest.fixture
def llm_replies(mocker):
    """llm_replies("client" | "judge_client", *texts): each model call returns
    the next text (dicts are sent as JSON). Returns the create() mock."""
    def install(attr, *replies):
        fake = mocker.patch.object(llm, attr)
        fake.chat.completions.create.side_effect = [
            mocker.MagicMock(choices=[mocker.MagicMock(message=mocker.MagicMock(
                content=json.dumps(r) if isinstance(r, dict) else r))])
            for r in replies
        ]
        return fake.chat.completions.create
    return install


@pytest.fixture
def failing_llm(mocker):
    """failing_llm("client" | "judge_client"): every model call raises."""
    def install(attr):
        fake = mocker.patch.object(llm, attr)
        fake.chat.completions.create.side_effect = RuntimeError("rate limited")
        return fake.chat.completions.create
    return install


@pytest.fixture
def mock_db(mocker):
    """mock_db(*fetchall_results): db.cursor() yields a mock cursor whose
    fetchall() returns the given results in order. Returns the cursor."""
    def install(*fetchall_results):
        cur = mocker.MagicMock()
        cur.fetchall.side_effect = list(fetchall_results)
        mocker.patch.object(db, "cursor").return_value.__enter__.return_value = cur
        return cur
    return install
