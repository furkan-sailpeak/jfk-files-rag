"""Groq clients and the three ways the pipeline calls a model: plain text,
streamed text, and JSON parsed into a pydantic model.

`client`/`judge_client` are looked up at call time, so tests can patch them."""
from groq import Groq

from . import config

client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None
judge_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None
if client:
    print(f"LLM provider: Groq, model={config.MODEL}; judges: {config.JUDGE_MODEL}")
else:
    print("WARNING: GROQ_API_KEY not found; /api/chat will 500.")


def reasoning_kwargs(model_name):
    """gpt-oss models emit chain-of-thought before the answer. Left at the
    defaults that costs real latency (0.40s to first token vs 0.14s) and,
    worse, the hidden reasoning still counts against max_tokens — a small cap
    can be consumed entirely by reasoning, returning empty content.

    'hidden' keeps reasoning out of the response, 'low' keeps it short.
    Returns {} for non-reasoning models so this stays a no-op after a swap.
    """
    if "gpt-oss" in model_name:
        return {"reasoning_effort": "low", "reasoning_format": "hidden"}
    return {}


def _target(judge):
    if judge:
        return judge_client, config.JUDGE_MODEL
    return client, config.MODEL


def complete(messages, *, judge=False, temperature=0, **kwargs):
    """One non-streamed chat completion; returns the reply text."""
    llm, model = _target(judge)
    res = llm.chat.completions.create(
        model=model, **reasoning_kwargs(model),
        messages=messages, temperature=temperature, **kwargs,
    )
    return res.choices[0].message.content or ""


def stream(messages, *, temperature=0):
    """Yield reply text chunks from the main model. If streaming fails before
    anything was produced, fall back to one non-streamed call so the request
    still gets an answer."""
    started = False
    try:
        chunks = client.chat.completions.create(
            model=config.MODEL, **reasoning_kwargs(config.MODEL),
            messages=messages, temperature=temperature, stream=True,
        )
        for chunk in chunks:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                started = True
                yield delta
    except Exception as e:
        if started:
            raise
        print(f"Streaming failed, falling back to non-streaming: {e}")
        yield complete(messages, temperature=temperature)


def complete_json(prompt, schema, *, judge=True, temperature=0, strict_schema=False):
    """Ask for JSON and parse it into `schema` (a pydantic model). Raises on
    API errors and on replies that don't fit -- callers decide the fallback.

    strict_schema=True sends the model's JSON schema for constrained decoding
    instead of plain JSON mode (needed where gpt-oss otherwise mangles arrays).
    """
    if strict_schema:
        response_format = {"type": "json_schema", "json_schema": {
            "name": schema.model_config.get("title") or schema.__name__,
            "strict": True,
            "schema": schema.model_json_schema(),
        }}
    else:
        response_format = {"type": "json_object"}
    text = complete(
        [{"role": "user", "content": prompt}],
        judge=judge, temperature=temperature, response_format=response_format,
    )
    return schema.model_validate_json(text)
