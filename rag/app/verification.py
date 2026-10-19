"""Checks on a generated answer, both run by the judge model against the
same source blocks the generator saw:
  - grounding: is the answer on-subject and supported? (fail -> expansion)
  - citations: does each [N] support the sentence it is attached to?"""
from pydantic import ValidationError

from . import config, llm, prompts
from .generation import build_context
from .postprocess import HAS_CITATION, split_cite_units
from .schemas import CitationAudit, GroundingVerdict, UnsupportedCitation


def check_answer_grounded(answer, picked, rewritten_query, query_type):
    """Returns (grounded, reason). Fails open: if the judge errors, the
    answer is treated as grounded rather than triggering expansion."""
    if query_type not in ("simple", "research"):
        return True, ""
    # Each source is capped individually rather than truncating the joined
    # block: a global cap showed the judge only the first ~4 of 20 sources,
    # so any citation past [4] looked invented and grounding failed on almost
    # every research answer.
    prompt = prompts.render(
        "grounding-judge.txt",
        rewritten_query=rewritten_query,
        sources_block=build_context(picked, cap=config.JUDGE_SOURCE_CHARS),
        answer=answer[:6000],
    )
    try:
        verdict = llm.complete_json(prompt, GroundingVerdict)
        return verdict.grounded, verdict.reason
    except Exception as e:
        print(f"Grounding check skipped: {e}")
        return True, ""


def verify_citations(answer, picked):
    """Per-citation-instance check. Returns (unit_index, source, correct_source)
    triples for postprocess.strip_unsupported_citations; correct_source is the
    source that does contain the claim, or None."""
    if not picked or not HAS_CITATION.search(answer):
        return []
    units, _ = split_cite_units(answer)
    numbered, budget = [], config.VERIFY_ANSWER_CHARS
    for i, unit in enumerate(units):
        if not unit.strip():
            continue
        line = f"U{i}: {unit.strip()}"
        if len(line) > budget:
            break
        budget -= len(line)
        numbered.append(line)

    prompt = prompts.render(
        "citation-verify.txt",
        sources_block=build_context(picked, cap=config.JUDGE_SOURCE_CHARS),
        answer="\n".join(numbered),
    )
    try:
        audit = llm.complete_json(prompt, CitationAudit)
    except Exception as e:
        print(f"Citation verification skipped: {e}")
        return []

    verdicts = []
    for item in audit.unsupported:
        # Bare numbers are the old, unscoped verdict form: they name a source
        # but not which of its citations failed, and acting on one means
        # deleting them all. Ignore them; an unjudged citation is safer.
        try:
            v = UnsupportedCitation.model_validate(item)
        except ValidationError:
            print(f"[cite-verify] ignoring malformed verdict: {item!r}")
            continue
        fix = v.correct_source
        if fix is not None and not (1 <= fix <= len(picked) and fix != v.source):
            fix = None
        verdicts.append((v.unit, v.source, fix))
    return verdicts
