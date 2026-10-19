from app import generation, verification


def test_grounding_judge_sees_full_pages(llm_replies):
    # Regression: a 1500-char cap hid quoted passages and failed correct answers.
    create = llm_replies("judge_client", {"grounded": True, "reason": "ok"})
    page = "x" * 3000 + " strictly personal motivation"
    verdict = verification.check_answer_grounded("Ruby acted on personal motives [1].", [(page, "a.pdf", 2)], "Why?", "research")
    assert verdict == (True, "ok")
    assert "strictly personal motivation" in create.call_args.kwargs["messages"][0]["content"]


def test_grounding_judge_parses_string_booleans(llm_replies):
    # bool("false") is True; pydantic parses it correctly.
    llm_replies("judge_client", {"grounded": "false", "reason": "off-topic"})
    assert verification.check_answer_grounded("A [1].", [("c", "a.pdf", 1)], "q", "simple") == (False, "off-topic")


def test_grounding_judge_failure_is_permissive(failing_llm):
    failing_llm("judge_client")
    assert verification.check_answer_grounded("A [1].", [("c", "a.pdf", 1)], "q", "simple") == (True, "")


def test_grounding_skipped_for_non_rag_types(llm_replies):
    create = llm_replies("judge_client")
    assert verification.check_answer_grounded("hi", [], "q", "conversational") == (True, "")
    create.assert_not_called()


def test_verify_citations_keeps_only_scoped_verdicts(llm_replies):
    llm_replies("judge_client", {"unsupported": [
        {"unit": 1, "source": 2, "correct_source": 1},
        {"unit": 0, "source": 1, "correct_source": 9},     # out of range -> no re-point
        {"unit": 0, "source": 1, "correct_source": 1},     # same source -> no re-point
        3, {"unit": "x", "source": 1},                      # unscoped / malformed -> ignored
    ]})
    picked = [("c1", "a.pdf", 1), ("c2", "b.pdf", 1)]
    assert verification.verify_citations("A [1]. B [2].", picked) == [(1, 2, 1), (0, 1, None), (0, 1, None)]


def test_verifier_and_generator_see_identical_source_blocks(llm_replies):
    create = llm_replies("judge_client", {"unsupported": []})
    picked = [("first page", "a.pdf", 1), ("second page", "b.pdf", 7)]
    verification.verify_citations("A [2].", picked)
    assert generation.build_context(picked) in create.call_args.kwargs["messages"][0]["content"]


def test_verify_citations_skips_uncited_answers(llm_replies):
    create = llm_replies("judge_client")
    assert verification.verify_citations("No citations here.", [("c", "a.pdf", 1)]) == []
    create.assert_not_called()
