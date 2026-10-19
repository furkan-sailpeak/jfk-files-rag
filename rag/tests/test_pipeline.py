"""Answer-path selection and the HTTP layer, with every stage mocked."""
import json

import pytest

from app import config, main, pipeline
from app.schemas import MetadataFilter, Route


def events(gen):
    """Parse SSE strings into (event, payload) pairs."""
    out = []
    for chunk in gen:
        head, data = chunk.strip().split("\n", 1)
        out.append((head.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def _route(mocker, **kw):
    return mocker.patch.object(pipeline, "route_query", return_value=Route(
        query_type=kw.get("query_type", "research"), rewritten_query="q", search_terms=["Ruby"],
        needs_retrieval=kw.get("needs_retrieval", True), metadata_filter=kw.get("metadata_filter")))


def test_doc_id_goes_to_document_path(mocker):
    path = mocker.patch.object(pipeline, "answer_document", return_value=iter([]))
    list(pipeline.run("Show me document 104-10004-10143", []))
    assert path.call_args.args[1] == "104-10004-10143"


def test_out_of_scope_returns_exact_refusal(mocker):
    mocker.patch.object(config, "REPLAY_BUDGET_S", 0)
    _route(mocker, query_type="out_of_scope")
    done = events(pipeline.run("Capital of Latvia?", []))[-1]
    assert done == ("done", {"answer": config.REFUSAL_LINE, "sources": [], "query_type": "out_of_scope",
                             "timings": done[1]["timings"]})


def test_metadata_with_unknown_column_falls_back_to_rag(mocker):
    _route(mocker, query_type="metadata", metadata_filter=MetadataFilter(column="drop table", value=True))
    rag = mocker.patch.object(pipeline, "answer_rag", return_value=iter([]))
    list(pipeline.run("q", []))
    assert rag.call_args.args[2].query_type == "simple"


def test_conversational_path(mocker):
    _route(mocker, query_type="conversational", needs_retrieval=False)
    conv = mocker.patch.object(pipeline, "answer_conversational", return_value=iter([]))
    list(pipeline.run("hi", []))
    conv.assert_called_once()


def test_rag_path_end_to_end_with_stages_mocked(mocker):
    mocker.patch.object(config, "REPLAY_BUDGET_S", 0)
    route = _route(mocker).return_value
    pages = [("Ruby owned the Carousel Club.", "a.pdf", 1), ("Other page.", "b.pdf", 2)]
    mocker.patch.object(pipeline, "hybrid_search", return_value=pages)
    mocker.patch.object(pipeline, "rerank", return_value=pages)
    mocker.patch.object(pipeline, "build_background_block", return_value="")
    mocker.patch.object(pipeline, "generate_answer", return_value="Ruby owned a club [2].")
    mocker.patch.object(pipeline, "check_answer_grounded", return_value=(True, ""))
    mocker.patch.object(pipeline, "verify_citations", return_value=[(0, 2, 1)])  # re-point [2] -> [1]
    _, done = events(pipeline.answer_rag("q", [], route, 0, {}))[-1]
    assert done["answer"] == "Ruby owned a club [1]."
    assert done["sources"][0] == {"filename": "a.pdf", "page": 1}


def test_stage_errors_become_an_error_event(mocker):
    mocker.patch.object(pipeline, "route_query", side_effect=RuntimeError("db down"))
    event, payload = events(pipeline.run("q", []))[-1]
    assert event == "error" and "db down" not in payload["message"]


# --- HTTP -------------------------------------------------------------------------
@pytest.fixture
def http():
    return main.app.test_client()


def test_chat_rejects_empty_query(http):
    assert http.post("/api/chat", json={"query": "  "}).status_code == 400


def test_chat_reports_missing_llm(http):
    assert http.post("/api/chat", json={"query": "Who was Jack Ruby?"}).status_code == 500


def test_analyze_rejects_unknown_action(http, mocker):
    mocker.patch.object(main.llm, "client")
    assert http.post("/api/analyze", json={"text": "x", "action": "delete"}).status_code == 400


def test_pdf_redirects_to_nara(http):
    res = http.get("/api/pdf/104-10004-10143.pdf")
    assert res.status_code == 302 and res.location.endswith("/104-10004-10143.pdf")


# --- Error text never reaches the client ---------------------------------------
SECRET_ERROR = 'connection to server at "db.internal" (10.9.9.9), port 6543 failed for user "postgres.REF"'


def test_chat_error_event_hides_exception_text(mocker):
    mocker.patch.object(pipeline, "route_query", side_effect=RuntimeError(SECRET_ERROR))
    event, payload = events(pipeline.run("q", []))[-1]
    assert event == "error" and "10.9.9.9" not in payload["message"] and "postgres" not in payload["message"]


def test_analyze_error_hides_exception_text(http, mocker):
    mocker.patch.object(main.llm, "client")
    mocker.patch.object(main.llm, "complete", side_effect=RuntimeError("Error code: 429 - org_SECRET Limit 200000"))
    res = http.post("/api/analyze", json={"text": "x", "action": "names"})
    assert res.status_code == 500 and "org_SECRET" not in res.get_data(as_text=True)


# --- Document path size cap ------------------------------------------------------
def test_within_budget_keeps_leading_pages_and_at_least_one():
    pages = [("x" * 100, "a.pdf", i) for i in range(1, 11)]
    assert [p[2] for p in pipeline._within_budget(pages, 350)] == [1, 2, 3]
    assert len(pipeline._within_budget([("x" * 999, "a.pdf", 1)], 10)) == 1


def test_document_path_truncates_and_tells_the_model(mocker):
    mocker.patch.object(config, "DOC_CONTEXT_CHARS", 250)
    pages = [(f"page {i} " + "x" * 100, "a.pdf", i) for i in range(1, 6)]
    mocker.patch.object(pipeline, "fetch_document", return_value=("a.pdf", pages))
    stream = mocker.patch.object(pipeline.llm, "stream", return_value=iter(["Answer [1]."]))
    _, done = events(pipeline.answer_document("summarize", "104-10004-10143", 0))[-1]
    user_msg = stream.call_args.args[0][1]["content"]
    assert "Only the first 2 of 5 pages" in user_msg and "page 3" not in user_msg
    assert [s["page"] for s in done["sources"]] == [1, 2]


# --- Security headers -------------------------------------------------------------
def test_security_headers_on_api_and_spa(http, mocker):
    mocker.patch.object(main.config, "CSP_ENFORCE", False)
    for res in (http.get("/api/pdf/x.pdf"), http.post("/api/chat", json={})):
        assert res.headers["X-Content-Type-Options"] == "nosniff"
        assert res.headers["X-Frame-Options"] == "DENY"
        assert "max-age=" in res.headers["Strict-Transport-Security"]
        assert "frame-ancestors 'none'" in res.headers["Content-Security-Policy-Report-Only"]
        assert "Content-Security-Policy" not in res.headers


def test_csp_enforced_when_flag_set(http, mocker):
    mocker.patch.object(main.config, "CSP_ENFORCE", True)
    res = http.get("/api/pdf/x.pdf")
    assert "img-src 'self' data:" in res.headers["Content-Security-Policy"]
