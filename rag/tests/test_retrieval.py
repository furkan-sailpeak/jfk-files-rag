from app import config, db, retrieval


def test_tokenize_terms_drops_stopwords_and_caps():
    terms = retrieval.tokenize_terms("What did the CIA know about Oswald in Mexico City during the autumn months?")
    assert "the" not in terms and "What" not in terms
    assert terms[:3] == ["CIA", "know", "Oswald"]
    assert len(terms) <= 6


# --- Full-text leg ----------------------------------------------------------------
def test_fts_keeps_and_query_when_enough_hits(mock_db):
    rows = [(f"c{i}", f"f{i}.pdf", 1) for i in range(config.FTS_MIN_HITS)]
    cur = mock_db(rows)
    assert retrieval.fts_search("Jack Ruby Oswald", ["Jack Ruby", "Oswald"]) == rows
    assert cur.execute.call_count == 1


def test_fts_falls_back_to_or_and_dedupes(mock_db):
    and_rows = [("c1", "f1.pdf", 1)]
    or_rows = [("c1", "f1.pdf", 1), ("c2", "f2.pdf", 3)]
    cur = mock_db(and_rows, or_rows)
    out = retrieval.fts_search("Kostikov Mexico City", ["Kostikov", "Mexico City"])
    assert out == [("c1", "f1.pdf", 1), ("c2", "f2.pdf", 3)]
    or_sql, or_params = cur.execute.call_args_list[1].args
    assert or_sql.count("||") == 2  # once in the rank expr, once in WHERE
    assert or_params == ["Kostikov", "Mexico City", "Kostikov", "Mexico City"]


def test_fts_no_or_fallback_for_single_term(mock_db):
    cur = mock_db([])
    assert retrieval.fts_search("Kostikov", ["Kostikov"]) == []
    assert cur.execute.call_count == 1


# --- Vector leg -------------------------------------------------------------------
def test_vector_search_sets_index_params_and_uses_configured_column(mocker, mock_db):
    mocker.patch.object(retrieval, "embed_query", return_value=[0.1] * config.EMBED_DIM)
    cur = mock_db([("c", "a.pdf", 1)])
    assert retrieval.vector_search("q") == [("c", "a.pdf", 1)]
    sqls = [c.args[0] for c in cur.execute.call_args_list]
    assert any("ivfflat.probes" in s for s in sqls) and any("hnsw.ef_search" in s for s in sqls)
    assert f"ORDER BY {config.EMBED_COLUMN} <=>" in sqls[-1]
    assert f"FROM {config.PAGES_TABLE}" in sqls[-1] and "{" not in sqls[-1]


def test_vector_search_skips_db_when_embedding_fails(mocker):
    mocker.patch.object(retrieval, "embed_query", return_value=None)
    cursor = mocker.patch.object(db, "cursor")
    assert retrieval.vector_search("q") == []
    cursor.assert_not_called()


# --- Rerank -----------------------------------------------------------------------
def test_rerank_snippet_centres_on_hit_across_ocr_newlines():
    content = "ROUTING SLIP " * 100 + "report on Jack\nRuby and his club"
    assert "Jack Ruby" in retrieval._rerank_snippet(content, ["Jack Ruby"])


def test_dedupe_picks_drops_near_duplicates_and_backfills():
    cands = [("same text " * 30, "a.pdf", 1), ("same text " * 30, "b.pdf", 1), ("other", "c.pdf", 1), ("third", "d.pdf", 1)]
    out = retrieval._dedupe_picks(cands, [0, 1, 2], 3, [3])
    assert [r[1] for r in out] == ["a.pdf", "c.pdf", "d.pdf"]


def test_rerank_uses_model_order(llm_replies):
    create = llm_replies("client", {"indices": [4, 2, 99, 0]})
    cands = [(f"text {i}", f"f{i}.pdf", 1) for i in range(5)]
    assert [r[1] for r in retrieval.rerank(cands, "q", 3, ["text"])] == ["f4.pdf", "f2.pdf", "f0.pdf"]
    # The reply shape is enforced with a strict JSON schema, not plain JSON mode.
    fmt = create.call_args.kwargs["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True


def test_rerank_falls_back_to_retrieval_order_on_error(failing_llm):
    failing_llm("client")
    cands = [(f"text {i}", f"f{i}.pdf", 1) for i in range(5)]
    assert retrieval.rerank(cands, "q", 3, ["text"]) == cands[:3]


def test_rerank_rejects_malformed_reply(llm_replies):
    llm_replies("client", {"indices": "024681012"})
    cands = [(f"text {i}", f"f{i}.pdf", 1) for i in range(5)]
    assert retrieval.rerank(cands, "q", 3, ["text"]) == cands[:3]


# --- Expansion --------------------------------------------------------------------
def test_expand_and_retrieve_merges_new_pages(mocker, llm_replies):
    llm_replies("judge_client", {"queries": ["Ruby motive", " ", "Ruby Dallas"]})
    search = mocker.patch.object(retrieval, "hybrid_search", return_value=[("new page", "n.pdf", 1)])
    out = retrieval.expand_and_retrieve("q", "reason", [("seed", "s.pdf", 1)])
    assert out == [("seed", "s.pdf", 1), ("new page", "n.pdf", 1)]
    assert search.call_count == 2  # blank query skipped


# --- Direct lookups ---------------------------------------------------------------
def test_fetch_document_prefers_suffixed_variant_when_needed(mock_db):
    cur = mock_db([("p1", "104-10004-10143_redacted.pdf", 1)])
    cur.fetchone.return_value = ("104-10004-10143_redacted",)
    filename, pages = retrieval.fetch_document("104-10004-10143")
    assert filename == "104-10004-10143_redacted.pdf" and len(pages) == 1


def test_fetch_document_not_found(mock_db):
    cur = mock_db()
    cur.fetchone.return_value = None
    assert retrieval.fetch_document("999-99999-99999") == ("999-99999-99999.pdf", [])
