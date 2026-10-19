import pytest

from app import config
from app.postprocess import (
    cite_table_source_columns,
    drop_bulk_citations,
    join_cite_units,
    normalize_citations,
    remap_citations,
    split_cite_units,
    strip_artifacts,
    strip_doc_echo,
    strip_unsupported_citations,
)

REFUSAL = config.REFUSAL_LINE


# --- Citation syntax ------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Claim 【6】.", "Claim [6]."),
    ("Claim 【2†L31-L38】.", "Claim [2]."),
    ("Claim ［11］.", "Claim [11]."),
    ("Claim [ 3 ].", "Claim [3]."),
])
def test_normalize_citations_variants(raw, expected):
    assert normalize_citations(raw) == expected


def test_normalize_citations_keeps_bracketed_years():
    assert normalize_citations("Filed [1963] in Dallas [2].") == "Filed [1963] in Dallas [2]."


def test_strip_artifacts_removes_reasoning_leaks():
    text = "Step 1: think\nThe final answer is: 42\nOswald was in Dallas [1]."
    assert strip_artifacts(text) == "Oswald was in Dallas [1]."


def test_strip_artifacts_unbolds_refusal():
    assert strip_artifacts(f"**{REFUSAL}**") == REFUSAL


def test_strip_artifacts_drops_citations_on_refusal():
    assert strip_artifacts(f"{REFUSAL} [1][2]") == REFUSAL


def test_strip_doc_echo_removes_question_echo():
    out = strip_doc_echo("USER INQUIRY: summarize this\n\n## Overview\nA memo [1].")
    assert out.startswith("## Overview")


# --- Renumbering ----------------------------------------------------------------
def test_remap_citations_renumbers_and_reorders_sources():
    sources = [{"filename": f"f{i}.pdf", "page": i} for i in range(1, 9)]
    text, cited = remap_citations("A [7]. B [2]. C [7][8].", sources)
    assert text == "A [2]. B [1]. C [2][3]."
    assert [s["page"] for s in cited] == [2, 7, 8]


def test_remap_citations_ignores_out_of_range():
    text, cited = remap_citations("A [1]. B [99].", [{"filename": "a.pdf", "page": 1}])
    assert cited == [{"filename": "a.pdf", "page": 1}]
    assert "[1]" in text


# --- Citation units and repair ----------------------------------------------------
@pytest.mark.parametrize("text", [
    "One [1]. Two [2]!\n\n- Three [3]\n- Four? [4]",
    "No.  12 file. Trailing space ",
    "",
])
def test_cite_unit_split_is_lossless(text):
    units, seps = split_cite_units(text)
    assert join_cite_units(units, seps) == text


def test_strip_unsupported_citations_is_scoped_to_one_unit():
    answer = "Ruby owned a club [2]. He shot Oswald [2][5]."
    units, _ = split_cite_units(answer)
    target = next(i for i, u in enumerate(units) if "shot" in u)
    assert strip_unsupported_citations(answer, [(target, 2)]) == "Ruby owned a club [2]. He shot Oswald [5]."


def test_strip_unsupported_citations_repoints_to_correct_source():
    answer = "Ruby acted on personal motives [3]. He owned a club [3]."
    out = strip_unsupported_citations(answer, [(0, 3, 1)])
    assert out == "Ruby acted on personal motives [1]. He owned a club [3]."


def test_strip_unsupported_repoint_does_not_cascade_or_duplicate():
    # [2]->[5] must not then be removed by the [5] verdict, and [5][5] collapses.
    assert strip_unsupported_citations("Claim [2][5].", [(0, 2, 5), (0, 5, None)]) == "Claim [5]."
    assert strip_unsupported_citations("Claim [2][4].", [(0, 2, 4)]) == "Claim [4]."


def test_tidy_handles_narrow_no_break_space():
    # The model often writes U+202F before a citation.
    answer = "Instruction to assassinate President Kennedy [5]."
    assert strip_unsupported_citations(answer, [(0, 5, None)]) == "Instruction to assassinate President Kennedy."


def test_drop_bulk_citations():
    out = drop_bulk_citations("Other files mention Ruby [1][2][3][4]. Fact [5][6].")
    assert out == "Other files mention Ruby. Fact [5][6]."


# --- Source tables ----------------------------------------------------------------
TABLE = """Intro [1].

| Source | File Identifier | Key Points |
|---|---|---|
| 1 | Interview 1966 | Denies training |
| 3 | Memo dated 27 Sept 1963 | Visa attempts |
| 25 | Out of range | Left alone |
| [2] | Already cited | Kept |

After table 4 stays plain."""


def test_cite_table_source_columns_wraps_bare_numbers():
    out = cite_table_source_columns(TABLE, n_sources=20)
    assert "| [1] | Interview 1966 |" in out
    assert "| [3] | Memo dated 27 Sept 1963 |" in out
    assert "| 25 | Out of range |" in out          # beyond the source list
    assert "| [2] | Already cited |" in out       # not double-wrapped
    assert "27 Sept 1963" in out                   # other columns untouched
    assert out.endswith("After table 4 stays plain.")


def test_cite_table_source_columns_ignores_tables_without_source_column():
    table = "| Date | Event |\n|---|---|\n| 1963 | 3 visits |"
    assert cite_table_source_columns(table, 20) == table


def test_table_rows_become_separate_verifier_units():
    units, _ = split_cite_units(cite_table_source_columns(TABLE, n_sources=20))
    assert any(u.startswith("| [3] |") for u in units)
