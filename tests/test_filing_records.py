from datetime import date

from equity_research.rag.filing_records import (
    FilingSection,
    chunk_filing,
    filing_metadata,
    filing_parent_id,
)


def _section(text, section="risk_factors"):
    return FilingSection(ticker="AAPL", section=section, filed_at=date(2023, 11, 3), form="10-K", text=text)


def test_parent_id_stable_and_section_sensitive():
    a = filing_parent_id(_section("x"))
    b = filing_parent_id(_section("y"))
    c = filing_parent_id(_section("x", section="mda"))
    assert a == b
    assert a != c


def test_metadata_shape():
    m = filing_metadata(_section("x"))
    assert m["doc_type"] == "filing_section"
    assert m["ticker"] == "AAPL"
    assert m["date_int"] == 20231103
    assert m["section"] == "risk_factors"


def test_chunk_short_is_single_child_with_parent_id():
    pid, parent_text, children = chunk_filing(_section("Short risk section."))
    assert parent_text == "Short risk section."
    assert len(children) == 1
    _, meta = children[0]
    assert meta["parent_id"] == pid
    assert meta["doc_type"] == "filing_section"


def test_chunk_long_splits_into_multiple_children():
    long_text = " ".join(f"sentence{i}." for i in range(600))
    pid, parent_text, children = chunk_filing(_section(long_text))
    assert len(children) > 1
    assert all(m["parent_id"] == pid for _, m in children)
