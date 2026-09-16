from datetime import date

import pytest


@pytest.mark.integration
def test_fetch_filing_sections_live():
    """Requires network (SEC). Run: uv run pytest -m integration."""
    from equity_research.rag.filings import fetch_filing_sections

    sections = fetch_filing_sections("AAPL", date.today(), "Equity Research test@example.com")
    assert len(sections) >= 1
    assert all(len(s.text) > 200 for s in sections)  # real section bodies, not stubs
    assert {s.section for s in sections} <= {"risk_factors", "mda", "business"}
