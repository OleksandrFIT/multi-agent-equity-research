from __future__ import annotations

from datetime import date

from equity_research.data.edgar import select_filing_asof
from equity_research.rag.filing_records import FilingSection


def fetch_filing_sections(ticker: str, as_of: date, user_agent: str) -> list[FilingSection]:
    """Extract Item 1 (Business) + Item 1A (Risk Factors) + Item 7 (MD&A) from the latest 10-K <= as_of.

    edgartools section attribute names are version-dependent — verify live and
    adjust only this function if they differ.
    """
    from edgar import Company, set_identity

    set_identity(user_agent)
    filings = Company(ticker).get_filings(form=["10-K"])
    filing = select_filing_asof(list(filings), as_of)
    tenk = filing.obj()
    filed = filing.filing_date
    out: list[FilingSection] = []
    for attr, name in [("risk_factors", "risk_factors"), ("management_discussion", "mda"), ("business", "business")]:
        text = getattr(tenk, attr, None)
        if text:
            out.append(FilingSection(ticker=ticker, section=name, filed_at=filed, form="10-K", text=str(text)))
    return out
