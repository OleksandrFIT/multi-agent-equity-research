from __future__ import annotations

import re
from datetime import date
from typing import Protocol


class NoFilingError(ValueError):
    pass


def select_filing_asof(filings, as_of):
    """Return the most recent filing with filing_date <= as_of, or raise NoFilingError."""
    eligible = [f for f in filings if f.filing_date <= as_of]
    if not eligible:
        raise NoFilingError(f"no filing on or before {as_of}")
    return max(eligible, key=lambda f: f.filing_date)


class FactsSource(Protocol):
    def company_facts(self, ticker: str, as_of: date) -> dict[str, float]: ...


def _prior_revenue(financials) -> float:
    """Prior fiscal-year revenue from the income statement, or 0.0 if unavailable.

    Best-effort and defensive: any change in the edgartools dataframe shape
    yields 0.0 (so revenue_growth degrades to NaN) rather than raising.
    """
    try:
        df = financials.income_statement().to_dataframe()
        period_cols = [c for c in df.columns if re.match(r"\d{4}-\d{2}-\d{2}", str(c))]
        if len(period_cols) < 2:
            return 0.0
        rev = df[df["standard_concept"] == "Revenue"]
        if rev.empty:
            return 0.0
        return float(rev.iloc[0][period_cols[1]])
    except Exception:
        return 0.0


class EdgarProvider:
    """Thin wrapper around edgartools, normalizing XBRL facts to a flat dict.

    Uses the high-level Financials accessors (get_financial_metrics) rather than
    parsing statement rows by tag, which is far more robust across filings.

    Note: `total_debt` is populated from total liabilities (edgartools exposes no
    dedicated long-term-debt accessor), so `debt_to_equity` is really a
    liabilities-to-equity leverage proxy. Refine when a debt-specific concept is
    wired in.
    """

    def __init__(self, user_agent: str):
        self.user_agent = user_agent

    def company_facts(self, ticker: str, as_of: date) -> dict[str, float]:
        from edgar import Company, set_identity

        set_identity(self.user_agent)
        company = Company(ticker)
        filings = company.get_filings(form=["10-K", "10-Q"])
        filing = select_filing_asof(list(filings), as_of)
        financials = filing.obj().financials
        m = financials.get_financial_metrics()

        net_income = float(m["net_income"])
        diluted_shares = float(m["shares_outstanding_diluted"])
        eps_ttm = net_income / diluted_shares if diluted_shares else 0.0

        return {
            "net_income": net_income,
            "revenue": float(m["revenue"]),
            "revenue_prev": _prior_revenue(financials),
            "equity": float(m["stockholders_equity"]),
            "total_debt": float(m["total_liabilities"]),
            "eps_ttm": eps_ttm,
        }
