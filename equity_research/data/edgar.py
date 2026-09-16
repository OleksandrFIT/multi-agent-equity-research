from __future__ import annotations

import math
import re
from datetime import date
from typing import Protocol


class NoFilingError(ValueError):
    pass


def _num(m: dict, key: str) -> float:
    """Numeric field or NaN — never raises on missing/None/non-numeric values."""
    try:
        f = float(m.get(key))
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")


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
        # Annual (10-K) only: a 10-Q reports a single quarter, which would distort
        # annual-scale metrics (P/E, revenue growth). The latest 10-K on or before
        # as_of is the conventional point-in-time fundamental snapshot.
        filings = company.get_filings(form=["10-K"])
        filing = select_filing_asof(list(filings), as_of)
        financials = filing.obj().financials
        m = financials.get_financial_metrics()
        net_income = _num(m, "net_income")
        shares = _num(m, "shares_outstanding_diluted")
        eps_ttm = net_income / shares if math.isfinite(net_income) and math.isfinite(shares) and shares else float("nan")
        return {
            "net_income": net_income,
            "revenue": _num(m, "revenue"),
            "revenue_prev": _prior_revenue(financials),
            "equity": _num(m, "stockholders_equity"),
            "total_debt": _num(m, "total_liabilities"),
            "eps_ttm": eps_ttm,
            "operating_income": _num(m, "operating_income"),
            "free_cash_flow": _num(m, "free_cash_flow"),
            "current_ratio": _num(m, "current_ratio"),
        }
