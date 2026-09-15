from __future__ import annotations


def company_name(ticker: str) -> str | None:
    """Best-effort company name for a ticker via yfinance; None on any failure.

    Used only to filter news for relevance, so it must never raise.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        name = info.get("shortName") or info.get("longName")
        return name or None
    except Exception:
        return None
