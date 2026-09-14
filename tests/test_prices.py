from datetime import date

import pandas as pd
import pytest

from equity_research.data.prices import PriceProvider, PriceValidationError


def _frame(closes, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=idx)


def test_returns_validated_history():
    y = _frame([100.0, 101.0, 102.0])
    provider = PriceProvider(fetch_yfinance=lambda t: y, fetch_stooq=lambda t: y)
    hist, note = provider.history("AAPL", as_of=date(2025, 1, 3))
    assert list(hist["Close"]) == [100.0, 101.0, 102.0]
    assert note is None


def test_flags_cross_source_discrepancy():
    y = _frame([100.0, 101.0, 102.0])
    s = _frame([100.0, 101.0, 130.0])  # >10% off on last close
    provider = PriceProvider(fetch_yfinance=lambda t: y, fetch_stooq=lambda t: s)
    hist, note = provider.history("AAPL", as_of=date(2025, 1, 3))
    assert note is not None and "discrepancy" in note.lower()


def test_rejects_empty_history():
    provider = PriceProvider(fetch_yfinance=lambda t: _frame([]), fetch_stooq=lambda t: _frame([]))
    with pytest.raises(PriceValidationError):
        provider.history("AAPL", as_of=date(2025, 1, 3))


def test_as_of_filters_future_rows():
    y = _frame([100.0, 101.0, 102.0, 103.0])
    provider = PriceProvider(fetch_yfinance=lambda t: y, fetch_stooq=lambda t: y)
    hist, _ = provider.history("AAPL", as_of=date(2025, 1, 2))
    assert len(hist) == 2  # only rows up to and including as_of
