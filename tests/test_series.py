import pandas as pd

from equity_research.analytics.series import build_price_series


def _close(n: int) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series([100.0 + i for i in range(n)], index=idx)


def test_candles_sliced_to_period_and_formatted():
    import re

    out = build_price_series(_close(300), period_days=63)
    assert len(out["candles"]) == 63
    pt = out["candles"][-1]
    assert set(pt) == {"time", "value"}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", pt["time"])  # YYYY-MM-DD
    assert isinstance(pt["value"], float)
    assert pt["value"] == 399.0  # last of 300 points: 100.0 + 299


def test_sma_and_rsi_defined_within_window_only():
    out = build_price_series(_close(300), period_days=63)
    assert len(out["sma50"]) == 63
    assert len(out["sma200"]) == 63
    assert len(out["rsi"]) == 63
    assert all(set(p) == {"time", "value"} for p in out["sma200"])


def test_sma200_empty_when_history_too_short():
    out = build_price_series(_close(120), period_days=63)
    assert out["sma200"] == []
    assert len(out["sma50"]) == 63
    assert len(out["candles"]) == 63


def test_window_larger_than_history_returns_all_points():
    out = build_price_series(_close(30), period_days=63)
    assert len(out["candles"]) == 30


def test_empty_close_returns_empty_series():
    out = build_price_series(pd.Series(dtype=float), period_days=63)
    assert out == {"candles": [], "sma50": [], "sma200": [], "rsi": []}
