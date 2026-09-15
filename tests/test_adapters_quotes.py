import pandas as pd

from equity_research.data import adapters


def test_fetch_quotes_maps_price_and_change(monkeypatch):
    idx = pd.date_range("2026-09-10", periods=3, freq="D")
    frame = pd.DataFrame({("Close", "AAPL"): [100.0, 110.0, 121.0],
                          ("Close", "MSFT"): [200.0, 200.0, 210.0]}, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    monkeypatch.setattr(adapters, "_yf_download", lambda tickers: frame)
    out = adapters.fetch_quotes(["AAPL", "MSFT"])
    assert out["AAPL"] == {"price": 121.0, "change_pct": 10.0}   # (121-110)/110
    assert out["MSFT"] == {"price": 210.0, "change_pct": 5.0}


def test_fetch_quotes_skips_missing(monkeypatch):
    idx = pd.date_range("2026-09-10", periods=1, freq="D")  # only one row -> no prev close
    frame = pd.DataFrame({("Close", "AAPL"): [100.0]}, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    monkeypatch.setattr(adapters, "_yf_download", lambda tickers: frame)
    assert adapters.fetch_quotes(["AAPL"]) == {}
