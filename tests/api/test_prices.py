import pandas as pd

import api.core as core


def test_prices_maps_period_and_builds_series(monkeypatch):
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    df = pd.DataFrame({"Close": [100.0 + i for i in range(400)]}, index=idx)
    monkeypatch.setattr(core, "_fetch_long_close", lambda ticker: df["Close"])
    out = core.prices("aapl", "3M")
    assert out["ticker"] == "AAPL"
    assert out["period"] == "3M"
    assert len(out["candles"]) == 63
    assert len(out["sma200"]) == 63


def test_prices_invalid_period_defaults_to_6m(monkeypatch):
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    df = pd.DataFrame({"Close": [100.0 + i for i in range(400)]}, index=idx)
    monkeypatch.setattr(core, "_fetch_long_close", lambda ticker: df["Close"])
    out = core.prices("AAPL", "bogus")
    assert out["period"] == "6M"
    assert len(out["candles"]) == 126
