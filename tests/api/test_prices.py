import pandas as pd
from fastapi.testclient import TestClient

import api.core as core
import api.main as main


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


def test_prices_endpoint(monkeypatch):
    monkeypatch.setattr(core, "prices", lambda ticker, period: {
        "ticker": ticker.upper(), "period": period,
        "candles": [{"time": "2024-01-02", "value": 101.0}],
        "sma50": [], "sma200": [], "rsi": [],
    })
    client = TestClient(main.app)
    resp = client.get("/api/prices?ticker=aapl&period=6M")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["period"] == "6M"
    assert body["candles"][0]["value"] == 101.0
