import api.core as core
from equity_research.data.resolve import resolve_ticker


def test_resolve_valid_ticker_passes_through():
    out = resolve_ticker("aapl", llm_guess=lambda q: "ZZZ", price_ok=lambda t: t == "AAPL")
    assert out == {"input": "aapl", "resolved": "AAPL", "corrected": False}


def test_resolve_corrects_typo_via_llm():
    out = resolve_ticker("APPL", llm_guess=lambda q: "AAPL", price_ok=lambda t: t == "AAPL")
    assert out == {"input": "APPL", "resolved": "AAPL", "corrected": True}


def test_resolve_none_when_llm_none_or_invalid():
    assert resolve_ticker("ZZZZ", llm_guess=lambda q: None, price_ok=lambda t: False)["resolved"] is None
    assert resolve_ticker("ZZZZ", llm_guess=lambda q: "QQQQ", price_ok=lambda t: t == "AAPL")["resolved"] is None


def test_quotes_caches_by_day(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(tickers):
        calls["n"] += 1
        return {t: {"price": 1.0, "change_pct": 0.5} for t in tickers}

    monkeypatch.setattr("equity_research.data.adapters.fetch_quotes", fake_fetch)
    core._QUOTES_CACHE.clear()
    a = core.quotes(["AAPL", "MSFT"])
    b = core.quotes(["AAPL", "MSFT"])  # served from cache
    assert calls["n"] == 1
    assert a == b == [{"ticker": "AAPL", "price": 1.0, "change_pct": 0.5},
                      {"ticker": "MSFT", "price": 1.0, "change_pct": 0.5}]


from fastapi.testclient import TestClient

import api.main as main


def test_quotes_endpoint(monkeypatch):
    monkeypatch.setattr(core, "quotes", lambda tickers: [{"ticker": tickers[0], "price": 1.0, "change_pct": 2.0}])
    client = TestClient(main.app)
    resp = client.get("/api/quotes?tickers=aapl,msft")
    assert resp.status_code == 200
    assert resp.json()[0]["ticker"] == "AAPL"


def test_resolve_endpoint(monkeypatch):
    monkeypatch.setattr(core, "resolve", lambda q: {"input": q, "resolved": "AAPL", "corrected": True})
    client = TestClient(main.app)
    assert client.get("/api/resolve?query=APPL").json()["resolved"] == "AAPL"
