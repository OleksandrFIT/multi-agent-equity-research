from fastapi.testclient import TestClient

import api.core as core
import api.main as main


def test_health(monkeypatch):
    monkeypatch.setattr(core, "health", lambda: {"ok": True, "model": "qwen2.5:7b", "ollama_reachable": True})
    client = TestClient(main.app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["ollama_reachable"] is True
    assert resp.json()["model"] == "qwen2.5:7b"


def test_analyze_streams_agents_then_verdict(monkeypatch):
    from datetime import date

    from equity_research.agents.base import AgentOpinion
    from equity_research.orchestration.aggregator import Verdict

    def fake_run(ticker, on_event):
        on_event({"agent": "technical",
                  "opinion": AgentOpinion(agent="technical", stance="bullish", score=0.5,
                                          confidence=0.8, rationale="r")})
        on_event({"agent": "sentiment", "skipped": True, "reason": "no news"})
        return Verdict(ticker=ticker, as_of=date(2026, 9, 15), verdict="buy", score=0.5,
                       confidence=0.8, narrative="n", opinions=[], skipped_agents=["sentiment"])

    monkeypatch.setattr(core, "run_analyze", fake_run)
    client = TestClient(main.app)
    body = client.get("/api/analyze?ticker=AAPL").text
    assert "event: agent" in body
    assert "technical" in body and "skipped" in body
    assert "event: verdict" in body
    assert '"verdict": "buy"' in body


def test_ingest(monkeypatch):
    monkeypatch.setattr(core, "ingest_ticker", lambda ticker: 7)
    client = TestClient(main.app)
    resp = client.post("/api/ingest", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    assert resp.json() == {"ticker": "AAPL", "ingested": 7}


def test_backtest_config(monkeypatch):
    monkeypatch.setattr(core, "backtest_config",
                        lambda: {"universe": ["AAPL"], "dates": ["2024-03-15"], "horizons": [21, 63]})
    client = TestClient(main.app)
    resp = client.get("/api/backtest/config")
    assert resp.json()["horizons"] == [21, 63]


def test_backtest_streams_progress_then_report(monkeypatch):
    from datetime import date

    from equity_research.eval.backtest import BacktestRecord

    def fake_run(on_progress):
        on_progress({"ticker": "AAPL", "as_of": "2024-03-15"})
        recs = [BacktestRecord("AAPL", date(2024, 3, 15), "buy", 0.5, {21: 0.1, 63: 0.2})]
        return recs, [21, 63]

    monkeypatch.setattr(core, "run_backtest_records", fake_run)
    client = TestClient(main.app)
    body = client.get("/api/backtest").text
    assert "event: progress" in body and "AAPL" in body
    assert "event: report" in body
    assert '"21"' in body
