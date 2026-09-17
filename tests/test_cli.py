from datetime import date
from pathlib import Path

from typer.testing import CliRunner

import equity_research.cli as cli_module
from equity_research.agents.base import AgentOpinion
from equity_research.data.models import Evidence
from equity_research.orchestration.aggregator import Verdict


def test_analyze_prints_markdown(monkeypatch, tmp_path):
    def fake_build(ticker, as_of, cfg_path):
        v = Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0, confidence=0.5,
                    narrative="n", opinions=[AgentOpinion(agent="technical", stance="neutral",
                    score=0.0, confidence=0.5, rationale="r")], skipped_agents=[])
        return v

    monkeypatch.setattr(cli_module, "analyze_ticker", fake_build)
    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["analyze", "AAPL"])
    assert result.exit_code == 0
    assert "AAPL" in result.stdout
    assert "HOLD" in result.stdout.upper()


def test_analyze_ticker_includes_risk_agent(monkeypatch):
    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator, critic=None):
            captured["agents"] = agents

        def run(self, ticker, as_of, on_event=None):
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    cli_module.analyze_ticker("AAPL", date(2026, 9, 15), "config.yaml")
    names = [a.name for a in captured["agents"]]
    assert "risk" in names
    assert "fundamentals" in names and "technical" in names


def test_analyze_ticker_wraps_fetchers_resiliently(monkeypatch):
    import pandas as pd

    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator, critic=None):
            captured["agents"] = agents

        def run(self, ticker, as_of, on_event=None):
            from equity_research.orchestration.aggregator import Verdict
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    calls = {"n": 0}

    def flaky(_t):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("blip")
        idx = pd.date_range("2025-01-01", periods=3, freq="D")
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    monkeypatch.setattr(cli_module, "fetch_yfinance", flaky)
    cli_module.analyze_ticker("AAPL", date(2026, 9, 15), "config.yaml")

    tech = next(a for a in captured["agents"] if a.name == "technical")
    # the unknown-ticker preflight inside analyze_ticker already exercised the
    # wrapped fetcher (and its retry) once; reset the counter to isolate the
    # wiring check below to a single fresh retry cycle
    calls["n"] = 0
    df = tech.prices.fetch_yfinance("AAPL")  # the wired (resilient) fetcher
    assert df.iloc[-1]["Close"] == 3.0
    assert calls["n"] == 2  # retried once after the first failure


def test_analyze_ticker_includes_sentiment_agent(monkeypatch):
    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator, critic=None):
            captured["agents"] = agents

        def run(self, ticker, as_of, on_event=None):
            from equity_research.orchestration.aggregator import Verdict
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    cli_module.analyze_ticker("AAPL", date(2026, 9, 15), "config.yaml")
    names = [a.name for a in captured["agents"]]
    assert "sentiment" in names
    assert {"fundamentals", "technical", "risk"} <= set(names)


def test_analyze_ticker_fundamentals_has_filing_retriever(monkeypatch):
    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator, critic=None):
            captured["agents"] = agents

        def run(self, ticker, as_of, on_event=None):
            from equity_research.orchestration.aggregator import Verdict
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    cli_module.analyze_ticker("AAPL", date(2026, 9, 15), "config.yaml")
    fund = next(a for a in captured["agents"] if a.name == "fundamentals")
    assert fund.filing_retriever is not None
    assert fund.filing_ingest_fn is not None


def test_backtest_builder_excludes_sentiment(monkeypatch):
    import equity_research.cli as cli_module
    from equity_research.config import Config

    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator, critic=None):
            captured["agents"] = [a.name for a in agents]

        def run(self, ticker, as_of):
            from equity_research.orchestration.aggregator import Verdict
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    cfg = Config.load("config.yaml")

    class FakeClient:
        cache = None
        seed = 1
        temperature = 0.0
        chat_fn = staticmethod(lambda **k: {"message": {"content": "{}"}})

    run_verdict = cli_module.build_backtest_verdict(cfg, FakeClient())
    run_verdict("AAPL", date(2024, 3, 15))
    assert "sentiment" not in captured["agents"]
    assert {"fundamentals", "technical", "risk"} == set(captured["agents"])


def test_unknown_ticker_verdict_short_circuits():
    from datetime import date

    from equity_research.cli import _unknown_ticker_verdict
    from equity_research.data.prices import PriceValidationError

    class FakePrices:
        def history(self, ticker, as_of):
            raise PriceValidationError(f"no price history for {ticker}")

    v = _unknown_ticker_verdict(FakePrices(), "APPL", date(2026, 9, 15))
    assert v is not None
    assert v.status == "unknown_ticker"
    assert v.opinions == []


def test_known_ticker_preflight_returns_none():
    from datetime import date

    from equity_research.cli import _unknown_ticker_verdict

    class FakePrices:
        def history(self, ticker, as_of):
            return ("df", None)  # any non-raising result

    assert _unknown_ticker_verdict(FakePrices(), "AAPL", date(2026, 9, 15)) is None


def test_transient_price_error_falls_through_to_none():
    from datetime import date

    from equity_research.cli import _unknown_ticker_verdict

    class FakePrices:
        def history(self, ticker, as_of):
            raise RuntimeError("boom")

    assert _unknown_ticker_verdict(FakePrices(), "AAPL", date(2026, 9, 15)) is None
