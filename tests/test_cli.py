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
        def __init__(self, agents, aggregator):
            captured["agents"] = agents

        def run(self, ticker, as_of):
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    cli_module.analyze_ticker("AAPL", date(2026, 9, 15), "config.yaml")
    names = [a.name for a in captured["agents"]]
    assert "risk" in names
    assert "fundamentals" in names and "technical" in names


def test_analyze_ticker_includes_sentiment_agent(monkeypatch):
    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator):
            captured["agents"] = agents

        def run(self, ticker, as_of):
            from equity_research.orchestration.aggregator import Verdict
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    cli_module.analyze_ticker("AAPL", date(2026, 9, 15), "config.yaml")
    names = [a.name for a in captured["agents"]]
    assert "sentiment" in names
    assert {"fundamentals", "technical", "risk"} <= set(names)
