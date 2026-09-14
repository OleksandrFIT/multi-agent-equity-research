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
    result = runner.invoke(cli_module.app, ["AAPL"])
    assert result.exit_code == 0
    assert "AAPL" in result.stdout
    assert "HOLD" in result.stdout.upper()
