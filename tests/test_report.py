import json
from datetime import date

from equity_research.agents.base import AgentOpinion
from equity_research.orchestration.aggregator import Verdict
from equity_research.reporting.report import render_json, render_markdown


def _verdict():
    return Verdict(
        ticker="AAPL", as_of=date(2026, 9, 15), verdict="buy", score=0.42,
        confidence=0.7, narrative="Looks strong.",
        opinions=[AgentOpinion(agent="fundamentals", stance="bullish", score=0.6,
                               confidence=0.8, rationale="cheap", key_facts=["pe 20"])],
        skipped_agents=["sentiment"],
    )


def test_markdown_has_verdict_and_disclaimer():
    md = render_markdown(_verdict())
    assert "AAPL" in md
    assert "BUY" in md.upper()
    assert "not investment advice" in md.lower()
    assert "fundamentals" in md


def test_json_roundtrips():
    data = json.loads(render_json(_verdict()))
    assert data["verdict"] == "buy"
    assert data["opinions"][0]["agent"] == "fundamentals"
