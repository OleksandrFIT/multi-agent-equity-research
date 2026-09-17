from datetime import date

import pytest
from pydantic import ValidationError

from equity_research.agents.base import AgentOpinion
from equity_research.data.models import Evidence


def test_evidence_defaults():
    e = Evidence(ticker="AAPL", as_of=date(2026, 9, 15), metrics={"pe": 30.0})
    assert e.context == []
    assert e.notes == []


def test_agent_opinion_score_bounds():
    with pytest.raises(ValidationError):
        AgentOpinion(agent="x", stance="bullish", score=2.0, confidence=0.5, rationale="r")


def test_agent_opinion_ok():
    o = AgentOpinion(agent="technical", stance="bearish", score=-0.4, confidence=0.7, rationale="r")
    assert o.key_facts == []


def test_agent_opinion_critique_defaults_none():
    from equity_research.agents.base import AgentOpinion

    op = AgentOpinion(agent="fundamentals", stance="bullish", score=0.5, confidence=0.7, rationale="r")
    assert op.critique is None
