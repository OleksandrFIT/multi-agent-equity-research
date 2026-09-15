from datetime import date

from equity_research.agents.base import AgentOpinion
from equity_research.agents.grounding import ground
from equity_research.data.models import Evidence


def _ev(metrics=None, context=None):
    return Evidence(ticker="AAPL", as_of=date(2026, 9, 15),
                    metrics=metrics or {}, context=context or [])


def test_drops_fact_with_unseen_number():
    op = AgentOpinion(agent="fundamentals", stance="bullish", score=0.5, confidence=0.8,
                      rationale="r", key_facts=["P/E is 44.6", "Made-up revenue of 999999"])
    ev = _ev(metrics={"pe": 44.6})
    grounded = ground(op, ev)
    assert "P/E is 44.6" in grounded.key_facts
    assert "Made-up revenue of 999999" not in grounded.key_facts
    assert any("dropped" in n.lower() for n in grounded.dropped_facts)


def test_grounds_fact_by_context_substring():
    op = AgentOpinion(agent="sentiment", stance="bearish", score=-0.3, confidence=0.6,
                      rationale="r", key_facts=["Analysts cite supply constraints"])
    ev = _ev(context=["Reuters: Apple faces supply constraints in Q4."])
    grounded = ground(op, ev)
    assert "Analysts cite supply constraints" in grounded.key_facts


def test_fact_without_numbers_or_evidence_is_dropped():
    op = AgentOpinion(agent="technical", stance="bullish", score=0.4, confidence=0.7,
                      rationale="r", key_facts=["The stock will definitely moon"])
    grounded = ground(op, _ev(metrics={"rsi14": 55.0}))
    assert grounded.key_facts == []
