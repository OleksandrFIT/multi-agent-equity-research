from datetime import date
from pathlib import Path

from equity_research.agents.base import AgentOpinion
from equity_research.agents.judge import judge_evidence
from equity_research.data.models import Evidence
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient


def _client(tmp_path: Path, content: str) -> OllamaClient:
    return OllamaClient(
        model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
        chat_fn=lambda **k: {"message": {"content": content}},
    )


def _evidence():
    return Evidence(ticker="AAPL", as_of=date(2026, 9, 15), metrics={"pe": 30.0})


def test_reconcile_flips_contradictory_stance():
    from equity_research.agents.judge import reconcile_stance
    op = AgentOpinion(agent="fundamentals", stance="bullish", score=-0.5, confidence=0.7, rationale="r")
    assert reconcile_stance(op).stance == "bearish"


def test_reconcile_leaves_consistent_stance():
    from equity_research.agents.judge import reconcile_stance
    op = AgentOpinion(agent="technical", stance="bullish", score=0.6, confidence=0.7, rationale="r")
    assert reconcile_stance(op).stance == "bullish"


def test_reconcile_leaves_neutral_small_score():
    from equity_research.agents.judge import reconcile_stance
    op = AgentOpinion(agent="sentiment", stance="neutral", score=0.05, confidence=0.5, rationale="r")
    assert reconcile_stance(op).stance == "neutral"


def test_judge_parses_opinion(tmp_path):
    client = _client(
        tmp_path,
        '{"reasoning":"cheap versus peers","stance":"bullish","score":0.6,"confidence":0.8,'
        '"rationale":"cheap","key_facts":["pe 30"]}',
    )
    op = judge_evidence("fundamentals", _evidence(), client)
    assert op.agent == "fundamentals"
    assert op.stance == "bullish"
    assert op.score == 0.6
    assert not hasattr(op, "reasoning")  # scratchpad stripped, not on the model


def test_judge_degrades_to_neutral_on_bad_output(tmp_path):
    client = OllamaClient(
        model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
        chat_fn=lambda **k: {"message": {"content": "not json at all"}},
        max_retries=2,
    )
    op = judge_evidence("technical", _evidence(), client)
    assert op.stance == "neutral"
    assert op.confidence == 0.0
    assert "degraded" in op.rationale.lower()
