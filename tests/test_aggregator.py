from datetime import date
from pathlib import Path

from equity_research.agents.base import AgentOpinion
from equity_research.config import Config
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient
from equity_research.orchestration.aggregator import Aggregator


def _cfg():
    return Config(
        model="m", temperature=0.0, seed=1, cache_dir=".cache",
        edgar_user_agent="x x@x.com",
        weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2},
    )


def _client(tmp_path):
    return OllamaClient(
        model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
        chat_fn=lambda **k: {"message": {"content": "Narrative text."}},
    )


def test_weighted_score_normalizes_over_available(tmp_path):
    ops = [
        AgentOpinion(agent="fundamentals", stance="bullish", score=1.0, confidence=1.0, rationale="r"),
        AgentOpinion(agent="technical", stance="bearish", score=-1.0, confidence=1.0, rationale="r"),
    ]
    agg = Aggregator(_cfg(), _client(tmp_path))
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=["sentiment", "risk"])
    expected = (0.4 * 1.0 - 0.25 * 1.0) / 0.65
    assert abs(v.score - expected) < 1e-9
    assert v.skipped_agents == ["sentiment", "risk"]


def test_verdict_thresholds(tmp_path):
    agg = Aggregator(_cfg(), _client(tmp_path))
    buy = [AgentOpinion(agent="fundamentals", stance="bullish", score=0.8, confidence=0.9, rationale="r")]
    v = agg.aggregate("AAPL", date(2026, 9, 15), buy, skipped=["technical", "sentiment", "risk"])
    assert v.verdict == "buy"
    assert v.disclaimer  # non-empty disclaimer present


def test_narrative_from_client(tmp_path):
    agg = Aggregator(_cfg(), _client(tmp_path))
    ops = [AgentOpinion(agent="technical", stance="neutral", score=0.0, confidence=0.5, rationale="r")]
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=["fundamentals", "sentiment", "risk"])
    assert v.narrative == "Narrative text."


def test_all_agents_skipped_returns_hold(tmp_path):
    agg = Aggregator(_cfg(), _client(tmp_path))
    v = agg.aggregate("AAPL", date(2026, 9, 15), [], skipped=["fundamentals", "technical", "sentiment", "risk"])
    assert v.verdict == "hold"
    assert v.score == 0.0
    assert v.confidence == 0.0
    assert v.opinions == []


def test_unweighted_agent_treated_as_skipped(tmp_path):
    ops = [
        AgentOpinion(agent="fundamentals", stance="bullish", score=0.5, confidence=0.8, rationale="r"),
        AgentOpinion(agent="mystery", stance="bearish", score=-1.0, confidence=1.0, rationale="r"),
    ]
    agg = Aggregator(_cfg(), _client(tmp_path))
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=[])
    assert "mystery" in v.skipped_agents
    assert [o.agent for o in v.opinions] == ["fundamentals"]
    assert v.verdict == "buy"  # only fundamentals counts, score 0.5 >= 0.2
