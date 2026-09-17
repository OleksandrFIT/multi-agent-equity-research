from datetime import date
from pathlib import Path

from equity_research.agents.base import AgentOpinion
from equity_research.agents.critic import CRITIC_SCHEMA, build_critic_prompt, critique, make_critic
from equity_research.data.models import Evidence
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient


def _client(tmp_path: Path, content: str) -> OllamaClient:
    return OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                        chat_fn=lambda **k: {"message": {"content": content}})


def _ev():
    return Evidence(ticker="AAPL", as_of=date(2026, 9, 15), metrics={"pe": 30.0})


def _op(confidence=0.8):
    return AgentOpinion(agent="fundamentals", stance="bullish", score=0.5,
                        confidence=confidence, rationale="cheap versus peers", key_facts=["P/E 30x"])


def test_schema_reasoning_first():
    assert list(CRITIC_SCHEMA["properties"])[0] == "reasoning"
    assert set(CRITIC_SCHEMA["required"]) == {"reasoning", "supported", "confidence", "issue"}


def test_prompt_has_opinion_and_metrics():
    p = build_critic_prompt("fundamentals", _ev(), _op())
    assert "AAPL" in p and "bullish" in p and "cheap versus peers" in p
    assert "P/E ratio" in p  # metric rendered via _format_metrics


def test_critique_only_lowers_confidence(tmp_path):
    client = _client(tmp_path, '{"reasoning":"ok","supported":true,"confidence":0.99,"issue":""}')
    out = critique(client, "fundamentals", _ev(), _op(confidence=0.6))
    assert out.confidence == 0.6  # critic cannot raise it
    assert out.critique is None


def test_critique_lowers_and_notes_when_unsupported(tmp_path):
    client = _client(tmp_path, '{"reasoning":"weak","supported":false,"confidence":0.2,"issue":"rationale not backed by metrics"}')
    out = critique(client, "fundamentals", _ev(), _op(confidence=0.8))
    assert out.confidence == 0.2
    assert "not backed" in out.critique


def test_critique_unchanged_on_failure(tmp_path):
    client = OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                          chat_fn=lambda **k: {"message": {"content": "not json"}}, max_retries=1)
    op = _op(confidence=0.7)
    out = critique(client, "fundamentals", _ev(), op)
    assert out.confidence == 0.7 and out.critique is None


def test_make_critic_skips_risk(tmp_path):
    client = _client(tmp_path, '{"reasoning":"x","supported":false,"confidence":0.0,"issue":"bad"}')
    c = make_critic(client)
    risk = AgentOpinion(agent="risk", stance="neutral", score=0.6, confidence=0.6, rationale="r")
    out = c("risk", _ev(), risk)
    assert out.confidence == 0.6 and out.critique is None  # untouched
