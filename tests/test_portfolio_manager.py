from pathlib import Path

from equity_research.agents.base import AgentOpinion
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient
from equity_research.orchestration.portfolio_manager import PM_SCHEMA, build_pm_prompt, run_pm


def _ops():
    return [
        AgentOpinion(agent="fundamentals", stance="bearish", score=-0.4, confidence=0.6,
                     rationale="pricey", key_facts=["P/E 40x"]),
        AgentOpinion(agent="technical", stance="bullish", score=0.5, confidence=0.7, rationale="uptrend"),
    ]


def test_pm_schema_has_reasoning_first():
    assert list(PM_SCHEMA["properties"])[0] == "reasoning"
    assert set(PM_SCHEMA["required"]) == {"reasoning", "score", "confidence", "narrative"}


def test_pm_prompt_lists_opinions_and_reference():
    prompt = build_pm_prompt("AAPL", 0.1, _ops())
    assert "AAPL" in prompt and "portfolio manager" in prompt.lower()
    assert "fundamentals" in prompt and "technical" in prompt
    assert "P/E 40x" in prompt          # key facts surfaced
    assert "+0.10" in prompt            # mechanical reference score shown


def test_run_pm_parses_json(tmp_path: Path):
    payload = ('{"reasoning":"analysts split; trend wins","score":0.3,'
               '"confidence":0.65,"narrative":"Cautiously constructive."}')
    client = OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                          chat_fn=lambda **k: {"message": {"content": payload}})
    out = run_pm(client, "AAPL", 0.1, _ops())
    assert out["score"] == 0.3 and out["narrative"] == "Cautiously constructive."
