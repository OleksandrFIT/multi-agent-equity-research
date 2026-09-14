from datetime import date
from pathlib import Path

from equity_research.agents.base import AgentOpinion
from equity_research.config import Config
from equity_research.data.models import Evidence
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient
from equity_research.orchestration.aggregator import Aggregator
from equity_research.orchestration.orchestrator import Orchestrator


class StubAgent:
    def __init__(self, name, score, fail=False):
        self.name = name
        self._score = score
        self._fail = fail

    def gather(self, ticker, as_of):
        if self._fail:
            raise RuntimeError("data unavailable")
        return Evidence(ticker=ticker, as_of=as_of, metrics={})

    def judge(self, evidence):
        return AgentOpinion(agent=self.name, stance="neutral", score=self._score,
                            confidence=0.5, rationale="r")


def _agg(tmp_path):
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".cache",
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    client = OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                          chat_fn=lambda **k: {"message": {"content": "n"}})
    return Aggregator(cfg, client)


def test_runs_all_agents(tmp_path):
    agents = [StubAgent("fundamentals", 0.5), StubAgent("technical", -0.1)]
    orch = Orchestrator(agents=agents, aggregator=_agg(tmp_path))
    v = orch.run("AAPL", as_of=date(2026, 9, 15))
    assert len(v.opinions) == 2
    assert v.skipped_agents == []


def test_failed_agent_is_skipped(tmp_path):
    agents = [StubAgent("fundamentals", 0.5), StubAgent("technical", 0.0, fail=True)]
    orch = Orchestrator(agents=agents, aggregator=_agg(tmp_path))
    v = orch.run("AAPL", as_of=date(2026, 9, 15))
    assert [o.agent for o in v.opinions] == ["fundamentals"]
    assert v.skipped_agents == ["technical"]
