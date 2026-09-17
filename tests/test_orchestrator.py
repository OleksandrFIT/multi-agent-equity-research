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


def test_failed_agent_reason_captured(tmp_path):
    agents = [StubAgent("fundamentals", 0.5), StubAgent("technical", 0.0, fail=True)]
    orch = Orchestrator(agents=agents, aggregator=_agg(tmp_path))
    v = orch.run("AAPL", as_of=date(2026, 9, 15))
    assert "technical" in v.skip_reasons
    assert "data unavailable" in v.skip_reasons["technical"]


def test_failed_agent_is_skipped(tmp_path):
    agents = [StubAgent("fundamentals", 0.5), StubAgent("technical", 0.0, fail=True)]
    orch = Orchestrator(agents=agents, aggregator=_agg(tmp_path))
    v = orch.run("AAPL", as_of=date(2026, 9, 15))
    assert [o.agent for o in v.opinions] == ["fundamentals"]
    assert v.skipped_agents == ["technical"]


def test_on_event_called_per_agent(tmp_path):
    agents = [StubAgent("fundamentals", 0.5), StubAgent("technical", 0.0, fail=True)]
    orch = Orchestrator(agents=agents, aggregator=_agg(tmp_path))
    events = []
    orch.run("AAPL", as_of=date(2026, 9, 15), on_event=events.append)
    assert events[0]["agent"] == "fundamentals" and "opinion" in events[0]
    assert events[1]["agent"] == "technical" and events[1]["skipped"] is True
    assert "data unavailable" in events[1]["reason"]


def test_orchestrator_surfaces_agent_metrics_and_drops_nan():
    import math
    from datetime import date

    from equity_research.agents.base import AgentOpinion
    from equity_research.data.models import Evidence
    from equity_research.orchestration.aggregator import Aggregator, Verdict
    from equity_research.orchestration.orchestrator import Orchestrator

    class StubAgent:
        name = "fundamentals"
        def gather(self, ticker, as_of):
            return Evidence(ticker=ticker, as_of=as_of,
                            metrics={"pe": 20.0, "revenue_growth": math.nan})
        def judge(self, evidence):
            return AgentOpinion(agent="fundamentals", stance="bullish", score=0.5,
                                confidence=0.8, rationale="r")

    captured = {}
    def on_event(ev):
        if "opinion" in ev:
            captured[ev["agent"]] = ev["opinion"]

    class StubAgg:
        def aggregate(self, ticker, as_of, opinions, skipped, skip_reasons):
            return Verdict(ticker=ticker, as_of=as_of, verdict="buy", score=0.5,
                           confidence=0.8, narrative="n", opinions=opinions)

    orch = Orchestrator([StubAgent()], StubAgg())
    verdict = orch.run("AAPL", date(2026, 9, 15), on_event=on_event)
    op = captured["fundamentals"]
    assert op.metrics == {"pe": 20.0}
    assert verdict.opinions[0].metrics == {"pe": 20.0}


def test_orchestrator_applies_critic():
    from datetime import date

    from equity_research.agents.base import AgentOpinion
    from equity_research.data.models import Evidence
    from equity_research.orchestration.aggregator import Verdict
    from equity_research.orchestration.orchestrator import Orchestrator

    class StubAgent:
        name = "fundamentals"
        def gather(self, ticker, as_of):
            return Evidence(ticker=ticker, as_of=as_of, metrics={"pe": 30.0})
        def judge(self, evidence):
            return AgentOpinion(agent="fundamentals", stance="bullish", score=0.5,
                                confidence=0.9, rationale="r")

    def critic(agent, evidence, opinion):
        return opinion.model_copy(update={"confidence": 0.3, "critique": "weak"})

    captured = {}
    def on_event(ev):
        if "opinion" in ev:
            captured[ev["agent"]] = ev["opinion"]

    class StubAgg:
        def aggregate(self, ticker, as_of, opinions, skipped, skip_reasons):
            return Verdict(ticker=ticker, as_of=as_of, verdict="buy", score=0.5,
                           confidence=0.3, narrative="n", opinions=opinions)

    orch = Orchestrator([StubAgent()], StubAgg(), critic=critic)
    orch.run("AAPL", date(2026, 9, 15), on_event=on_event)
    op = captured["fundamentals"]
    assert op.confidence == 0.3 and op.critique == "weak"
