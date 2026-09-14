from __future__ import annotations

from datetime import date

from equity_research.agents.base import Agent, AgentOpinion
from equity_research.orchestration.aggregator import Aggregator, Verdict


class Orchestrator:
    def __init__(self, agents: list[Agent], aggregator: Aggregator):
        self.agents = agents
        self.aggregator = aggregator

    def run(self, ticker: str, as_of: date) -> Verdict:
        opinions: list[AgentOpinion] = []
        skipped: list[str] = []
        for agent in self.agents:
            try:
                evidence = agent.gather(ticker, as_of)
                opinions.append(agent.judge(evidence))
            except Exception:
                skipped.append(agent.name)
        return self.aggregator.aggregate(ticker, as_of, opinions, skipped)
