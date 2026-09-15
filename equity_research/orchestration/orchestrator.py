from __future__ import annotations

import logging
from datetime import date

from equity_research.agents.base import Agent, AgentOpinion
from equity_research.orchestration.aggregator import Aggregator, Verdict

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, agents: list[Agent], aggregator: Aggregator):
        self.agents = agents
        self.aggregator = aggregator

    def run(self, ticker: str, as_of: date) -> Verdict:
        opinions: list[AgentOpinion] = []
        skipped: list[str] = []
        skip_reasons: dict[str, str] = {}
        for agent in self.agents:
            try:
                evidence = agent.gather(ticker, as_of)
                opinions.append(agent.judge(evidence))
            except Exception as exc:
                logger.exception("agent %s failed during run", agent.name)
                skipped.append(agent.name)
                skip_reasons[agent.name] = f"{type(exc).__name__}: {exc}"
        return self.aggregator.aggregate(ticker, as_of, opinions, skipped, skip_reasons)
