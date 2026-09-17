from __future__ import annotations

import logging
import math
from datetime import date

from equity_research.agents.base import Agent, AgentOpinion
from equity_research.orchestration.aggregator import Aggregator, Verdict

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, agents: list[Agent], aggregator: Aggregator, critic=None):
        self.agents = agents
        self.aggregator = aggregator
        self.critic = critic

    def run(self, ticker: str, as_of: date, on_event=None) -> Verdict:
        opinions: list[AgentOpinion] = []
        skipped: list[str] = []
        skip_reasons: dict[str, str] = {}
        for agent in self.agents:
            try:
                evidence = agent.gather(ticker, as_of)
                opinion = agent.judge(evidence)
                if self.critic is not None:
                    opinion = self.critic(agent.name, evidence, opinion)
                opinion.metrics = {
                    k: float(v) for k, v in evidence.metrics.items()
                    if isinstance(v, (int, float)) and math.isfinite(v)
                }
                opinions.append(opinion)
                if on_event is not None:
                    on_event({"agent": agent.name, "opinion": opinion})
            except Exception as exc:
                logger.exception("agent %s failed during run", agent.name)
                skipped.append(agent.name)
                reason = f"{type(exc).__name__}: {exc}"
                skip_reasons[agent.name] = reason
                if on_event is not None:
                    on_event({"agent": agent.name, "skipped": True, "reason": reason})
        return self.aggregator.aggregate(ticker, as_of, opinions, skipped, skip_reasons)
