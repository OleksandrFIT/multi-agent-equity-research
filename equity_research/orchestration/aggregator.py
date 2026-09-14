from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel

from equity_research.agents.base import AgentOpinion
from equity_research.config import Config
from equity_research.llm.ollama_client import OllamaClient

DISCLAIMER = (
    "This is an automated research aid, not investment advice. Figures may be "
    "incomplete or wrong. Do your own due diligence."
)


class Verdict(BaseModel):
    ticker: str
    as_of: date
    verdict: Literal["buy", "hold", "sell"]
    score: float
    confidence: float
    narrative: str
    opinions: list[AgentOpinion]
    disclaimer: str = DISCLAIMER
    skipped_agents: list[str] = []


class Aggregator:
    def __init__(self, config: Config, client: OllamaClient, buy_th: float = 0.2, sell_th: float = -0.2):
        self.config = config
        self.client = client
        self.buy_th = buy_th
        self.sell_th = sell_th

    def aggregate(self, ticker: str, as_of: date, opinions: list[AgentOpinion], skipped: list[str]) -> Verdict:
        skipped = list(skipped)
        weighted = [o for o in opinions if o.agent in self.config.weights]
        skipped += [o.agent for o in opinions if o.agent not in self.config.weights]
        if not weighted:
            return Verdict(
                ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                confidence=0.0, narrative="No agent produced an opinion; no data available.",
                opinions=[], skipped_agents=skipped,
            )
        available = [o.agent for o in weighted]
        weights = self.config.normalized_weights(available)
        score = sum(weights[o.agent] * o.score for o in weighted)
        confidence = sum(weights[o.agent] * o.confidence for o in weighted)
        verdict = "buy" if score >= self.buy_th else "sell" if score <= self.sell_th else "hold"
        narrative = self.client.generate_text(self._narrative_prompt(ticker, verdict, score, weighted))
        return Verdict(
            ticker=ticker, as_of=as_of, verdict=verdict, score=score,
            confidence=confidence, narrative=narrative, opinions=weighted,
            skipped_agents=skipped,
        )

    def _narrative_prompt(self, ticker, verdict, score, opinions) -> str:
        lines = "\n".join(f"- {o.agent}: {o.stance} (score {o.score}) — {o.rationale}" for o in opinions)
        return (
            f"Summarize the investment view on {ticker}. Overall verdict is {verdict} "
            f"(score {score:.2f}). Explain in 3-4 sentences and explicitly note any "
            f"disagreement between analysts. Do not use guarantees or price targets.\n"
            f"Analyst opinions:\n{lines}"
        )
