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

DIRECTIONAL_AGENTS = {"fundamentals", "technical", "sentiment"}
PRICE_DIRECTIONAL = {"fundamentals", "technical"}


class Verdict(BaseModel):
    ticker: str
    as_of: date
    verdict: Literal["buy", "hold", "sell"]
    status: Literal["ok", "unknown_ticker", "insufficient_data"] = "ok"
    score: float
    confidence: float
    narrative: str
    opinions: list[AgentOpinion]
    disclaimer: str = DISCLAIMER
    caution: str | None = None
    skipped_agents: list[str] = []
    skip_reasons: dict[str, str] = {}


class Aggregator:
    def __init__(self, config: Config, client: OllamaClient, buy_th: float = 0.2, sell_th: float = -0.2):
        self.config = config
        self.client = client
        self.buy_th = buy_th
        self.sell_th = sell_th

    def aggregate(self, ticker: str, as_of: date, opinions: list[AgentOpinion], skipped: list[str],
                  skip_reasons: dict[str, str] | None = None) -> Verdict:
        skipped = list(skipped)
        skip_reasons = dict(skip_reasons or {})
        risk_op = next((o for o in opinions if o.agent == "risk"), None)
        directional = [o for o in opinions if o.agent in DIRECTIONAL_AGENTS and o.agent in self.config.weights]
        directional_names = {o.agent for o in directional}
        skipped += [o.agent for o in opinions if o.agent != "risk" and o.agent not in directional_names]

        price_directional = [o for o in directional if o.agent in PRICE_DIRECTIONAL]
        if not price_directional:
            return Verdict(
                ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                confidence=0.0, status="insufficient_data",
                narrative="Insufficient data: no price-based analysis available for this ticker.",
                opinions=opinions, skipped_agents=skipped, skip_reasons=skip_reasons,
            )

        weights = self.config.normalized_weights([o.agent for o in directional])
        score = sum(weights[o.agent] * o.score for o in directional)
        base_conf = sum(weights[o.agent] * o.confidence for o in directional)
        verdict = "buy" if score >= self.buy_th else "sell" if score <= self.sell_th else "hold"

        confidence = base_conf
        caution = None
        if risk_op is not None:
            rl = risk_op.score
            confidence = max(0.0, min(1.0, base_conf * (1 - self.config.risk["gate_strength"] * rl)))
            if rl >= self.config.risk["caution_threshold"]:
                caution = f"Elevated risk (level {rl:.0%}): {risk_op.rationale}"

        narrative = self.client.generate_text(self._narrative_prompt(ticker, verdict, score, directional))
        opinions_out = directional + ([risk_op] if risk_op is not None else [])
        return Verdict(
            ticker=ticker, as_of=as_of, verdict=verdict, score=score,
            confidence=confidence, narrative=narrative, opinions=opinions_out,
            caution=caution, skipped_agents=skipped, skip_reasons=skip_reasons,
        )

    def _narrative_prompt(self, ticker, verdict, score, opinions) -> str:
        lines = "\n".join(f"- {o.agent}: {o.stance} (score {o.score}) — {o.rationale}" for o in opinions)
        return (
            f"Summarize the investment view on {ticker}. Overall verdict is {verdict} "
            f"(score {score:.2f}). Explain in 3-4 sentences and explicitly note any "
            f"disagreement between analysts. Do not use guarantees or price targets.\n"
            f"Analyst opinions:\n{lines}"
        )
