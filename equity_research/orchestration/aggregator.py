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
    calibration_note: str | None = None
    skipped_agents: list[str] = []
    skip_reasons: dict[str, str] = {}


class Aggregator:
    def __init__(self, config: Config, client: OllamaClient, buy_th: float = 0.2, sell_th: float = -0.2, calibration: dict | None = None):
        self.config = config
        self.client = client
        self.buy_th = buy_th
        self.sell_th = sell_th
        self.calibration = calibration

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
        mech_score = sum(weights[o.agent] * o.score for o in directional)
        base_conf = sum(weights[o.agent] * o.confidence for o in directional)

        score, base_confidence, narrative = self._decide(ticker, mech_score, base_conf, directional)
        verdict = "buy" if score >= self.buy_th else "sell" if score <= self.sell_th else "hold"

        confidence = base_confidence
        caution = None
        if risk_op is not None:
            rl = risk_op.score
            confidence = max(0.0, min(1.0, base_confidence * (1 - self.config.risk["gate_strength"] * rl)))
            if rl >= self.config.risk["caution_threshold"]:
                caution = f"Elevated risk (level {rl:.0%}): {risk_op.rationale}"

        calibration_note = None
        if self.calibration is not None:
            from equity_research.eval.calibration import calibration_factor
            factor = calibration_factor(self.calibration, verdict, self.config.calibration_horizon)
            if factor < 1.0:
                confidence = confidence * factor
                calibration_note = f"Confidence tempered ×{factor:.2f} by historical {verdict} accuracy"

        opinions_out = directional + ([risk_op] if risk_op is not None else [])
        return Verdict(
            ticker=ticker, as_of=as_of, verdict=verdict, score=score,
            confidence=confidence, narrative=narrative, opinions=opinions_out,
            caution=caution, calibration_note=calibration_note, skipped_agents=skipped, skip_reasons=skip_reasons,
        )

    def _decide(self, ticker, mech_score, base_conf, directional):
        """(score, confidence, narrative): LLM-PM blend when enabled, else mechanical.

        The mechanical score is the deterministic anchor; the PM only shifts it by
        (1 - pm_weight). Any PM failure falls back to the mechanical path.
        """
        if getattr(self.config, "pm_enabled", False):
            from equity_research.orchestration.portfolio_manager import run_pm
            try:
                pm = run_pm(self.client, ticker, mech_score, directional)
                w = self.config.pm_weight
                return (w * mech_score + (1 - w) * pm["score"], pm["confidence"], pm["narrative"])
            except Exception:
                pass  # PM unavailable/invalid -> mechanical fallback
        mech_verdict = "buy" if mech_score >= self.buy_th else "sell" if mech_score <= self.sell_th else "hold"
        narrative = self.client.generate_text(self._narrative_prompt(ticker, mech_verdict, mech_score, directional))
        return mech_score, base_conf, narrative

    def _narrative_prompt(self, ticker, verdict, score, opinions) -> str:
        lines = "\n".join(f"- {o.agent}: {o.stance} (score {o.score}) — {o.rationale}" for o in opinions)
        return (
            f"Summarize the investment view on {ticker}. Overall verdict is {verdict} "
            f"(score {score:.2f}). Explain in 3-4 sentences and explicitly note any "
            f"disagreement between analysts. Do not use guarantees or price targets.\n"
            f"Analyst opinions:\n{lines}"
        )
