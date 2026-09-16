from __future__ import annotations

import json

from equity_research.agents.base import AgentOpinion
from equity_research.llm.ollama_client import OllamaClient

PM_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "narrative": {"type": "string"},
    },
    "required": ["reasoning", "score", "confidence", "narrative"],
}


def build_pm_prompt(ticker: str, mech_score: float, opinions: list[AgentOpinion]) -> str:
    lines = "\n".join(
        f"- {o.agent}: {o.stance} (score {o.score:+.2f}, confidence {o.confidence:.0%}) — {o.rationale}"
        + (f" [facts: {', '.join(o.key_facts)}]" if o.key_facts else "")
        for o in opinions
    )
    return (
        f"You are the portfolio manager deciding on {ticker}. Your analysts' opinions:\n"
        f"{lines}\n"
        f"A mechanical weighted blend of their scores is {mech_score:+.2f} (a reference, not a rule).\n"
        "First reason step by step in `reasoning`: weigh agreement versus disagreement, the strength "
        "of the evidence, and risk. Then output a final score in [-1,1] (sign = buy/sell direction, "
        "magnitude = conviction), a confidence in [0,1], and a 3-4 sentence narrative explaining the "
        "decision and noting any disagreement. Do not use price targets or guarantees.\n"
        f"JSON schema: {json.dumps(PM_SCHEMA)}"
    )


def run_pm(client: OllamaClient, ticker: str, mech_score: float,
           opinions: list[AgentOpinion]) -> dict:
    return client.generate_json(build_pm_prompt(ticker, mech_score, opinions), PM_SCHEMA)
