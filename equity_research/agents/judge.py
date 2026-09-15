from __future__ import annotations

from equity_research.agents.base import AgentOpinion
from equity_research.agents.grounding import ground
from equity_research.agents.prompts import OPINION_SCHEMA, build_judge_prompt
from equity_research.data.models import Evidence
from equity_research.llm.ollama_client import OllamaClient


def reconcile_stance(opinion: AgentOpinion) -> AgentOpinion:
    """Make stance agree with the score sign; only fix hard contradictions."""
    if opinion.stance == "bullish" and opinion.score <= -0.1:
        return opinion.model_copy(update={"stance": "bearish"})
    if opinion.stance == "bearish" and opinion.score >= 0.1:
        return opinion.model_copy(update={"stance": "bullish"})
    return opinion


def judge_evidence(agent: str, evidence: Evidence, client: OllamaClient) -> AgentOpinion:
    prompt = build_judge_prompt(agent, evidence)
    try:
        raw = client.generate_json(prompt, OPINION_SCHEMA)
        return ground(reconcile_stance(AgentOpinion(agent=agent, **raw)), evidence)
    except Exception:  # invalid JSON after retries, or schema validation failure
        return AgentOpinion(
            agent=agent,
            stance="neutral",
            score=0.0,
            confidence=0.0,
            rationale="LLM output invalid; degraded to neutral.",
            key_facts=[],
        )
