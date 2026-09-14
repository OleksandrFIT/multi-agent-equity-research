from __future__ import annotations

from equity_research.agents.base import AgentOpinion
from equity_research.agents.prompts import OPINION_SCHEMA, build_judge_prompt
from equity_research.data.models import Evidence
from equity_research.llm.ollama_client import OllamaClient


def judge_evidence(agent: str, evidence: Evidence, client: OllamaClient) -> AgentOpinion:
    prompt = build_judge_prompt(agent, evidence)
    try:
        raw = client.generate_json(prompt, OPINION_SCHEMA)
        return AgentOpinion(agent=agent, **raw)
    except Exception:  # invalid JSON after retries, or schema validation failure
        return AgentOpinion(
            agent=agent,
            stance="neutral",
            score=0.0,
            confidence=0.0,
            rationale="LLM output invalid; degraded to neutral.",
            key_facts=[],
        )
