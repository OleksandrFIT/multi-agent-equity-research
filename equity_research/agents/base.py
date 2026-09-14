from __future__ import annotations

from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from equity_research.data.models import Evidence

Stance = Literal["bullish", "neutral", "bearish"]


class AgentOpinion(BaseModel):
    agent: str
    stance: Stance
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    rationale: str
    key_facts: list[str] = Field(default_factory=list)


class Agent(Protocol):
    name: str

    def gather(self, ticker: str, as_of: date) -> Evidence: ...

    def judge(self, evidence: Evidence) -> AgentOpinion: ...
