from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    ticker: str
    as_of: date
    metrics: dict[str, float] = Field(default_factory=dict)
    context: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
