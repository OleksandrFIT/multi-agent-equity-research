from __future__ import annotations

from datetime import date

from equity_research.agents.base import AgentOpinion
from equity_research.agents.judge import judge_evidence
from equity_research.analytics.indicators import compute_indicators
from equity_research.data.models import Evidence
from equity_research.data.prices import PriceProvider
from equity_research.llm.ollama_client import OllamaClient


class TechnicalAgent:
    name = "technical"

    def __init__(self, prices: PriceProvider, client: OllamaClient):
        self.prices = prices
        self.client = client

    def gather(self, ticker: str, as_of: date) -> Evidence:
        hist, note = self.prices.history(ticker, as_of)
        metrics = compute_indicators(hist["Close"])
        notes = [note] if note else []
        return Evidence(ticker=ticker, as_of=as_of, metrics=metrics, notes=notes)

    def judge(self, evidence: Evidence) -> AgentOpinion:
        return judge_evidence(self.name, evidence, self.client)
