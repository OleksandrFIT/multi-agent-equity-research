from __future__ import annotations

from datetime import date

from equity_research.agents.base import AgentOpinion
from equity_research.agents.judge import judge_evidence
from equity_research.analytics.fundamentals import compute_fundamental_metrics
from equity_research.data.edgar import FactsSource
from equity_research.data.models import Evidence
from equity_research.data.prices import PriceProvider
from equity_research.llm.ollama_client import OllamaClient


class FundamentalsAgent:
    name = "fundamentals"

    def __init__(self, facts_source: FactsSource, prices: PriceProvider, client: OllamaClient):
        self.facts_source = facts_source
        self.prices = prices
        self.client = client

    def gather(self, ticker: str, as_of: date) -> Evidence:
        hist, note = self.prices.history(ticker, as_of)
        price = float(hist["Close"].iloc[-1])
        facts = self.facts_source.company_facts(ticker)
        metrics = compute_fundamental_metrics(facts, price=price)
        notes = [note] if note else []
        return Evidence(ticker=ticker, as_of=as_of, metrics=metrics, notes=notes)

    def judge(self, evidence: Evidence) -> AgentOpinion:
        return judge_evidence(self.name, evidence, self.client)
