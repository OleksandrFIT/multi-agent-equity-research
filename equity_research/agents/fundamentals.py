from __future__ import annotations

from datetime import date

from equity_research.agents.base import AgentOpinion
from equity_research.agents.judge import judge_evidence
from equity_research.analytics.fundamentals import compute_fundamental_metrics
from equity_research.data.edgar import FactsSource
from equity_research.data.models import Evidence
from equity_research.data.prices import PriceProvider
from equity_research.llm.ollama_client import OllamaClient

_FILING_QUERY = "risk factors, financial health, business outlook and liquidity"


class FundamentalsAgent:
    name = "fundamentals"

    def __init__(self, facts_source, prices, client,
                 filing_retriever=None, filing_ingest_fn=None, k: int = 3, candidate_k: int = 12):
        self.facts_source = facts_source
        self.prices = prices
        self.client = client
        self.filing_retriever = filing_retriever
        self.filing_ingest_fn = filing_ingest_fn
        self.k = k
        self.candidate_k = candidate_k

    def gather(self, ticker: str, as_of: date) -> Evidence:
        hist, note = self.prices.history(ticker, as_of)
        price = float(hist["Close"].iloc[-1])
        facts = self.facts_source.company_facts(ticker, as_of)
        metrics = compute_fundamental_metrics(facts, price=price)
        context: list[str] = []
        if self.filing_ingest_fn is not None:
            self.filing_ingest_fn(ticker)
        if self.filing_retriever is not None:
            context = self.filing_retriever.retrieve(ticker, _FILING_QUERY, as_of, self.k, self.candidate_k)
        notes = [note] if note else []
        return Evidence(ticker=ticker, as_of=as_of, metrics=metrics, context=context, notes=notes)

    def judge(self, evidence: Evidence) -> AgentOpinion:
        return judge_evidence(self.name, evidence, self.client)
