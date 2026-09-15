from __future__ import annotations

from datetime import date
from typing import Callable

from equity_research.agents.base import AgentOpinion
from equity_research.agents.judge import judge_evidence
from equity_research.data.models import Evidence
from equity_research.llm.ollama_client import OllamaClient

_QUERY = "recent news sentiment, outlook, risks and catalysts"


class SentimentAgent:
    name = "sentiment"

    def __init__(self, retriever, ingest_fn: Callable[[str], object], client: OllamaClient,
                 k: int = 6, candidate_k: int = 20):
        self.retriever = retriever
        self.ingest_fn = ingest_fn
        self.client = client
        self.k = k
        self.candidate_k = candidate_k

    def gather(self, ticker: str, as_of: date) -> Evidence:
        self.ingest_fn(ticker)  # pull fresh news into the store first (live)
        texts = self.retriever.retrieve(ticker, _QUERY, as_of, self.k, self.candidate_k)
        notes = [] if texts else ["no news retrieved"]
        return Evidence(ticker=ticker, as_of=as_of, metrics={}, context=texts, notes=notes)

    def judge(self, evidence: Evidence) -> AgentOpinion:
        if not evidence.context:
            return AgentOpinion(agent=self.name, stance="neutral", score=0.0, confidence=0.0,
                                rationale="no news available", key_facts=[])
        return judge_evidence(self.name, evidence, self.client)
