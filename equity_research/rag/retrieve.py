from __future__ import annotations

from datetime import date

from equity_research.rag.reranker import Scorer, rerank


class NewsRetriever:
    def __init__(self, store, scorer: Scorer | None):
        self.store = store
        self.scorer = scorer

    def retrieve(self, ticker: str, query: str, as_of: date, k: int, candidate_k: int) -> list[str]:
        candidates = self.store.search(query, ticker, as_of, candidate_k)
        reranked = rerank(query, candidates, self.scorer)
        return [text for text, _ in reranked[:k]]
