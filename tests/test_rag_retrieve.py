from datetime import date

from equity_research.rag.retrieve import NewsRetriever


class FakeNewsStore:
    def __init__(self, results):
        self._results = results
        self.searched = None

    def search(self, query, ticker, as_of, k):
        self.searched = (query, ticker, as_of, k)
        return self._results


def test_retrieve_reranks_and_trims_to_top_k():
    candidates = [("a", {}), ("b", {}), ("c", {})]
    store = FakeNewsStore(candidates)
    scores = {"a": 0.1, "b": 0.9, "c": 0.5}
    retriever = NewsRetriever(store, scorer=lambda q, ts: [scores[t] for t in ts])
    texts = retriever.retrieve("AAPL", "apple news", as_of=date(2026, 9, 15), k=2, candidate_k=3)
    assert texts == ["b", "c"]  # reranked desc, trimmed to k=2
    assert store.searched[3] == 3  # searched with candidate_k


def test_retrieve_without_scorer_keeps_store_order():
    candidates = [("x", {}), ("y", {})]
    retriever = NewsRetriever(FakeNewsStore(candidates), scorer=None)
    texts = retriever.retrieve("AAPL", "q", as_of=date(2026, 9, 15), k=5, candidate_k=10)
    assert texts == ["x", "y"]
