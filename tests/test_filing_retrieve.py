from datetime import date

from equity_research.rag.filing_retrieve import FilingRetriever


class FakeFilingStore:
    def __init__(self, results):
        self._results = results
        self.searched = None

    def search(self, query, ticker, as_of, candidate_k):
        self.searched = (query, ticker, as_of, candidate_k)
        return self._results


def test_retrieve_reranks_and_trims():
    cands = [("risk A", {}), ("risk B", {}), ("risk C", {})]
    store = FakeFilingStore(cands)
    scores = {"risk A": 0.1, "risk B": 0.9, "risk C": 0.5}
    r = FilingRetriever(store, scorer=lambda q, ts: [scores[t] for t in ts])
    out = r.retrieve("AAPL", "risks", as_of=date(2024, 6, 1), k=2, candidate_k=3)
    assert out == ["risk B", "risk C"]
    assert store.searched[3] == 3


def test_retrieve_without_scorer_keeps_order():
    store = FakeFilingStore([("a", {}), ("b", {})])
    r = FilingRetriever(store, scorer=None)
    assert r.retrieve("AAPL", "q", as_of=date(2024, 6, 1), k=5, candidate_k=8) == ["a", "b"]
