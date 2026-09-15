from datetime import date

from equity_research.rag.store import NewsStore


class FakeVectorStore:
    def __init__(self):
        self.docs = {}  # id -> (text, meta)
        self.last_query = None

    def existing_ids(self, ids):
        return {i for i in ids if i in self.docs}

    def add(self, ids, texts, metadatas):
        for i, t, m in zip(ids, texts, metadatas):
            self.docs[i] = (t, m)

    def mmr_search(self, query, where, k):
        self.last_query = (query, where, k)
        return [(t, m) for (t, m) in self.docs.values()][:k]


def _chunk(hash_, ticker="AAPL", date_int=20260910):
    return ("Apple beat estimates.", {"doc_type": "news", "ticker": ticker,
            "date_int": date_int, "content_hash": hash_})


def test_upsert_dedups_by_content_hash():
    fake = FakeVectorStore()
    store = NewsStore(fake)
    store.upsert([_chunk("h1"), _chunk("h1"), _chunk("h2")])  # h1 twice
    assert len(fake.docs) == 2  # only h1 and h2


def test_search_builds_ticker_and_date_filter():
    fake = FakeVectorStore()
    store = NewsStore(fake)
    store.upsert([_chunk("h1")])
    store.search("news about apple", ticker="AAPL", as_of=date(2026, 9, 15), k=6)
    query, where, k = fake.last_query
    assert where == {"$and": [{"ticker": {"$eq": "AAPL"}}, {"date_int": {"$lte": 20260915}}]}
    assert k == 6
