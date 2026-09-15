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
    assert where == {"$and": [
        {"doc_type": {"$eq": "news"}},
        {"ticker": {"$eq": "AAPL"}},
        {"date_int": {"$lte": 20260915}},
    ]}
    assert k == 6


def test_chroma_store_retries_embed_ops():
    from equity_research.rag.chroma_store import ChromaVectorStore

    class FlakyDB:
        def __init__(self):
            self.calls = 0

        def max_marginal_relevance_search(self, query, k, filter):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection reset by peer")

            class Doc:
                page_content = "ok"
                metadata = {"id": "1"}

            return [Doc()]

    store = ChromaVectorStore.__new__(ChromaVectorStore)  # bypass real Chroma init
    store._db = FlakyDB()
    store._net = {"data_timeout": 5, "data_attempts": 3, "data_base_delay": 0}
    out = store.mmr_search("q", {"doc_type": {"$eq": "news"}}, k=1)
    assert out == [("ok", {"id": "1"})]
    assert store._db.calls == 2  # retried once


def test_chroma_store_no_net_calls_once():
    from equity_research.rag.chroma_store import ChromaVectorStore

    class DB:
        def __init__(self):
            self.calls = 0

        def get(self, ids):
            self.calls += 1
            return {"ids": ["a"]}

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._db = DB()
    store._net = None
    assert store.existing_ids(["a"]) == {"a"}
    assert store._db.calls == 1
