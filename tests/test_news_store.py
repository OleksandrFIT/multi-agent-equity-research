from datetime import date

from equity_research.rag.chunking import chunk_news
from equity_research.rag.records import NewsItem
from equity_research.rag.store import NewsStore


class FakeVS:
    def __init__(self):
        self.docs = {}

    def existing_ids(self, ids):
        return {i for i in ids if i in self.docs}

    def add(self, ids, texts, metas):
        for i, t, m in zip(ids, texts, metas):
            self.docs[i] = (t, m)

    def mmr_search(self, query, where, k):
        return []


def _item(url):
    return NewsItem(ticker="AAPL", title="t", text="Body text here.", url=url,
                    source="s", published_at=date(2026, 9, 10))


def test_upsert_returns_new_chunk_count():
    store = NewsStore(FakeVS())
    chunks = chunk_news(_item("u1")) + chunk_news(_item("u2"))
    assert store.upsert(chunks) == 2   # both new
    assert store.upsert(chunks) == 0   # already stored


def test_upsert_empty_returns_zero():
    assert NewsStore(FakeVS()).upsert([]) == 0
