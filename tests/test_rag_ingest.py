from datetime import date

from equity_research.rag.ingest import ingest_news
from equity_research.rag.records import NewsItem


class FakeNewsStore:
    def __init__(self):
        self.upserted = []

    def upsert(self, chunks):
        self.upserted.extend(chunks)


def _item(url):
    return NewsItem(ticker="AAPL", title="t", text="Body text here.",
                    url=url, source="yahoo", published_at=date(2026, 9, 10))


def test_ingest_chunks_and_upserts():
    store = FakeNewsStore()
    n = ingest_news(lambda t: [_item("u1"), _item("u2")], store, "AAPL")
    assert n == 2  # two items ingested
    assert len(store.upserted) == 2  # each short item -> one chunk


def test_ingest_empty_is_noop():
    store = FakeNewsStore()
    assert ingest_news(lambda t: [], store, "AAPL") == 0
    assert store.upserted == []
