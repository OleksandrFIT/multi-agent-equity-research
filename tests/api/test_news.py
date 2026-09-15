from datetime import date

import api.core as core
from equity_research.rag.records import NewsItem


def test_news_resolves_fetches_and_reports(monkeypatch):
    monkeypatch.setattr(core, "resolve", lambda q: {"input": q, "resolved": "AAPL", "corrected": True})
    items = [NewsItem(ticker="AAPL", title="Apple news", text="body", url="u1",
                      source="s", published_at=date(2026, 9, 10))]
    monkeypatch.setattr("equity_research.rag.adapters.fetch_news", lambda t: items)

    class FakeVS:
        def existing_ids(self, ids):
            return set()

        def add(self, *a):
            pass

        def mmr_search(self, *a):
            return []

    monkeypatch.setattr("equity_research.rag.chroma_store.ChromaVectorStore", lambda **kw: FakeVS())

    out = core.news("APPL")
    assert out["resolved"] == "AAPL" and out["corrected"] is True
    assert out["fetched"] == 1 and out["added"] == 1
    assert out["items"][0]["title"] == "Apple news"
    assert out["items"][0]["url"] == "u1"


def test_news_not_found(monkeypatch):
    monkeypatch.setattr(core, "resolve", lambda q: {"input": q, "resolved": None, "corrected": False})
    out = core.news("ZZZZ")
    assert out == {"query": "ZZZZ", "resolved": None, "corrected": False,
                   "items": [], "fetched": 0, "added": 0}


from fastapi.testclient import TestClient

import api.main as main


def test_news_endpoint(monkeypatch):
    monkeypatch.setattr(core, "news", lambda q: {"query": q, "resolved": "AAPL", "corrected": False,
                                                 "items": [], "fetched": 0, "added": 0})
    client = TestClient(main.app)
    resp = client.post("/api/news", json={"query": "Apple"})
    assert resp.status_code == 200
    assert resp.json()["resolved"] == "AAPL"
