from datetime import date

from equity_research.rag.filing_records import FilingSection
from equity_research.rag.filing_store import FilingStore
from equity_research.rag.parent_store import ParentStore


class FakeVectorStore:
    def __init__(self):
        self.docs = {}
        self.last_where = None

    def existing_ids(self, ids):
        return {i for i in ids if i in self.docs}

    def add(self, ids, texts, metadatas):
        for i, t, m in zip(ids, texts, metadatas):
            self.docs[i] = (t, m)

    def mmr_search(self, query, where, k):
        self.last_where = where
        return [(t, m) for (t, m) in self.docs.values()][:k]


def _section(text, section="risk_factors"):
    return FilingSection(ticker="AAPL", section=section, filed_at=date(2023, 11, 3), form="10-K", text=text)


def test_upsert_stores_parent_and_children(tmp_path):
    vs = FakeVectorStore()
    ps = ParentStore(tmp_path)
    store = FilingStore(vs, ps)
    store.upsert([_section("A risk factors section with some length.")])
    assert len(vs.docs) >= 1
    pid = next(iter(vs.docs.values()))[1]["parent_id"]
    assert ps.get(pid).startswith("A risk factors")


def test_search_filters_doc_type_and_returns_parent_texts(tmp_path):
    vs = FakeVectorStore()
    ps = ParentStore(tmp_path)
    store = FilingStore(vs, ps)
    store.upsert([_section("Risk factors body text here.")])
    hits = store.search("risks", ticker="AAPL", as_of=date(2024, 6, 1), candidate_k=8)
    assert vs.last_where == {"$and": [
        {"doc_type": {"$eq": "filing_section"}},
        {"ticker": {"$eq": "AAPL"}},
        {"date_int": {"$lte": 20240601}},
    ]}
    assert len(hits) == 1
    assert hits[0][0] == "Risk factors body text here."
