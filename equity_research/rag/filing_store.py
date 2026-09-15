from __future__ import annotations

from datetime import date

from equity_research.rag.filing_records import chunk_filing


class FilingStore:
    def __init__(self, vector_store, parent_store):
        self.vs = vector_store
        self.ps = parent_store

    def upsert(self, sections) -> None:
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict] = []
        for section in sections:
            pid, parent_text, children = chunk_filing(section)
            self.ps.put(pid, parent_text)
            for i, (ctext, cmeta) in enumerate(children):
                ids.append(f"{pid}:{i}")
                texts.append(ctext)
                metas.append(cmeta)
        if not ids:
            return
        existing = self.vs.existing_ids(ids)
        keep = [(i, t, m) for i, t, m in zip(ids, texts, metas) if i not in existing]
        if keep:
            self.vs.add([i for i, _, _ in keep], [t for _, t, _ in keep], [m for _, _, m in keep])

    def search(self, query: str, ticker: str, as_of: date, candidate_k: int) -> list[tuple[str, dict]]:
        where = {"$and": [
            {"doc_type": {"$eq": "filing_section"}},
            {"ticker": {"$eq": ticker}},
            {"date_int": {"$lte": int(as_of.strftime("%Y%m%d"))}},
        ]}
        hits = self.vs.mmr_search(query, where, candidate_k)
        out: list[tuple[str, dict]] = []
        seen: set[str] = set()
        for _text, meta in hits:
            pid = meta.get("parent_id")
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            out.append((self.ps.get(pid), meta))
        return out
