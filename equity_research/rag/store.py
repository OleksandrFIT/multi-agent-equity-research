from __future__ import annotations

from datetime import date
from typing import Protocol


class VectorStore(Protocol):
    def existing_ids(self, ids: list[str]) -> set[str]: ...
    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None: ...
    def mmr_search(self, query: str, where: dict, k: int) -> list[tuple[str, dict]]: ...


class NewsStore:
    def __init__(self, store: VectorStore):
        self.store = store

    def upsert(self, chunks: list[tuple[str, dict]]) -> None:
        ids, texts, metas = [], [], []
        # Dedup exact (content_hash, text) duplicates within this batch first,
        # then assign ids as content_hash:<index among kept chunks of that hash>
        # so genuinely distinct chunks of one document (same hash, different
        # text) still get distinct ids.
        seen_pairs: set[tuple[str, str]] = set()
        counts: dict[str, int] = {}
        for text, meta in chunks:
            h = meta["content_hash"]
            pair = (h, text)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            idx = counts.get(h, 0)
            counts[h] = idx + 1
            cid = f"{h}:{idx}"
            ids.append(cid)
            texts.append(text)
            metas.append(meta)
        existing = self.store.existing_ids(ids)
        fresh = [i for i in ids if i not in existing]
        if not fresh:
            return
        fresh_set = set(fresh)
        keep = [(i, t, m) for i, t, m in zip(ids, texts, metas) if i in fresh_set]
        self.store.add([i for i, _, _ in keep], [t for _, t, _ in keep], [m for _, _, m in keep])

    def search(self, query: str, ticker: str, as_of: date, k: int) -> list[tuple[str, dict]]:
        where = {"$and": [
            {"doc_type": {"$eq": "news"}},
            {"ticker": {"$eq": ticker}},
            {"date_int": {"$lte": int(as_of.strftime("%Y%m%d"))}},
        ]}
        return self.store.mmr_search(query, where, k)
