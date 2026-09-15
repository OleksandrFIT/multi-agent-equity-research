from __future__ import annotations

from typing import Callable

from equity_research.rag.chunking import chunk_news
from equity_research.rag.records import NewsItem

NewsFetcher = Callable[[str], list[NewsItem]]


def ingest_news(fetch: NewsFetcher, news_store, ticker: str) -> int:
    items = fetch(ticker)
    chunks = []
    for item in items:
        chunks.extend(chunk_news(item))
    if chunks:
        news_store.upsert(chunks)
    return len(items)
