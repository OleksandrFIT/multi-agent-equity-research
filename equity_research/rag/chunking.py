from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from equity_research.rag.records import NewsItem, news_metadata

# ~1000 tokens ≈ ~4000 chars; below this a news item stays a single chunk.
_MAX_SINGLE_CHARS = 4000
_splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=480)


def chunk_news(item: NewsItem) -> list[tuple[str, dict]]:
    meta = news_metadata(item)
    body = f"{item.title}\n{item.text}"
    if len(body) <= _MAX_SINGLE_CHARS:
        return [(body, meta)]
    return [(chunk, meta) for chunk in _splitter.split_text(body)]
