from __future__ import annotations

import hashlib
from datetime import date

from pydantic import BaseModel


class NewsItem(BaseModel):
    ticker: str
    title: str
    text: str
    url: str
    source: str
    published_at: date


def content_hash(item: NewsItem) -> str:
    raw = f"{item.ticker}|{item.url}|{item.title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def news_metadata(item: NewsItem) -> dict:
    return {
        "doc_type": "news",
        "ticker": item.ticker,
        "date_int": int(item.published_at.strftime("%Y%m%d")),
        "content_hash": content_hash(item),
        "source": item.source,
        "url": item.url,
        "title": item.title,
    }
