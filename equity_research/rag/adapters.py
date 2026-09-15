from __future__ import annotations

from datetime import date, datetime

from equity_research.rag.records import NewsItem


def fetch_news(ticker: str) -> list[NewsItem]:
    """Fetch recent news for a ticker via yfinance. Thin; verified live in a later task."""
    import yfinance as yf

    raw = getattr(yf.Ticker(ticker), "news", []) or []
    items: list[NewsItem] = []
    for entry in raw:
        content = entry.get("content", entry)  # yfinance schema varies by version
        title = content.get("title") or ""
        url = (content.get("canonicalUrl", {}) or {}).get("url") or content.get("link") or ""
        summary = content.get("summary") or content.get("description") or title
        pub = content.get("pubDate") or content.get("providerPublishTime")
        try:
            published = (datetime.fromisoformat(pub.replace("Z", "+00:00")).date()
                         if isinstance(pub, str) else
                         datetime.utcfromtimestamp(int(pub)).date() if pub else date.today())
        except Exception:
            published = date.today()
        if not title or not url:
            continue
        items.append(NewsItem(ticker=ticker, title=title, text=summary, url=url,
                              source="yfinance", published_at=published))
    return items
