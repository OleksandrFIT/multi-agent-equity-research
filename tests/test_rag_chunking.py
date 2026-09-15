from datetime import date

from equity_research.rag.chunking import chunk_news
from equity_research.rag.records import NewsItem


def _item(text):
    return NewsItem(ticker="AAPL", title="Apple beats", text=text,
                    url="http://x/1", source="yahoo", published_at=date(2026, 9, 10))


def test_short_news_is_single_chunk_with_title_prefix():
    chunks = chunk_news(_item("Short body."))
    assert len(chunks) == 1
    text, meta = chunks[0]
    assert text.startswith("Apple beats")  # title prefixed for embedding context
    assert "Short body." in text
    assert meta["ticker"] == "AAPL"


def test_long_news_splits_into_multiple_chunks():
    long_text = " ".join(f"sentence{i}." for i in range(400))  # well over ~1000 tokens
    chunks = chunk_news(_item(long_text))
    assert len(chunks) > 1
    # every chunk carries the same metadata content_hash
    assert len({m["content_hash"] for _, m in chunks}) == 1
