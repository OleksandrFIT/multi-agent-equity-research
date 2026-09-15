from datetime import date

from equity_research.rag.records import NewsItem, content_hash, news_metadata


def _item(**kw):
    base = dict(ticker="AAPL", title="Apple beats", text="Apple beat estimates.",
                url="http://x/1", source="yahoo", published_at=date(2026, 9, 10))
    base.update(kw)
    return NewsItem(**base)


def test_content_hash_stable_and_url_sensitive():
    a = content_hash(_item())
    b = content_hash(_item())
    c = content_hash(_item(url="http://x/2"))
    assert a == b
    assert a != c


def test_metadata_has_date_int_and_ticker():
    m = news_metadata(_item())
    assert m["doc_type"] == "news"
    assert m["ticker"] == "AAPL"
    assert m["date_int"] == 20260910  # YYYYMMDD int for range filtering
    assert m["content_hash"] == content_hash(_item())
