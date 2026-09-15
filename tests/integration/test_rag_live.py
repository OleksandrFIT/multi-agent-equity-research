from datetime import date

import pytest


@pytest.mark.integration
def test_chroma_roundtrip_and_news_fetch(tmp_path):
    """Requires Ollama (nomic-embed-text pulled) + network. Run: uv run pytest -m integration."""
    from equity_research.rag.adapters import fetch_news
    from equity_research.rag.chroma_store import ChromaVectorStore
    from equity_research.rag.ingest import ingest_news
    from equity_research.rag.store import NewsStore

    vs = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"),
                           embed_model="nomic-embed-text", collection="news_test")
    store = NewsStore(vs)
    n = ingest_news(fetch_news, store, "AAPL")
    assert n >= 0  # network-dependent; ingest must not raise
    hits = store.search("apple earnings and outlook", ticker="AAPL", as_of=date.today(), k=3)
    assert isinstance(hits, list)
