import json
from datetime import date
from pathlib import Path

import pytest

FIX = Path(__file__).parent.parent / "fixtures" / "hitk_corpus.json"


@pytest.mark.integration
def test_hitk_frozen_corpus(tmp_path):
    """Requires Ollama (nomic-embed-text) + chromadb. Run: uv run pytest -m integration."""
    from equity_research.eval.retrieval_eval import hit_at_k
    from equity_research.rag.chroma_store import ChromaVectorStore
    from equity_research.rag.store import NewsStore

    corpus = json.loads(FIX.read_text())
    vs = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"),
                           embed_model="nomic-embed-text", collection="hitk")
    ids = [d["id"] for d in corpus["documents"]]
    texts = [d["text"] for d in corpus["documents"]]
    metas = [{"doc_type": "news", "ticker": d["ticker"], "date_int": 20240101,
              "content_hash": d["id"]} for d in corpus["documents"]]
    vs.add(ids, texts, metas)
    store = NewsStore(vs)

    def retrieve_fn(ticker, question):
        hits = store.search(question, ticker=ticker, as_of=date(2024, 6, 1), k=3)
        return [m["content_hash"] for _, m in hits]

    score = hit_at_k(retrieve_fn, corpus["cases"], k=3)
    assert score >= 0.6
