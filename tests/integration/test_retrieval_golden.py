import json
from datetime import date
from pathlib import Path

import pytest

FIX = Path(__file__).parent.parent / "fixtures" / "golden" / "retrieval_golden.json"


@pytest.mark.integration
def test_retrieval_golden_hit_at_3():
    """Real retriever over the golden corpus. Requires Ollama embeddings.

    Run: uv run pytest -m integration tests/integration/test_retrieval_golden.py -s
    """
    import tempfile

    from equity_research.config import Config
    from equity_research.eval.retrieval_eval import hit_at_k, mrr
    from equity_research.rag.chroma_store import ChromaVectorStore
    from equity_research.rag.chunking import chunk_news
    from equity_research.rag.records import NewsItem
    from equity_research.rag.reranker import default_scorer
    from equity_research.rag.retrieve import NewsRetriever
    from equity_research.rag.store import NewsStore

    cfg = Config.load("config.yaml")
    data = json.loads(FIX.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        vs = ChromaVectorStore(persist_dir=tmp, embed_model=cfg.rag["embed_model"], net=cfg.net)
        store = NewsStore(vs)
        # Each labeled doc becomes one news item dated before the query as_of.
        # chunk_news builds body = f"{title}\n{text}", so we recover a doc id from
        # a retrieved chunk by substring containment on the doc text.
        chunks = []
        for d in data["documents"]:
            item = NewsItem(ticker=d["ticker"], title=d["id"], text=d["text"],
                            url=d["id"], source="golden", published_at=date(2024, 1, 1))
            chunks.extend(chunk_news(item))
        store.upsert(chunks)

        retriever = NewsRetriever(store, scorer=default_scorer(cfg.rag["rerank_model"]))

        def retrieve_ids(ticker, question):
            texts = retriever.retrieve(ticker, question, date(2024, 6, 14), k=3, candidate_k=10)
            ids = []
            for rt in texts:
                for d in data["documents"]:
                    if d["text"] in rt:
                        ids.append(d["id"])
                        break
            return ids

        h3 = hit_at_k(retrieve_ids, data["cases"], k=3)
        m = mrr(retrieve_ids, data["cases"], k=3)
        print(f"hit@3={h3:.3f} mrr={m:.3f}")
        assert h3 >= 0.8, f"hit@3 below threshold: {h3:.3f}"
