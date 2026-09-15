from __future__ import annotations

from equity_research.util.resilient import resilient_call


class ChromaVectorStore:
    """Thin Chroma wrapper implementing the VectorStore protocol.

    When a `net` config is supplied, embed-backed operations are retried, so a
    transient Ollama embedding failure (e.g. a reset /tokenize call) does not
    abort an agent.
    """

    def __init__(self, persist_dir: str, embed_model: str, collection: str = "news", net: dict | None = None):
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        self._db = Chroma(
            collection_name=collection,
            persist_directory=persist_dir,
            embedding_function=OllamaEmbeddings(model=embed_model),
        )
        self._net = net

    def _call(self, fn):
        if self._net is None:
            return fn()
        return resilient_call(fn, timeout=self._net["data_timeout"],
                              attempts=self._net["data_attempts"], base_delay=self._net["data_base_delay"])

    def existing_ids(self, ids: list[str]) -> set[str]:
        return self._call(lambda: set(self._db.get(ids=ids).get("ids", [])))

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        self._call(lambda: self._db.add_texts(texts=texts, metadatas=metadatas, ids=ids))

    def mmr_search(self, query: str, where: dict, k: int) -> list[tuple[str, dict]]:
        return self._call(lambda: [(d.page_content, d.metadata)
                                   for d in self._db.max_marginal_relevance_search(query, k=k, filter=where)])
