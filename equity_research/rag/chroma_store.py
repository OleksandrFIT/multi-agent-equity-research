from __future__ import annotations


class ChromaVectorStore:
    """Thin Chroma wrapper implementing the VectorStore protocol.

    API specifics (langchain-chroma / chromadb versions) are verified live in a
    later task; adjust only this file if they differ.
    """

    def __init__(self, persist_dir: str, embed_model: str, collection: str = "news"):
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        self._db = Chroma(
            collection_name=collection,
            persist_directory=persist_dir,
            embedding_function=OllamaEmbeddings(model=embed_model),
        )

    def existing_ids(self, ids: list[str]) -> set[str]:
        got = self._db.get(ids=ids)
        return set(got.get("ids", []))

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        self._db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def mmr_search(self, query: str, where: dict, k: int) -> list[tuple[str, dict]]:
        docs = self._db.max_marginal_relevance_search(query, k=k, filter=where)
        return [(d.page_content, d.metadata) for d in docs]
