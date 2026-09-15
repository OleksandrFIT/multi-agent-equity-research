from __future__ import annotations

from typing import Callable

Scorer = Callable[[str, list[str]], list[float]]


def rerank(query: str, items: list[tuple[str, dict]], scorer: Scorer | None) -> list[tuple[str, dict]]:
    if scorer is None or not items:
        return items
    texts = [t for t, _ in items]
    scores = scorer(query, texts)
    order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    return [items[i] for i in order]


def default_scorer(model_name: str) -> Scorer | None:
    """Cross-encoder scorer if sentence-transformers is installed, else None (fallback to MMR order).

    Verify model download/behavior in the integration task; requires the `rerank` extra.
    """
    try:
        from sentence_transformers import CrossEncoder
        encoder = CrossEncoder(model_name)
    except Exception:
        return None

    def score(query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in encoder.predict([(query, t) for t in texts])]

    return score
