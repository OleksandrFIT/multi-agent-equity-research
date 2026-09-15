from __future__ import annotations

from typing import Callable


def hit_at_k(retrieve_fn: Callable[[str, str], list[str]], cases: list[dict], k: int) -> float:
    """Fraction of cases whose expected doc id appears in the top-k retrieved ids."""
    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        ids = retrieve_fn(case["ticker"], case["question"])[:k]
        if case["expected"] in ids:
            hits += 1
    return hits / len(cases)
