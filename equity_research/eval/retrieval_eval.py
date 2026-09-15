from __future__ import annotations

from typing import Callable


def _expected_ids(case: dict) -> set[str]:
    exp = case["expected"]
    return set(exp) if isinstance(exp, list) else {exp}


def hit_at_k(retrieve_fn: Callable[[str, str], list[str]], cases: list[dict], k: int) -> float:
    """Fraction of cases whose expected id(s) appear in the top-k retrieved ids.

    `expected` may be a single id or a list; a case is a hit if ANY expected id
    is in the top-k.
    """
    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        ids = retrieve_fn(case["ticker"], case["question"])[:k]
        if _expected_ids(case) & set(ids):
            hits += 1
    return hits / len(cases)


def mrr(retrieve_fn: Callable[[str, str], list[str]], cases: list[dict], k: int) -> float:
    """Mean Reciprocal Rank over cases (1/rank of the first relevant id in top-k)."""
    if not cases:
        return 0.0
    total = 0.0
    for case in cases:
        wanted = _expected_ids(case)
        ids = retrieve_fn(case["ticker"], case["question"])[:k]
        for rank, doc_id in enumerate(ids, start=1):
            if doc_id in wanted:
                total += 1.0 / rank
                break
    return total / len(cases)
