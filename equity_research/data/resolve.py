from __future__ import annotations

from typing import Callable


def resolve_ticker(query: str, llm_guess: Callable[[str], str | None],
                   price_ok: Callable[[str], bool]) -> dict:
    """Resolve a raw query to a valid ticker.

    Valid as-typed -> pass through. Else the LLM proposes a candidate which is
    accepted only if it has price data (guards against hallucinated symbols).
    Otherwise resolved is None (unknown).
    """
    q = query.strip().upper()
    if not q:
        return {"input": query, "resolved": None, "corrected": False}
    if price_ok(q):
        return {"input": query, "resolved": q, "corrected": False}
    cand = llm_guess(query)
    cand = cand.strip().upper() if cand else None
    if cand and cand != q and price_ok(cand):
        return {"input": query, "resolved": cand, "corrected": True}
    return {"input": query, "resolved": None, "corrected": False}
