from __future__ import annotations

import re

from equity_research.agents.base import AgentOpinion
from equity_research.data.models import Evidence

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _norm(tok: str) -> str:
    try:
        f = float(tok)
    except ValueError:
        return tok
    return ("%f" % f).rstrip("0").rstrip(".")  # 44.600000 -> 44.6 ; 100.000000 -> 100


def _numbers(s: str) -> set[str]:
    return {_norm(n) for n in _NUM.findall(s)}


def ground(opinion: AgentOpinion, evidence: Evidence) -> AgentOpinion:
    metric_numbers: set[str] = set()
    for v in evidence.metrics.values():
        metric_numbers |= _numbers(str(v))
    context_blob = "\n".join(evidence.context).lower()

    kept: list[str] = []
    dropped: list[str] = []
    for fact in opinion.key_facts:
        fact_numbers = _numbers(fact)
        number_supported = bool(fact_numbers & metric_numbers)
        words = [w for w in re.findall(r"[a-zA-Z]{5,}", fact.lower())]
        text_supported = any(w in context_blob for w in words) if context_blob else False
        if number_supported or text_supported:
            kept.append(fact)
        else:
            dropped.append(f"dropped unsupported: {fact}")
    return opinion.model_copy(update={"key_facts": kept, "dropped_facts": dropped})
