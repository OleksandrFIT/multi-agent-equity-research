from __future__ import annotations

import math
import re

from equity_research.agents.base import AgentOpinion
from equity_research.data.models import Evidence

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _metric_values(evidence: Evidence) -> list[float]:
    vals = []
    for v in evidence.metrics.values():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            vals.append(f)
    return vals


def _number_supported(fact: str, metric_values: list[float]) -> bool:
    for tok in _NUM.findall(fact):
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        f = float(tok)
        if any(round(m, decimals) == round(f, decimals) for m in metric_values):
            return True
    return False


def ground(opinion: AgentOpinion, evidence: Evidence) -> AgentOpinion:
    metric_values = _metric_values(evidence)
    context_blob = "\n".join(evidence.context).lower()

    kept: list[str] = []
    dropped: list[str] = []
    for fact in opinion.key_facts:
        number_supported = _number_supported(fact, metric_values)
        words = re.findall(r"[a-zA-Z]{5,}", fact.lower())
        text_supported = any(w in context_blob for w in words) if context_blob else False
        if number_supported or text_supported:
            kept.append(fact)
        else:
            dropped.append(f"dropped unsupported: {fact}")
    return opinion.model_copy(update={"key_facts": kept, "dropped_facts": dropped})
