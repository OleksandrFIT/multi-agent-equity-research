from __future__ import annotations

import math


def _safe_div(a: float, b: float) -> float:
    return a / b if b else math.nan


def compute_fundamental_metrics(facts: dict[str, float], price: float) -> dict[str, float]:
    eps = facts.get("eps_ttm", 0.0)
    rev = facts.get("revenue", 0.0)
    rev_prev = facts.get("revenue_prev", 0.0)
    return {
        "pe": _safe_div(price, eps),
        "roe": _safe_div(facts.get("net_income", 0.0), facts.get("equity", 0.0)),
        "debt_to_equity": _safe_div(facts.get("total_debt", 0.0), facts.get("equity", 0.0)),
        "revenue_growth": _safe_div(rev - rev_prev, rev_prev) if rev_prev else math.nan,
        "operating_margin": _safe_div(facts.get("operating_income", math.nan), rev),
        "net_margin": _safe_div(facts.get("net_income", 0.0), rev),
        "fcf_margin": _safe_div(facts.get("free_cash_flow", math.nan), rev),
        "current_ratio": facts.get("current_ratio", math.nan),
    }
