from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from equity_research.eval.forward import forward_return
from equity_research.eval.metrics import (
    hit_rate_by_class,
    information_coefficient,
    long_short_curve,
    mean_return_by_class,
)


@dataclass
class BacktestRecord:
    ticker: str
    as_of: date
    verdict: str
    score: float
    fwd_returns: dict[int, float | None]


def run_backtest(universe, dates, horizons, run_verdict: Callable, price_history: Callable) -> list[BacktestRecord]:
    records: list[BacktestRecord] = []
    for ticker in universe:
        try:
            close: pd.Series = price_history(ticker)
        except Exception:
            continue
        for as_of in dates:
            try:
                verdict = run_verdict(ticker, as_of)
            except Exception:
                continue
            fwd = {h: forward_return(close, as_of, h) for h in horizons}
            records.append(BacktestRecord(ticker, as_of, verdict.verdict, verdict.score, fwd))
    return records


def metrics_for_horizon(records: list[BacktestRecord], horizon: int) -> dict:
    rows = [
        {"verdict": r.verdict, "score": r.score, "fwd_return": r.fwd_returns[horizon]}
        for r in records
        if r.fwd_returns.get(horizon) is not None
    ]
    return {
        "n": len(rows),
        "hit_rate": hit_rate_by_class(rows),
        "mean_return": mean_return_by_class(rows),
        "ic": information_coefficient(rows),
        "long_short_curve": long_short_curve(rows),
    }
