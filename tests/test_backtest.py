from datetime import date

import pandas as pd

from equity_research.eval.backtest import BacktestRecord, metrics_for_horizon, run_backtest


class _V:
    def __init__(self, verdict, score):
        self.verdict = verdict
        self.score = score


def _close(vals, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series([float(v) for v in vals], index=idx)


def test_run_backtest_builds_records_with_forward_returns():
    prices = {"AAA": _close(list(range(100, 200)))}

    def run_verdict(ticker, as_of):
        return _V("buy", 0.5)

    recs = run_backtest(["AAA"], [date(2023, 1, 5), date(2023, 1, 10)], [3],
                        run_verdict, lambda t: prices[t])
    assert len(recs) == 2
    assert all(isinstance(r, BacktestRecord) for r in recs)
    assert recs[0].fwd_returns[3] is not None


def test_run_backtest_skips_failed_verdict():
    prices = {"AAA": _close(list(range(100, 200)))}

    def run_verdict(ticker, as_of):
        raise RuntimeError("no data")

    recs = run_backtest(["AAA"], [date(2023, 1, 5)], [3], run_verdict, lambda t: prices[t])
    assert recs == []


def test_metrics_for_horizon_drops_none_returns():
    recs = [
        BacktestRecord("AAA", date(2023, 1, 5), "buy", 0.5, {21: 0.1}),
        BacktestRecord("BBB", date(2023, 1, 5), "sell", -0.5, {21: None}),
    ]
    m = metrics_for_horizon(recs, 21)
    assert m["n"] == 1
    assert m["hit_rate"]["buy"] == 1.0
