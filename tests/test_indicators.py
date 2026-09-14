import numpy as np
import pandas as pd

from equity_research.analytics.indicators import compute_indicators, rsi


def _series(values):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_rsi_all_gains_is_100():
    s = _series([float(i) for i in range(1, 30)])
    assert round(rsi(s, period=14).iloc[-1], 2) == 100.0


def test_rsi_all_losses_is_0():
    s = _series([float(i) for i in range(30, 1, -1)])
    assert round(rsi(s, period=14).iloc[-1], 2) == 0.0


def test_compute_indicators_shape():
    close = _series(list(np.linspace(100, 200, 260)))
    out = compute_indicators(close)
    assert set(out) >= {"rsi14", "sma50", "sma200", "trend_pct"}
    assert out["sma50"] > 0
    assert out["trend_pct"] > 0  # rising series
