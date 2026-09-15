import math

import pandas as pd

from equity_research.analytics.risk import annualized_volatility, beta, max_drawdown


def _s(values):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.Series([float(v) for v in values], index=idx)


def test_volatility_constant_series_is_zero():
    assert annualized_volatility(_s([100, 100, 100])) == 0.0


def test_volatility_known_value():
    # returns [+0.1, -0.1] -> std(ddof=1) = 0.1*sqrt(2); annualized *sqrt(252)
    vol = annualized_volatility(_s([100, 110, 99]))
    assert abs(vol - 0.1 * math.sqrt(2) * math.sqrt(252)) < 1e-9


def test_max_drawdown():
    assert abs(max_drawdown(_s([100, 120, 90, 110])) - 0.25) < 1e-9


def test_beta_two_x_market():
    market = _s([100, 110, 99])   # returns [+0.1, -0.1]
    ticker = _s([100, 120, 96])   # returns [+0.2, -0.2] = 2x market
    assert abs(beta(ticker, market) - 2.0) < 1e-9


def test_beta_nan_when_market_flat():
    market = _s([100, 100, 100])  # zero variance
    ticker = _s([100, 110, 99])
    assert math.isnan(beta(ticker, market))
