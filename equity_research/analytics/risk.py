from __future__ import annotations

import math

import pandas as pd


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualized_volatility(close: pd.Series, periods_per_year: int = 252) -> float:
    r = _returns(close)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def max_drawdown(close: pd.Series) -> float:
    close = close.dropna()
    if close.empty:
        return float("nan")
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    return float(-drawdown.min())


def beta(ticker_close: pd.Series, market_close: pd.Series) -> float:
    joined = pd.concat([_returns(ticker_close), _returns(market_close)], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return float("nan")
    y = joined.iloc[:, 0]
    x = joined.iloc[:, 1]
    var = float(x.var(ddof=1))
    if var == 0:
        return float("nan")
    return float(y.cov(x) / var)
