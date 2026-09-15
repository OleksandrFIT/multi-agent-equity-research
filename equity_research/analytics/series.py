from __future__ import annotations

import pandas as pd

from equity_research.analytics.indicators import rsi


def _series(s: pd.Series, window: pd.Index) -> list[dict]:
    """Slice `s` to the window, drop NaN, format as [{time, value}]."""
    s = s.reindex(window).dropna()
    return [{"time": ts.strftime("%Y-%m-%d"), "value": float(v)} for ts, v in s.items()]


def build_price_series(close: pd.Series, period_days: int) -> dict:
    """Price + SMA50/SMA200 + RSI series for the last `period_days` points.

    SMA and RSI are computed over the full history, then sliced to the window,
    so SMA200 is defined even near the start of the window. Undefined points
    (NaN, e.g. SMA200 with <200 points of history) are dropped, so a young
    ticker simply yields an empty sma200 list.
    """
    close = close.dropna()
    if close.empty:
        return {"candles": [], "sma50": [], "sma200": [], "rsi": []}
    window = close.index[-period_days:]
    return {
        "candles": _series(close, window),
        "sma50": _series(close.rolling(50).mean(), window),
        "sma200": _series(close.rolling(200).mean(), window),
        "rsi": _series(rsi(close), window),
    }
