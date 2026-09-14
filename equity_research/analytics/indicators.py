from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    out = out.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return out


def compute_indicators(close: pd.Series) -> dict[str, float]:
    close = close.dropna()
    out: dict[str, float] = {}
    if len(close) >= 15:
        out["rsi14"] = float(rsi(close).iloc[-1])
    if len(close) >= 50:
        out["sma50"] = float(close.rolling(50).mean().iloc[-1])
    if len(close) >= 200:
        out["sma200"] = float(close.rolling(200).mean().iloc[-1])
    window = close.iloc[-63:] if len(close) >= 63 else close
    out["trend_pct"] = float((window.iloc[-1] - window.iloc[0]) / window.iloc[0] * 100.0)
    return out
