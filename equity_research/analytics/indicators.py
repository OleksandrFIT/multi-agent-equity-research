from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, out)
    out = out.mask((avg_gain == 0) & (avg_loss != 0), 0.0)
    return out


def compute_indicators(close: pd.Series) -> dict[str, float]:
    close = close.dropna()
    sma50 = close.rolling(50, min_periods=1).mean().iloc[-1]
    sma200 = close.rolling(200, min_periods=1).mean().iloc[-1]
    rsi14 = rsi(close).iloc[-1]
    window = close.iloc[-63:] if len(close) >= 63 else close
    trend_pct = float((window.iloc[-1] - window.iloc[0]) / window.iloc[0] * 100.0)
    return {
        "rsi14": float(rsi14),
        "sma50": float(sma50),
        "sma200": float(sma200),
        "trend_pct": trend_pct,
    }
