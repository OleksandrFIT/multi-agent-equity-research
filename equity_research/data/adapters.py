from __future__ import annotations

import pandas as pd


def fetch_yfinance(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
    return df[["Close"]].astype(float) if not df.empty else pd.DataFrame({"Close": []})


def fetch_yfinance_long(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="6y", auto_adjust=True)
    return df[["Close"]].astype(float) if not df.empty else pd.DataFrame({"Close": []})


def fetch_stooq(ticker: str) -> pd.DataFrame:
    from pandas_datareader.stooq import StooqDailyReader

    df = StooqDailyReader(symbols=ticker).read()
    if df.empty:
        return pd.DataFrame({"Close": []})
    df = df.sort_index()
    return df[["Close"]].astype(float)


def _yf_download(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(tickers, period="5d", auto_adjust=True, progress=False)


def fetch_quotes(tickers: list[str]) -> dict[str, dict]:
    """Last price and daily % change per ticker (batch). Missing tickers are omitted."""
    df = _yf_download(tickers)
    if df is None or df.empty:
        return {}
    close = df["Close"]
    out: dict[str, dict] = {}
    for t in tickers:
        try:
            col = close[t] if hasattr(close, "columns") and t in close.columns else close
        except Exception:
            continue
        s = col.dropna() if hasattr(col, "dropna") else col
        if len(s) < 2:
            continue
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        if prev == 0:
            continue
        out[t] = {"price": round(last, 2), "change_pct": round((last - prev) / prev * 100, 2)}
    return out
