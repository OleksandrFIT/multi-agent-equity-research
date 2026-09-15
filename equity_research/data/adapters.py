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
