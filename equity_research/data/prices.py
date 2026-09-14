from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd
import pandera as pa

PriceFetcher = Callable[[str], pd.DataFrame]


class PriceValidationError(ValueError):
    pass


_SCHEMA = pa.DataFrameSchema(
    {"Close": pa.Column(float, checks=pa.Check.gt(0))},
    index=pa.Index(pa.dtypes.Timestamp),
    strict=False,
)


class PriceProvider:
    def __init__(self, fetch_yfinance: PriceFetcher, fetch_stooq: PriceFetcher, tolerance: float = 0.10):
        self.fetch_yfinance = fetch_yfinance
        self.fetch_stooq = fetch_stooq
        self.tolerance = tolerance

    def history(self, ticker: str, as_of: date) -> tuple[pd.DataFrame, str | None]:
        primary = self.fetch_yfinance(ticker)
        if primary is None or primary.empty:
            raise PriceValidationError(f"no price history for {ticker}")
        cutoff = pd.Timestamp(as_of)
        primary = primary[primary.index <= cutoff]
        if primary.empty:
            raise PriceValidationError(f"no price history for {ticker} as of {as_of}")
        _SCHEMA.validate(primary)

        note = None
        secondary = self.fetch_stooq(ticker)
        if secondary is not None and not secondary.empty:
            secondary = secondary[secondary.index <= cutoff]
            if not secondary.empty:
                p_last = float(primary["Close"].iloc[-1])
                s_last = float(secondary["Close"].iloc[-1])
                if abs(p_last - s_last) / p_last > self.tolerance:
                    note = (
                        f"price discrepancy: yfinance {p_last:.2f} vs stooq {s_last:.2f} "
                        f"(> {self.tolerance:.0%})"
                    )
        return primary, note
