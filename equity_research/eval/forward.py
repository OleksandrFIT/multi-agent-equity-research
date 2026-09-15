from __future__ import annotations

from datetime import date

import pandas as pd


def forward_return(close: pd.Series, as_of: date, horizon: int) -> float | None:
    """Return over `horizon` trading rows starting from the last close on/before as_of.

    close: full-history Close series with an ascending, tz-naive DatetimeIndex.
    Returns None if as_of predates the history or there are not enough future rows.
    """
    entry_pos = int(close.index.searchsorted(pd.Timestamp(as_of), side="right")) - 1
    if entry_pos < 0:
        return None
    exit_pos = entry_pos + horizon
    if exit_pos >= len(close):
        return None
    entry = float(close.iloc[entry_pos])
    if entry == 0:
        return None
    return float(close.iloc[exit_pos]) / entry - 1.0
