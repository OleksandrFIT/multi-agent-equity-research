from datetime import date

import pandas as pd

from equity_research.eval.forward import forward_return


def _close(vals, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series([float(v) for v in vals], index=idx)


def test_forward_return_basic():
    close = _close([100, 101, 102, 103, 104, 105])
    r = forward_return(close, date(2025, 1, 1), horizon=3)
    assert abs(r - (103 / 100 - 1)) < 1e-9


def test_uses_last_row_on_or_before_as_of():
    close = _close([100, 101, 102, 103, 104, 105])
    r = forward_return(close, date(2025, 1, 3), horizon=2)
    assert abs(r - (104 / 102 - 1)) < 1e-9


def test_none_when_not_enough_future():
    close = _close([100, 101, 102])
    assert forward_return(close, date(2025, 1, 2), horizon=5) is None


def test_none_when_as_of_before_history():
    close = _close([100, 101, 102], start="2025-01-05")
    assert forward_return(close, date(2025, 1, 1), horizon=1) is None
