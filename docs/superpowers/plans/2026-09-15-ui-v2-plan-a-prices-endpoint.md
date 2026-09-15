# UI v2 — Plan A: `/api/prices` endpoint (backend)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend endpoint that returns price + SMA50/SMA200 + RSI series for a ticker over a chosen period, so the web client can render price/RSI charts.

**Architecture:** A pure function `build_price_series` (in `equity_research/analytics/`) computes the series from a full-history Close Series (SMA/RSI over full history, then sliced to the period window so SMA200 is defined). A thin `core.prices` seam fetches long history (resilient `fetch_yfinance_long`), normalizes tz, and calls the pure function. `GET /api/prices` exposes it as JSON. No agent/eval/LLM code is touched.

**Tech Stack:** Python, pandas, FastAPI, pytest + FastAPI TestClient. Reuses `equity_research.analytics.indicators.rsi`, `equity_research.data.adapters.fetch_yfinance_long`, `equity_research.data.prices._strip_tz`, `equity_research.util.resilient.resilient`.

**Branch:** work on `feature/ui-v2-prices-endpoint` (do not commit to `master` directly).

---

### Task 1: `build_price_series` pure function

Computes candle/SMA/RSI series from a full-history Close Series and slices to a period window.

**Files:**
- Create: `equity_research/analytics/series.py`
- Test: `tests/test_series.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_series.py
import pandas as pd

from equity_research.analytics.series import build_price_series


def _close(n: int) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series([100.0 + i for i in range(n)], index=idx)


def test_candles_sliced_to_period_and_formatted():
    import re

    out = build_price_series(_close(300), period_days=63)
    assert len(out["candles"]) == 63
    pt = out["candles"][-1]
    assert set(pt) == {"time", "value"}
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", pt["time"])  # YYYY-MM-DD
    assert isinstance(pt["value"], float)
    assert pt["value"] == 399.0  # last of 300 points: 100.0 + 299


def test_sma_and_rsi_defined_within_window_only():
    out = build_price_series(_close(300), period_days=63)
    # 300 points of history -> SMA200 is defined for the last window
    assert len(out["sma50"]) == 63
    assert len(out["sma200"]) == 63
    assert len(out["rsi"]) == 63
    assert all(set(p) == {"time", "value"} for p in out["sma200"])


def test_sma200_empty_when_history_too_short():
    out = build_price_series(_close(120), period_days=63)
    assert out["sma200"] == []          # never 200 points -> undefined
    assert len(out["sma50"]) == 63      # sma50 defined for full window
    assert len(out["candles"]) == 63


def test_window_larger_than_history_returns_all_points():
    out = build_price_series(_close(30), period_days=63)
    assert len(out["candles"]) == 30


def test_empty_close_returns_empty_series():
    out = build_price_series(pd.Series(dtype=float), period_days=63)
    assert out == {"candles": [], "sma50": [], "sma200": [], "rsi": []}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_series.py -v`
Expected: FAIL with `ModuleNotFoundError: equity_research.analytics.series`

- [ ] **Step 3: Implement `build_price_series`**

```python
# equity_research/analytics/series.py
from __future__ import annotations

import pandas as pd

from equity_research.analytics.indicators import rsi


def _series(s: pd.Series, window: pd.Index) -> list[dict]:
    """Slice `s` to the window, drop NaN, format as [{time, value}]."""
    s = s.reindex(window).dropna()
    return [{"time": ts.strftime("%Y-%m-%d"), "value": float(v)} for ts, v in s.items()]


def build_price_series(close: pd.Series, period_days: int) -> dict:
    """Price + SMA50/SMA200 + RSI series for the last `period_days` points.

    SMA and RSI are computed over the *full* history, then sliced to the window,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_series.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add equity_research/analytics/series.py tests/test_series.py
git commit -m "feat(analytics): build_price_series (price + SMA + RSI, sliced to period)"
```

---

### Task 2: `core.prices` seam

Maps period label to days, fetches long history resiliently, normalizes tz, calls `build_price_series`.

**Files:**
- Modify: `api/core.py`
- Test: `tests/api/test_prices.py` (new file, Task 3 adds endpoint tests here too)

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_prices.py
import pandas as pd

import api.core as core


def test_prices_maps_period_and_builds_series(monkeypatch):
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    df = pd.DataFrame({"Close": [100.0 + i for i in range(400)]}, index=idx)
    monkeypatch.setattr(core, "_fetch_long_close", lambda ticker: df["Close"])
    out = core.prices("aapl", "3M")
    assert out["ticker"] == "AAPL"
    assert out["period"] == "3M"
    assert len(out["candles"]) == 63          # 3M -> 63 trading days
    assert len(out["sma200"]) == 63           # 400 points -> defined


def test_prices_invalid_period_defaults_to_6m(monkeypatch):
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    df = pd.DataFrame({"Close": [100.0 + i for i in range(400)]}, index=idx)
    monkeypatch.setattr(core, "_fetch_long_close", lambda ticker: df["Close"])
    out = core.prices("AAPL", "bogus")
    assert out["period"] == "6M"
    assert len(out["candles"]) == 126
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_prices.py -v`
Expected: FAIL with `AttributeError: module 'api.core' has no attribute 'prices'`

- [ ] **Step 3: Implement the seam in `api/core.py`**

Add near the top (after existing imports):

```python
_PERIOD_DAYS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def _fetch_long_close(ticker: str):
    from equity_research.config import Config
    from equity_research.data.adapters import fetch_yfinance_long
    from equity_research.data.prices import _strip_tz
    from equity_research.util.resilient import resilient

    cfg = Config.load(CONFIG_PATH)
    df = resilient(fetch_yfinance_long, cfg.net)(ticker)
    df = _strip_tz(df)
    return df["Close"] if not df.empty else __import__("pandas").Series(dtype=float)


def prices(ticker: str, period: str) -> dict:
    from equity_research.analytics.series import build_price_series

    period = period if period in _PERIOD_DAYS else "6M"
    ticker = ticker.upper()
    close = _fetch_long_close(ticker)
    series = build_price_series(close, _PERIOD_DAYS[period])
    return {"ticker": ticker, "period": period, **series}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/api/test_prices.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add api/core.py tests/api/test_prices.py
git commit -m "feat(api): core.prices seam (period->days, resilient long history)"
```

---

### Task 3: `GET /api/prices` endpoint

Exposes `core.prices` as JSON.

**Files:**
- Modify: `api/main.py`
- Test: `tests/api/test_prices.py` (append endpoint test)

- [ ] **Step 1: Write the failing test (append to `tests/api/test_prices.py`)**

```python
from fastapi.testclient import TestClient

import api.main as main


def test_prices_endpoint(monkeypatch):
    monkeypatch.setattr(core, "prices", lambda ticker, period: {
        "ticker": ticker.upper(), "period": period,
        "candles": [{"time": "2024-01-02", "value": 101.0}],
        "sma50": [], "sma200": [], "rsi": [],
    })
    client = TestClient(main.app)
    resp = client.get("/api/prices?ticker=aapl&period=6M")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["period"] == "6M"
    assert body["candles"][0]["value"] == 101.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_prices.py::test_prices_endpoint -v`
Expected: FAIL with 404 (route not defined)

- [ ] **Step 3: Add the route to `api/main.py`**

Add after the `/api/health` route (before `/api/analyze`):

```python
@app.get("/api/prices")
def prices(ticker: str, period: str = "6M"):
    return core.prices(ticker, period)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/api/test_prices.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full test suite (no regressions)**

Run: `uv run pytest -q`
Expected: all pass (integration tests marked separately are skipped without `-m integration`).

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/api/test_prices.py
git commit -m "feat(api): GET /api/prices endpoint"
```

---

### Task 4: Live verification

Confirm the endpoint returns real data end-to-end.

- [ ] **Step 1: Start the API**

```bash
uv run uvicorn api.main:app --port 8000
```

- [ ] **Step 2: Hit the endpoint (in another shell)**

```bash
curl -s "http://localhost:8000/api/prices?ticker=AAPL&period=6M" | python -m json.tool | head -40
```

Expected: JSON with `ticker: "AAPL"`, `period: "6M"`, ~126 candle points, non-empty `sma50`/`sma200`/`rsi`, each point `{time: "YYYY-MM-DD", value: <float>}`. Try `period=1Y` and an invalid `period=xx` (should behave as `6M`).

- [ ] **Step 3: Stop the server (Ctrl-C).** No commit (verification only).

---

## Completion

After all tasks pass, use **superpowers:finishing-a-development-branch** to merge `feature/ui-v2-prices-endpoint` into `master`. All commits authored as OleksandrFIT with NO `Co-Authored-By` trailer (repo convention for this phase).
