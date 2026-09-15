# Phase 3B — Backtest engine + metrics + hit@k Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit attribution:** commit ONLY as the repo author (OleksandrFIT). Do NOT add any `Co-Authored-By` trailer.

**Goal:** A historical backtest that runs the point-in-time agents (fundamentals as-of + technical + risk, no sentiment) over a small ticker×date grid, scores each verdict against 21- and 63-trading-day forward returns, and reports hit rate, mean return by class, information coefficient, and a naive long-short curve — plus a reproducible frozen-corpus retrieval hit@k.

**Architecture:** Pure, injectable units: `forward_return` (price → return), `metrics` (records → stats), `run_backtest` (engine driven by injected `run_verdict` and `price_history`), `report` (records → MD/JSON). The CLI wires the real no-sentiment orchestrator and a long-history price fetcher. hit@k ingests a frozen labeled news corpus and measures top-k retrieval (integration).

**Tech Stack:** existing (pandas, pydantic, typer, ollama, chromadb). No new deps (Spearman is hand-rolled — no scipy).

**Scope:** Plan 3B only (3A as-of EDGAR already merged). Spec: `docs/superpowers/specs/2026-09-15-phase-3-eval-backtest-design.md`.

## File Structure

- `equity_research/eval/__init__.py`
- `equity_research/eval/forward.py` — `forward_return(close, as_of, horizon)`.
- `equity_research/eval/metrics.py` — `hit_rate_by_class`, `mean_return_by_class`, `information_coefficient`, `long_short_curve`.
- `equity_research/eval/backtest.py` — `BacktestRecord`, `run_backtest`, `metrics_for_horizon`.
- `equity_research/eval/report.py` — `render_backtest_markdown`, `render_backtest_json`.
- `equity_research/eval/retrieval_eval.py` — `hit_at_k`.
- `equity_research/data/adapters.py` — add `fetch_yfinance_long`.
- `equity_research/config.py` + `config.yaml` — `backtest` block.
- `equity_research/cli.py` — `backtest` command + no-sentiment verdict builder.
- Tests mirror each; `tests/fixtures/hitk_corpus.json` for hit@k.

Naming contract: `forward_return(close, as_of, horizon)`, `hit_rate_by_class(records)`, `mean_return_by_class(records)`, `information_coefficient(records)`, `long_short_curve(records)`, `BacktestRecord(ticker, as_of, verdict, score, fwd_returns)`, `run_backtest(universe, dates, horizons, run_verdict, price_history)`, `metrics_for_horizon(records, horizon)`, `hit_at_k(retrieve_fn, cases, k)`.

---

## Task 1: forward_return (pure)

**Files:**
- Create: `equity_research/eval/__init__.py` (empty)
- Create: `equity_research/eval/forward.py`
- Test: `tests/test_forward.py`

- [ ] **Step 1: Write the failing test**

`tests/test_forward.py`:
```python
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
    # as_of falls on 2025-01-03 (value 102); +2 rows -> 104
    r = forward_return(close, date(2025, 1, 3), horizon=2)
    assert abs(r - (104 / 102 - 1)) < 1e-9


def test_none_when_not_enough_future():
    close = _close([100, 101, 102])
    assert forward_return(close, date(2025, 1, 2), horizon=5) is None


def test_none_when_as_of_before_history():
    close = _close([100, 101, 102], start="2025-01-05")
    assert forward_return(close, date(2025, 1, 1), horizon=1) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_forward.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/eval/__init__.py`: empty.

`equity_research/eval/forward.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_forward.py -v`
Expected: PASS (4 tests). Full suite `uv run pytest -m "not integration" -p no:warnings -q` green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/eval/__init__.py equity_research/eval/forward.py tests/test_forward.py
git commit -m "feat(eval): forward_return over trading-day horizon"
```

---

## Task 2: metrics (pure)

**Files:**
- Create: `equity_research/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_eval_metrics.py`:
```python
from equity_research.eval.metrics import (
    hit_rate_by_class,
    information_coefficient,
    long_short_curve,
    mean_return_by_class,
)


def _rec(verdict, score, fwd):
    return {"verdict": verdict, "score": score, "fwd_return": fwd}


def test_hit_rate_by_class():
    recs = [_rec("buy", 0.5, 0.1), _rec("buy", 0.4, -0.1), _rec("sell", -0.5, -0.2)]
    hr = hit_rate_by_class(recs)
    assert hr["buy"] == 0.5   # one of two buys went up
    assert hr["sell"] == 1.0  # the one sell went down


def test_mean_return_by_class():
    recs = [_rec("buy", 0.5, 0.1), _rec("buy", 0.4, -0.1), _rec("hold", 0.0, 0.05)]
    mr = mean_return_by_class(recs)
    assert abs(mr["buy"] - 0.0) < 1e-9
    assert abs(mr["hold"] - 0.05) < 1e-9
    assert "sell" not in mr  # no sell records


def test_information_coefficient_monotonic_is_one():
    recs = [_rec("x", 0.1, 1.0), _rec("x", 0.2, 2.0), _rec("x", 0.3, 3.0)]
    assert abs(information_coefficient(recs) - 1.0) < 1e-9


def test_long_short_curve():
    recs = [_rec("buy", 0.5, 0.1), _rec("sell", -0.5, 0.2), _rec("hold", 0.0, 0.9)]
    curve = long_short_curve(recs)
    # +0.1, then -0.2 (short a +0.2 move), then 0 -> cumulative
    assert [round(c, 4) for c in curve] == [0.1, -0.1, -0.1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/eval/metrics.py`:
```python
from __future__ import annotations


def hit_rate_by_class(records: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    buys = [r for r in records if r["verdict"] == "buy"]
    sells = [r for r in records if r["verdict"] == "sell"]
    if buys:
        out["buy"] = sum(1 for r in buys if r["fwd_return"] > 0) / len(buys)
    if sells:
        out["sell"] = sum(1 for r in sells if r["fwd_return"] < 0) / len(sells)
    return out


def mean_return_by_class(records: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls in ("buy", "hold", "sell"):
        rs = [r["fwd_return"] for r in records if r["verdict"] == cls]
        if rs:
            out[cls] = sum(rs) / len(rs)
    return out


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def information_coefficient(records: list[dict]) -> float | None:
    if len(records) < 2:
        return None
    ra = _ranks([r["score"] for r in records])
    rb = _ranks([r["fwd_return"] for r in records])
    n = len(records)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def long_short_curve(records: list[dict]) -> list[float]:
    curve: list[float] = []
    cum = 0.0
    for r in records:
        if r["verdict"] == "buy":
            cum += r["fwd_return"]
        elif r["verdict"] == "sell":
            cum -= r["fwd_return"]
        curve.append(cum)
    return curve
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_metrics.py -v`
Expected: PASS (4 tests). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): backtest metrics (hit rate, mean, IC, long-short)"
```

---

## Task 3: backtest engine

**Files:**
- Create: `equity_research/eval/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test**

`tests/test_backtest.py`:
```python
from datetime import date

import pandas as pd

from equity_research.eval.backtest import BacktestRecord, metrics_for_horizon, run_backtest


class _V:
    def __init__(self, verdict, score):
        self.verdict = verdict
        self.score = score


def _close(vals, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series([float(v) for v in vals], index=idx)


def test_run_backtest_builds_records_with_forward_returns():
    prices = {"AAA": _close(list(range(100, 200)))}  # rising

    def run_verdict(ticker, as_of):
        return _V("buy", 0.5)

    recs = run_backtest(["AAA"], [date(2023, 1, 5), date(2023, 1, 10)], [3],
                        run_verdict, lambda t: prices[t])
    assert len(recs) == 2
    assert all(isinstance(r, BacktestRecord) for r in recs)
    assert recs[0].fwd_returns[3] is not None


def test_run_backtest_skips_failed_verdict():
    prices = {"AAA": _close(list(range(100, 200)))}

    def run_verdict(ticker, as_of):
        raise RuntimeError("no data")

    recs = run_backtest(["AAA"], [date(2023, 1, 5)], [3], run_verdict, lambda t: prices[t])
    assert recs == []


def test_metrics_for_horizon_drops_none_returns():
    recs = [
        BacktestRecord("AAA", date(2023, 1, 5), "buy", 0.5, {21: 0.1}),
        BacktestRecord("BBB", date(2023, 1, 5), "sell", -0.5, {21: None}),
    ]
    m = metrics_for_horizon(recs, 21)
    assert m["n"] == 1  # the None-return row is excluded
    assert m["hit_rate"]["buy"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/eval/backtest.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd

from equity_research.eval.forward import forward_return
from equity_research.eval.metrics import (
    hit_rate_by_class,
    information_coefficient,
    long_short_curve,
    mean_return_by_class,
)


@dataclass
class BacktestRecord:
    ticker: str
    as_of: date
    verdict: str
    score: float
    fwd_returns: dict[int, float | None]


def run_backtest(universe, dates, horizons, run_verdict: Callable, price_history: Callable) -> list[BacktestRecord]:
    records: list[BacktestRecord] = []
    for ticker in universe:
        try:
            close: pd.Series = price_history(ticker)
        except Exception:
            continue
        for as_of in dates:
            try:
                verdict = run_verdict(ticker, as_of)
            except Exception:
                continue
            fwd = {h: forward_return(close, as_of, h) for h in horizons}
            records.append(BacktestRecord(ticker, as_of, verdict.verdict, verdict.score, fwd))
    return records


def metrics_for_horizon(records: list[BacktestRecord], horizon: int) -> dict:
    rows = [
        {"verdict": r.verdict, "score": r.score, "fwd_return": r.fwd_returns[horizon]}
        for r in records
        if r.fwd_returns.get(horizon) is not None
    ]
    return {
        "n": len(rows),
        "hit_rate": hit_rate_by_class(rows),
        "mean_return": mean_return_by_class(rows),
        "ic": information_coefficient(rows),
        "long_short_curve": long_short_curve(rows),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS (3 tests). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/eval/backtest.py tests/test_backtest.py
git commit -m "feat(eval): backtest engine (records + per-horizon metrics)"
```

---

## Task 4: backtest report

**Files:**
- Create: `equity_research/eval/report.py`
- Test: `tests/test_backtest_report.py`

- [ ] **Step 1: Write the failing test**

`tests/test_backtest_report.py`:
```python
import json
from datetime import date

from equity_research.eval.backtest import BacktestRecord
from equity_research.eval.report import render_backtest_json, render_backtest_markdown


def _recs():
    return [
        BacktestRecord("AAA", date(2023, 1, 5), "buy", 0.5, {21: 0.1, 63: 0.2}),
        BacktestRecord("BBB", date(2023, 1, 5), "sell", -0.4, {21: -0.05, 63: None}),
    ]


def test_markdown_has_horizons_and_metrics():
    md = render_backtest_markdown(_recs(), horizons=[21, 63])
    assert "21" in md and "63" in md
    assert "hit rate" in md.lower()
    assert "information coefficient" in md.lower()


def test_json_roundtrips_metrics():
    data = json.loads(render_backtest_json(_recs(), horizons=[21, 63]))
    assert "21" in data["horizons"]
    assert data["horizons"]["21"]["n"] == 2
    assert data["n_records"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_report.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/eval/report.py`:
```python
from __future__ import annotations

import json

from equity_research.eval.backtest import BacktestRecord, metrics_for_horizon

DISCLAIMER = "Backtest on a small sample; not investment advice. Past performance is not predictive."


def _horizon_dict(records: list[BacktestRecord], horizons: list[int]) -> dict:
    return {str(h): metrics_for_horizon(records, h) for h in horizons}


def render_backtest_json(records: list[BacktestRecord], horizons: list[int]) -> str:
    return json.dumps({
        "n_records": len(records),
        "horizons": _horizon_dict(records, horizons),
        "disclaimer": DISCLAIMER,
    }, indent=2)


def render_backtest_markdown(records: list[BacktestRecord], horizons: list[int]) -> str:
    lines = [f"# Backtest report ({len(records)} verdicts)", ""]
    for h in horizons:
        m = metrics_for_horizon(records, h)
        lines.append(f"## Horizon {h} trading days (n={m['n']})")
        ic = "n/a" if m["ic"] is None else f"{m['ic']:+.3f}"
        lines.append(f"- Information coefficient (score vs return): {ic}")
        lines.append("- Hit rate: " + (", ".join(f"{k} {v:.0%}" for k, v in m["hit_rate"].items()) or "n/a"))
        lines.append("- Mean forward return: " + (", ".join(f"{k} {v:+.2%}" for k, v in m["mean_return"].items()) or "n/a"))
        curve = m["long_short_curve"]
        final = curve[-1] if curve else 0.0
        lines.append(f"- Naive long-short cumulative return: {final:+.2%}")
        lines.append("")
    lines += ["---", f"> {DISCLAIMER}"]
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backtest_report.py -v`
Expected: PASS (2 tests). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/eval/report.py tests/test_backtest_report.py
git commit -m "feat(eval): backtest report (Markdown + JSON)"
```

---

## Task 5: config backtest block + long-history fetcher

**Files:**
- Modify: `equity_research/config.py`
- Modify: `config.yaml`
- Modify: `equity_research/data/adapters.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:
```python
def test_config_has_backtest_defaults():
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".cache",
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    assert cfg.backtest["horizons"] == [21, 63]
    assert len(cfg.backtest["universe"]) >= 1
    assert cfg.backtest["report_path"].endswith(".md")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_config_has_backtest_defaults -v`
Expected: FAIL — `Config` has no `backtest`.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/config.py`, add a `backtest` field to `Config` (after `net`):
```python
    backtest: dict = Field(default_factory=lambda: {
        "universe": ["AAPL", "MSFT", "KO", "JPM", "XOM"],
        "dates": ["2024-03-15", "2024-06-14", "2024-09-13", "2024-12-13"],
        "horizons": [21, 63],
        "report_path": "backtest_report.md",
    })
```

In `config.yaml`, add:
```yaml
backtest:
  universe: [AAPL, MSFT, KO, JPM, XOM]
  dates: ["2024-03-15", "2024-06-14", "2024-09-13", "2024-12-13"]
  horizons: [21, 63]
  report_path: backtest_report.md
```

In `equity_research/data/adapters.py`, add a longer-history fetcher (backtest dates need years of history for both the as-of agent view and forward returns):
```python
def fetch_yfinance_long(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="6y", auto_adjust=True)
    return df[["Close"]].astype(float) if not df.empty else pd.DataFrame({"Close": []})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS. Confirm `uv run python -c "import equity_research.data.adapters"` works offline.

- [ ] **Step 5: Commit**

```bash
git add equity_research/config.py config.yaml equity_research/data/adapters.py tests/test_config.py
git commit -m "chore(eval): backtest config block + long-history fetcher"
```

---

## Task 6: CLI backtest command

**Files:**
- Modify: `equity_research/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:
```python
def test_backtest_builder_excludes_sentiment(monkeypatch):
    # the backtest verdict-builder must wire fundamentals+technical+risk, NOT sentiment
    import equity_research.cli as cli_module
    from equity_research.config import Config

    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator):
            captured["agents"] = [a.name for a in agents]

        def run(self, ticker, as_of):
            from equity_research.orchestration.aggregator import Verdict
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    cfg = Config.load("config.yaml")

    class FakeClient:
        pass

    run_verdict = cli_module.build_backtest_verdict(cfg, FakeClient())
    run_verdict("AAPL", date(2024, 3, 15))
    assert "sentiment" not in captured["agents"]
    assert {"fundamentals", "technical", "risk"} == set(captured["agents"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_backtest_builder_excludes_sentiment -v`
Expected: FAIL — `build_backtest_verdict` does not exist.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/cli.py`, add the imports:
```python
from datetime import date, datetime
from pathlib import Path

from equity_research.data.adapters import fetch_yfinance_long
from equity_research.eval.backtest import run_backtest
from equity_research.eval.report import render_backtest_json, render_backtest_markdown
```
(Adjust the existing `from datetime import date` line to also import `datetime`.)

Add a verdict-builder factory (no sentiment) and a full-history price fetcher, plus the `backtest` command:
```python
def build_backtest_verdict(cfg: Config, client):
    prices = PriceProvider(fetch_yfinance=resilient(fetch_yfinance_long, cfg.net),
                           fetch_stooq=resilient(fetch_stooq, cfg.net))
    edgar = EdgarProvider(user_agent=cfg.edgar_user_agent)
    edgar.company_facts = resilient(edgar.company_facts, cfg.net)

    def run_verdict(ticker, as_of):
        agents = [
            FundamentalsAgent(facts_source=edgar, prices=prices, client=client),
            TechnicalAgent(prices=prices, client=client),
            RiskAgent(prices=prices, client=client, benchmark=cfg.benchmark, risk_cfg=cfg.risk),
        ]
        return Orchestrator(agents=agents, aggregator=Aggregator(cfg, client)).run(ticker, as_of)

    return run_verdict


def _full_close(ticker: str):
    df = fetch_yfinance_long(ticker)
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df["Close"]


@app.command()
def backtest(config: str = "config.yaml"):
    cfg = Config.load(config)
    from ollama import Client

    chat_fn = Client(host=cfg.ollama_host, timeout=cfg.net["ollama_timeout"]).chat
    client = OllamaClient(model=cfg.model, cache=DiskCache(cfg.cache_dir),
                          seed=cfg.seed, temperature=cfg.temperature, chat_fn=chat_fn)
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in cfg.backtest["dates"]]
    records = run_backtest(cfg.backtest["universe"], dates, cfg.backtest["horizons"],
                           build_backtest_verdict(cfg, client), _full_close)
    md = render_backtest_markdown(records, cfg.backtest["horizons"])
    Path(cfg.backtest["report_path"]).write_text(md)
    Path(cfg.backtest["report_path"].replace(".md", ".json")).write_text(
        render_backtest_json(records, cfg.backtest["horizons"]))
    typer.echo(md)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (existing CLI tests + new one). Confirm `uv run python -c "import equity_research.cli"` succeeds offline. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/cli.py tests/test_cli.py
git commit -m "feat(cli): backtest command (no-sentiment verdict builder + report)"
```

---

## Task 7: retrieval hit@k (frozen corpus)

**Files:**
- Create: `equity_research/eval/retrieval_eval.py`
- Create: `tests/test_retrieval_eval.py`
- Create: `tests/fixtures/hitk_corpus.json`
- Create: `tests/integration/test_hitk_live.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retrieval_eval.py` (pure hit@k logic with a fake retriever):
```python
from equity_research.eval.retrieval_eval import hit_at_k


def test_hit_at_k_counts_expected_in_topk():
    # retrieve_fn(ticker, question) -> list of doc ids (ranked)
    def retrieve_fn(ticker, question):
        return {"q1": ["d1", "d2", "d3"], "q2": ["dx", "dy", "dz"]}[question]

    cases = [
        {"ticker": "AAA", "question": "q1", "expected": "d2"},  # in top-3
        {"ticker": "AAA", "question": "q2", "expected": "d9"},  # not present
    ]
    assert hit_at_k(retrieve_fn, cases, k=3) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval_eval.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/eval/retrieval_eval.py`:
```python
from __future__ import annotations

from typing import Callable


def hit_at_k(retrieve_fn: Callable[[str, str], list[str]], cases: list[dict], k: int) -> float:
    """Fraction of cases whose expected doc id appears in the top-k retrieved ids."""
    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        ids = retrieve_fn(case["ticker"], case["question"])[:k]
        if case["expected"] in ids:
            hits += 1
    return hits / len(cases)
```

`tests/fixtures/hitk_corpus.json` (frozen labeled corpus — small, self-contained):
```json
{
  "documents": [
    {"id": "aapl-earnings", "ticker": "AAPL", "text": "Apple reported record quarterly revenue driven by strong iPhone and services growth."},
    {"id": "aapl-lawsuit", "ticker": "AAPL", "text": "Apple faces an antitrust investigation over App Store fees in Europe."},
    {"id": "aapl-supply", "ticker": "AAPL", "text": "Apple warns of supply chain constraints affecting Mac production this quarter."}
  ],
  "cases": [
    {"ticker": "AAPL", "question": "How were Apple earnings and revenue?", "expected": "aapl-earnings"},
    {"ticker": "AAPL", "question": "Any legal or regulatory problems for Apple?", "expected": "aapl-lawsuit"},
    {"ticker": "AAPL", "question": "Supply chain or production issues?", "expected": "aapl-supply"}
  ]
}
```

`tests/integration/test_hitk_live.py`:
```python
import json
from datetime import date
from pathlib import Path

import pytest

FIX = Path(__file__).parent.parent / "fixtures" / "hitk_corpus.json"


@pytest.mark.integration
def test_hitk_frozen_corpus(tmp_path):
    """Requires Ollama (nomic-embed-text) + chromadb. Run: uv run pytest -m integration."""
    from equity_research.eval.retrieval_eval import hit_at_k
    from equity_research.rag.chroma_store import ChromaVectorStore
    from equity_research.rag.store import NewsStore

    corpus = json.loads(FIX.read_text())
    vs = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"),
                           embed_model="nomic-embed-text", collection="hitk")
    ids = [d["id"] for d in corpus["documents"]]
    texts = [d["text"] for d in corpus["documents"]]
    metas = [{"doc_type": "news", "ticker": d["ticker"], "date_int": 20240101,
              "content_hash": d["id"]} for d in corpus["documents"]]
    vs.add(ids, texts, metas)
    store = NewsStore(vs)

    def retrieve_fn(ticker, question):
        hits = store.search(question, ticker=ticker, as_of=date(2024, 6, 1), k=3)
        return [m["content_hash"] for _, m in hits]

    score = hit_at_k(retrieve_fn, corpus["cases"], k=3)
    assert score >= 0.6  # embeddings should rank the labeled doc into the top-3
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_retrieval_eval.py -v`
Expected: PASS (1 test). Full suite green; the live hit@k test is deselected by default.

- [ ] **Step 5: Commit**

```bash
git add equity_research/eval/retrieval_eval.py tests/test_retrieval_eval.py tests/fixtures/hitk_corpus.json tests/integration/test_hitk_live.py
git commit -m "feat(eval): retrieval hit@k on frozen corpus (+ live integration)"
```

---

## Task 8: Full suite + live backtest verification

**Files:** none (verification), unless a live issue needs a wrapper fix.

- [ ] **Step 1: Full unit suite**

Run: `uv run pytest -m "not integration" -p no:warnings -q`
Expected: all PASS.

- [ ] **Step 2: Live hit@k**

Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run pytest -m integration -q`
Expected: PASS (incl. the new frozen-corpus hit@k and prior live tests).

- [ ] **Step 3: Live backtest (small)**

Temporarily shrink the run for a quick smoke: `DISABLE_PANDERA_IMPORT_WARNING=True uv run python -m equity_research.cli backtest` — but first, to keep it fast, edit `config.yaml` `backtest.universe` to a single ticker and `dates` to one date, run, confirm a `backtest_report.md` / `.json` is written and the report renders with both horizons. Then restore the config. (This first run is slow: real LLM + EDGAR + prices per (ticker×date).)
Expected: report file written; metrics render (n may be small); no crash. If the long-history price index or forward-return alignment misbehaves on real data, fix `_full_close` / `forward.py` only if it is a real bug (do not change the pure tests).

- [ ] **Step 4: Commit any fix**

```bash
git add -A && git commit -m "fix(eval): backtest live-integration adjustments"
```
(Skip if nothing changed. Restore config.yaml to the full universe/dates before committing.)

---

## Self-Review (completed during authoring)

- **Spec coverage (3B):** forward returns 21d/63d (Task 1), all four metrics incl. IC and long-short (Task 2), engine over universe×dates with no-sentiment orchestrator (Tasks 3, 6), MD+JSON report (Task 4), backtest config block + long-history fetch (Task 5), CLI `backtest` (Task 6), frozen-corpus hit@k with reproducible integration (Task 7), full+live verification (Task 8). Sentiment exclusion is enforced in `build_backtest_verdict` (Task 6). Point-in-time: agents use `PriceProvider.history(ticker, as_of)` (as-of) + as-of EDGAR from 3A; forward returns use a separate full-history series — clean separation.
- **Placeholder scan:** none. The only non-code artifact is the frozen hit@k corpus (real labeled text). CLI wiring is concrete.
- **Type consistency:** `forward_return(close, as_of, horizon)`, the record dict shape `{verdict, score, fwd_return}` consumed by all metric functions, `BacktestRecord(ticker, as_of, verdict, score, fwd_returns: dict[int, float|None])`, `run_backtest(universe, dates, horizons, run_verdict, price_history)`, `metrics_for_horizon(records, horizon)`, `render_backtest_markdown/json(records, horizons)`, `build_backtest_verdict(cfg, client)`, `hit_at_k(retrieve_fn, cases, k)` are consistent across tasks and align with existing types (`Verdict`, `PriceProvider`, `OllamaClient`, `Orchestrator`, `ChromaVectorStore`/`NewsStore`).
