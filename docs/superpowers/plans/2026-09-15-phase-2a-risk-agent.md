# Phase 2A — Risk Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Risk agent that computes deterministic price-risk metrics (volatility, max drawdown, beta vs SPY) and gates the final `Verdict` confidence, without voting on buy/hold/sell direction.

**Architecture:** Risk is a confidence gate, not a directional agent. `analytics/risk.py` holds pure metric functions plus a deterministic `risk_level(metrics, cfg) -> [0,1]`. `RiskAgent` gathers metrics (ticker + benchmark prices) and returns an `AgentOpinion` whose `score` field carries the risk level (documented overload). The `Aggregator` computes direction only from `{fundamentals, technical, sentiment}` and multiplies base confidence by `(1 - gate_strength * risk_level)`, adding a caution note when risk is high.

**Tech Stack:** existing (pandas/numpy, pydantic, typer, ollama). No new dependencies.

**Scope:** Plan 2A only (Risk). RAG + Sentiment + grounding-check are Plan 2B. Spec: `docs/superpowers/specs/2026-09-15-phase-2-risk-rag-sentiment-design.md`.

## File Structure

- Create `equity_research/analytics/risk.py` — pure: `annualized_volatility`, `max_drawdown`, `beta`, `risk_level`.
- Create `equity_research/agents/risk.py` — `RiskAgent` (Agent protocol).
- Modify `equity_research/config.py` — add `benchmark` and `risk` fields.
- Modify `config.yaml` — add `benchmark` + `risk` block.
- Modify `equity_research/orchestration/aggregator.py` — directional set, risk gate, `Verdict.caution`.
- Modify `equity_research/reporting/report.py` — render caution line.
- Modify `equity_research/cli.py` — wire `RiskAgent` into `analyze_ticker`.
- Tests mirror each.

Naming contract (do not rename): `annualized_volatility(close)`, `max_drawdown(close)`, `beta(ticker_close, market_close)`, `risk_level(metrics: dict, cfg: dict) -> float`, `RiskAgent(prices, client, benchmark, risk_cfg)`, `Verdict.caution: str | None`, `DIRECTIONAL_AGENTS`.

---

## Task 1: Pure risk metrics

**Files:**
- Create: `equity_research/analytics/risk.py`
- Test: `tests/test_risk_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_risk_metrics.py`:
```python
import math

import pandas as pd

from equity_research.analytics.risk import annualized_volatility, beta, max_drawdown


def _s(values):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.Series([float(v) for v in values], index=idx)


def test_volatility_constant_series_is_zero():
    assert annualized_volatility(_s([100, 100, 100])) == 0.0


def test_volatility_known_value():
    # returns [+0.1, -0.1] -> std(ddof=1) = 0.1*sqrt(2); annualized *sqrt(252)
    vol = annualized_volatility(_s([100, 110, 99]))
    assert abs(vol - 0.1 * math.sqrt(2) * math.sqrt(252)) < 1e-9


def test_max_drawdown():
    assert abs(max_drawdown(_s([100, 120, 90, 110])) - 0.25) < 1e-9


def test_beta_two_x_market():
    market = _s([100, 110, 99])   # returns [+0.1, -0.1]
    ticker = _s([100, 120, 96])   # returns [+0.2, -0.2] = 2x market
    assert abs(beta(ticker, market) - 2.0) < 1e-9


def test_beta_nan_when_market_flat():
    market = _s([100, 100, 100])  # zero variance
    ticker = _s([100, 110, 99])
    assert math.isnan(beta(ticker, market))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_risk_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: equity_research.analytics.risk`.

- [ ] **Step 3: Write minimal implementation**

`equity_research/analytics/risk.py`:
```python
from __future__ import annotations

import math

import pandas as pd


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def annualized_volatility(close: pd.Series, periods_per_year: int = 252) -> float:
    r = _returns(close)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def max_drawdown(close: pd.Series) -> float:
    close = close.dropna()
    if close.empty:
        return float("nan")
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    return float(-drawdown.min())


def beta(ticker_close: pd.Series, market_close: pd.Series) -> float:
    joined = pd.concat([_returns(ticker_close), _returns(market_close)], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return float("nan")
    y = joined.iloc[:, 0]
    x = joined.iloc[:, 1]
    var = float(x.var(ddof=1))
    if var == 0:
        return float("nan")
    return float(y.cov(x) / var)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_risk_metrics.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/analytics/risk.py tests/test_risk_metrics.py
git commit -m "feat(risk): pure volatility, max drawdown, beta"
```

---

## Task 2: risk_level + Config extension

**Files:**
- Modify: `equity_research/analytics/risk.py`
- Modify: `equity_research/config.py`
- Modify: `config.yaml`
- Test: `tests/test_risk_level.py`

- [ ] **Step 1: Write the failing test**

`tests/test_risk_level.py`:
```python
from equity_research.analytics.risk import risk_level
from equity_research.config import Config

CFG = {"vol_low": 0.15, "vol_high": 0.60, "beta_high": 1.5, "drawdown_high": 0.40}


def test_low_everything_is_zero():
    m = {"volatility": 0.15, "beta": 0.0, "max_drawdown": 0.0}
    assert risk_level(m, CFG) == 0.0


def test_high_everything_is_one():
    m = {"volatility": 0.60, "beta": 1.5, "max_drawdown": 0.40}
    assert abs(risk_level(m, CFG) - 1.0) < 1e-9


def test_averages_available_components():
    # vol normalized 1.0, drawdown normalized 0.0, beta absent -> average of [1.0, 0.0] = 0.5
    m = {"volatility": 0.60, "max_drawdown": 0.0}
    assert abs(risk_level(m, CFG) - 0.5) < 1e-9


def test_empty_metrics_is_zero():
    assert risk_level({}, CFG) == 0.0


def test_config_has_risk_defaults():
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".cache",
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    assert cfg.benchmark == "SPY"
    assert cfg.risk["gate_strength"] == 0.5
    assert cfg.risk["vol_high"] == 0.60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_risk_level.py -v`
Expected: FAIL — `ImportError: cannot import name 'risk_level'` (and Config has no `benchmark`).

- [ ] **Step 3: Write minimal implementation**

Append to `equity_research/analytics/risk.py`:
```python
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def risk_level(metrics: dict[str, float], cfg: dict) -> float:
    components: list[float] = []
    vol = metrics.get("volatility")
    if vol is not None and not math.isnan(vol):
        lo, hi = cfg["vol_low"], cfg["vol_high"]
        components.append(_clamp01((vol - lo) / (hi - lo)))
    b = metrics.get("beta")
    if b is not None and not math.isnan(b):
        components.append(_clamp01(abs(b) / cfg["beta_high"]))
    dd = metrics.get("max_drawdown")
    if dd is not None and not math.isnan(dd):
        components.append(_clamp01(dd / cfg["drawdown_high"]))
    if not components:
        return 0.0
    return sum(components) / len(components)
```

In `equity_research/config.py`, add `Field` import and two new fields to `Config` (place after `weights`):
```python
from pydantic import BaseModel, Field
```
```python
    benchmark: str = "SPY"
    risk: dict[str, float] = Field(default_factory=lambda: {
        "gate_strength": 0.5,
        "caution_threshold": 0.6,
        "vol_low": 0.15,
        "vol_high": 0.60,
        "beta_high": 1.5,
        "drawdown_high": 0.40,
    })
```

In `config.yaml`, add at the end (top level):
```yaml
benchmark: SPY
risk:
  gate_strength: 0.5
  caution_threshold: 0.6
  vol_low: 0.15
  vol_high: 0.60
  beta_high: 1.5
  drawdown_high: 0.40
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_risk_level.py tests/test_config.py -v`
Expected: PASS (new risk_level tests + existing config tests still green — defaults mean existing Config(...) calls without `risk`/`benchmark` still work).

- [ ] **Step 5: Commit**

```bash
git add equity_research/analytics/risk.py equity_research/config.py config.yaml tests/test_risk_level.py
git commit -m "feat(risk): deterministic risk_level + config risk block"
```

---

## Task 3: RiskAgent

**Files:**
- Create: `equity_research/agents/risk.py`
- Test: `tests/test_risk_agent.py`

- [ ] **Step 1: Write the failing test**

`tests/test_risk_agent.py`:
```python
from datetime import date
from pathlib import Path

import pandas as pd

from equity_research.agents.risk import RiskAgent
from equity_research.data.prices import PriceProvider
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient

RISK_CFG = {"gate_strength": 0.5, "caution_threshold": 0.6,
            "vol_low": 0.15, "vol_high": 0.60, "beta_high": 1.5, "drawdown_high": 0.40}


def _prices(closes):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": [float(c) for c in closes]}, index=idx)


def _client(tmp_path):
    return OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                        chat_fn=lambda **k: {"message": {"content": "Risk narrative."}})


def _provider(closes):
    frame = _prices(closes)
    return PriceProvider(fetch_yfinance=lambda t: frame, fetch_stooq=lambda t: frame)


def test_gather_computes_risk_metrics(tmp_path):
    closes = [100, 110, 99, 120, 90, 130, 100, 140]
    agent = RiskAgent(prices=_provider(closes), client=_client(tmp_path), benchmark="SPY", risk_cfg=RISK_CFG)
    ev = agent.gather("AAPL", as_of=date(2025, 3, 1))
    assert "volatility" in ev.metrics
    assert "max_drawdown" in ev.metrics
    assert "beta" in ev.metrics


def test_judge_score_is_risk_level_and_neutral(tmp_path):
    closes = [100, 160, 80, 200, 60, 220, 70, 240]  # very volatile -> high risk
    agent = RiskAgent(prices=_provider(closes), client=_client(tmp_path), benchmark="SPY", risk_cfg=RISK_CFG)
    op = agent.judge(agent.gather("AAPL", as_of=date(2025, 3, 1)))
    assert op.agent == "risk"
    assert op.stance == "neutral"
    assert 0.0 <= op.score <= 1.0
    assert op.rationale == "Risk narrative."


def test_beta_nan_is_tolerated(tmp_path):
    # benchmark identical to ticker with zero-variance market slice -> beta nan, still no crash
    flat = _prices([100, 100, 100, 100])
    provider = PriceProvider(fetch_yfinance=lambda t: flat, fetch_stooq=lambda t: flat)
    agent = RiskAgent(prices=provider, client=_client(tmp_path), benchmark="SPY", risk_cfg=RISK_CFG)
    ev = agent.gather("AAPL", as_of=date(2025, 3, 1))
    op = agent.judge(ev)
    assert op.score == 0.0  # flat prices: vol 0, drawdown 0, beta nan -> level 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_risk_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: equity_research.agents.risk`.

- [ ] **Step 3: Write minimal implementation**

`equity_research/agents/risk.py`:
```python
from __future__ import annotations

from datetime import date

from equity_research.agents.base import AgentOpinion
from equity_research.analytics.risk import annualized_volatility, beta, max_drawdown, risk_level
from equity_research.data.models import Evidence
from equity_research.data.prices import PriceProvider
from equity_research.llm.ollama_client import OllamaClient


class RiskAgent:
    name = "risk"

    def __init__(self, prices: PriceProvider, client: OllamaClient, benchmark: str, risk_cfg: dict):
        self.prices = prices
        self.client = client
        self.benchmark = benchmark
        self.risk_cfg = risk_cfg

    def gather(self, ticker: str, as_of: date) -> Evidence:
        hist, note = self.prices.history(ticker, as_of)
        close = hist["Close"]
        metrics = {
            "volatility": annualized_volatility(close),
            "max_drawdown": max_drawdown(close),
        }
        try:
            bench, _ = self.prices.history(self.benchmark, as_of)
            metrics["beta"] = beta(close, bench["Close"])
        except Exception:
            metrics["beta"] = float("nan")
        notes = [note] if note else []
        return Evidence(ticker=ticker, as_of=as_of, metrics=metrics, notes=notes)

    def judge(self, evidence: Evidence) -> AgentOpinion:
        level = risk_level(evidence.metrics, self.risk_cfg)
        narrative = self.client.generate_text(self._prompt(evidence, level))
        key_facts = [f"{k}: {v:.4f}" for k, v in evidence.metrics.items()] + [f"risk_level: {level:.2f}"]
        return AgentOpinion(
            agent=self.name, stance="neutral", score=level, confidence=level,
            rationale=narrative, key_facts=key_facts,
        )

    def _prompt(self, evidence: Evidence, level: float) -> str:
        metrics = "\n".join(f"- {k}: {v:.4f}" for k, v in evidence.metrics.items())
        return (
            f"You are a risk analyst for {evidence.ticker}. Given these price-risk metrics, "
            f"write 2-3 sentences describing the risk profile. Do not give a buy/sell opinion.\n"
            f"{metrics}\n- computed risk level (0=low, 1=high): {level:.2f}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_risk_agent.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/agents/risk.py tests/test_risk_agent.py
git commit -m "feat(risk): RiskAgent (deterministic level + LLM narrative)"
```

---

## Task 4: Aggregator risk gate + Verdict.caution

**Files:**
- Modify: `equity_research/orchestration/aggregator.py`
- Test: `tests/test_aggregator.py` (add cases)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_aggregator.py`:
```python
def test_risk_gates_confidence_and_is_not_directional(tmp_path):
    ops = [
        AgentOpinion(agent="fundamentals", stance="bullish", score=0.8, confidence=1.0, rationale="r"),
        AgentOpinion(agent="risk", stance="neutral", score=0.8, confidence=0.8, rationale="risky"),
    ]
    agg = Aggregator(_cfg(), _client(tmp_path))
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=["technical", "sentiment"])
    # direction uses only fundamentals -> score 0.8, verdict buy
    assert v.verdict == "buy"
    assert abs(v.score - 0.8) < 1e-9
    # base confidence 1.0 gated: 1.0 * (1 - 0.5*0.8) = 0.6
    assert abs(v.confidence - 0.6) < 1e-9
    # risk is in the trace but not the directional set
    assert any(o.agent == "risk" for o in v.opinions)


def test_high_risk_adds_caution(tmp_path):
    ops = [
        AgentOpinion(agent="technical", stance="bullish", score=0.5, confidence=0.8, rationale="r"),
        AgentOpinion(agent="risk", stance="neutral", score=0.9, confidence=0.9, rationale="very risky"),
    ]
    agg = Aggregator(_cfg(), _client(tmp_path))
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=["fundamentals", "sentiment"])
    assert v.caution is not None
    assert "risk" in v.caution.lower()


def test_no_risk_agent_no_gate(tmp_path):
    ops = [AgentOpinion(agent="technical", stance="bullish", score=0.5, confidence=0.8, rationale="r")]
    agg = Aggregator(_cfg(), _client(tmp_path))
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=["fundamentals", "sentiment", "risk"])
    assert abs(v.confidence - 0.8) < 1e-9  # unchanged
    assert v.caution is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aggregator.py -v`
Expected: FAIL — `Verdict` has no `caution`; risk currently counted as a weighted directional agent.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/orchestration/aggregator.py`, add a module constant near the top (after imports):
```python
DIRECTIONAL_AGENTS = {"fundamentals", "technical", "sentiment"}
```

Add a `caution` field to `Verdict` (after `disclaimer`):
```python
    caution: str | None = None
```

Replace the entire `aggregate` method with:
```python
    def aggregate(self, ticker: str, as_of: date, opinions: list[AgentOpinion], skipped: list[str]) -> Verdict:
        skipped = list(skipped)
        risk_op = next((o for o in opinions if o.agent == "risk"), None)
        directional = [o for o in opinions if o.agent in DIRECTIONAL_AGENTS and o.agent in self.config.weights]
        directional_names = {o.agent for o in directional}
        skipped += [o.agent for o in opinions if o.agent != "risk" and o.agent not in directional_names]

        if not directional:
            return Verdict(
                ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                confidence=0.0, narrative="No agent produced an opinion; no data available.",
                opinions=[o for o in [risk_op] if o], skipped_agents=skipped,
            )

        weights = self.config.normalized_weights([o.agent for o in directional])
        score = sum(weights[o.agent] * o.score for o in directional)
        base_conf = sum(weights[o.agent] * o.confidence for o in directional)
        verdict = "buy" if score >= self.buy_th else "sell" if score <= self.sell_th else "hold"

        confidence = base_conf
        caution = None
        if risk_op is not None:
            rl = risk_op.score
            confidence = max(0.0, min(1.0, base_conf * (1 - self.config.risk["gate_strength"] * rl)))
            if rl >= self.config.risk["caution_threshold"]:
                caution = f"Elevated risk (level {rl:.0%}): {risk_op.rationale}"

        narrative = self.client.generate_text(self._narrative_prompt(ticker, verdict, score, directional))
        opinions_out = directional + ([risk_op] if risk_op is not None else [])
        return Verdict(
            ticker=ticker, as_of=as_of, verdict=verdict, score=score,
            confidence=confidence, narrative=narrative, opinions=opinions_out,
            caution=caution, skipped_agents=skipped,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aggregator.py -v`
Expected: PASS (all prior aggregator tests + 3 new). Note: `test_unweighted_agent_treated_as_skipped` still passes — `mystery` is not in `DIRECTIONAL_AGENTS`, so it lands in `skipped`; `test_all_agents_skipped_returns_hold` still passes via the empty-`directional` branch.

- [ ] **Step 5: Commit**

```bash
git add equity_research/orchestration/aggregator.py tests/test_aggregator.py
git commit -m "feat(orchestration): risk gate on confidence + Verdict.caution"
```

---

## Task 5: Report caution line

**Files:**
- Modify: `equity_research/reporting/report.py`
- Test: `tests/test_report.py` (add case)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_report.py`:
```python
def test_markdown_shows_caution_when_present():
    v = _verdict()
    v = v.model_copy(update={"caution": "Elevated risk (level 90%): very volatile"})
    md = render_markdown(v)
    assert "Elevated risk" in md


def test_markdown_omits_caution_when_absent():
    md = render_markdown(_verdict())  # _verdict() has no caution
    assert "Elevated risk" not in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL — caution not rendered.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/reporting/report.py`, inside `render_markdown`, immediately before the final disclaimer block (`lines += ["", "---", f"> {verdict.disclaimer}"]`), insert:
```python
    if verdict.caution:
        lines += ["", f"**Risk caution:** {verdict.caution}"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_report.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add equity_research/reporting/report.py tests/test_report.py
git commit -m "feat(reporting): show risk caution line when present"
```

---

## Task 6: Wire RiskAgent into the CLI

**Files:**
- Modify: `equity_research/cli.py`
- Test: `tests/test_cli.py` (existing test still passes; add a wiring assertion)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:
```python
def test_analyze_ticker_includes_risk_agent(monkeypatch, tmp_path):
    # analyze_ticker should build a RiskAgent among its agents. We stub Orchestrator
    # to capture the agents passed in, avoiding any network/LLM.
    import equity_research.cli as cli_module
    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator):
            captured["agents"] = agents

        def run(self, ticker, as_of):
            from equity_research.orchestration.aggregator import Verdict
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    # Point config at the repo config.yaml; ollama Client is constructed but never called
    # because FakeOrch.run does not invoke agents.
    from datetime import date
    cli_module.analyze_ticker("AAPL", date(2026, 9, 15), "config.yaml")
    names = [a.name for a in captured["agents"]]
    assert "risk" in names
    assert "fundamentals" in names and "technical" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_analyze_ticker_includes_risk_agent -v`
Expected: FAIL — no `risk` agent wired.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/cli.py`, add the import near the other agent imports:
```python
from equity_research.agents.risk import RiskAgent
```

In `analyze_ticker`, change the `agents` list to include the risk agent:
```python
    agents = [
        FundamentalsAgent(facts_source=edgar, prices=prices, client=client),
        TechnicalAgent(prices=prices, client=client),
        RiskAgent(prices=prices, client=client, benchmark=cfg.benchmark, risk_cfg=cfg.risk),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (existing CLI test + new wiring test).

- [ ] **Step 5: Commit**

```bash
git add equity_research/cli.py tests/test_cli.py
git commit -m "feat(cli): wire RiskAgent into analyze"
```

---

## Task 7: Full suite + live smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite**

Run: `uv run pytest -m "not integration" -p no:warnings -q`
Expected: all PASS (Phase 0/1 suite + all Phase 2A tests).

- [ ] **Step 2: Run the live smoke (requires Ollama + network)**

Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run pytest -m integration -q`
Expected: PASS. The live `analyze AAPL` now also runs the risk agent; if SPY fetch or edgar shapes drift, fix the thin wrapper, not the pure logic.

- [ ] **Step 3: Eyeball a real run**

Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run python -m equity_research.cli AAPL`
Expected: report includes a `risk` opinion in the trace and, if risk level is high, a `Risk caution:` line; confidence visibly gated relative to the base directional confidence.

- [ ] **Step 4: Commit (if any wrapper fix was needed)**

```bash
git add -A && git commit -m "fix(risk): live-integration adjustments"
```
(Skip if nothing changed.)

---

## Self-Review (completed during authoring)

- **Spec coverage (2A):** metrics (Task 1), `risk_level` + config block + benchmark (Task 2), RiskAgent gather/judge with beta-NaN tolerance (Task 3), aggregator directional-set + confidence gate + caution + `Verdict.caution` (Task 4), report caution line (Task 5), CLI wiring (Task 6), verification incl. live smoke (Task 7). Spec's "risk score = risk level (documented overload)" is realized in Task 3/4. "weights.risk kept but ignored for direction" is realized by `DIRECTIONAL_AGENTS` excluding risk (Task 4). Deferred to 2B (documented): RAG, Sentiment, grounding-check.
- **Placeholder scan:** none — every code step is complete.
- **Type consistency:** `annualized_volatility/max_drawdown/beta/risk_level`, `RiskAgent(prices, client, benchmark, risk_cfg)`, `Config.benchmark`/`Config.risk`, `DIRECTIONAL_AGENTS`, `Verdict.caution`, `Aggregator.aggregate(ticker, as_of, opinions, skipped)` are consistent across tasks and match the existing Phase 0/1 signatures (`AgentOpinion`, `Evidence`, `PriceProvider.history -> (df, note)`, `OllamaClient.generate_text`).
