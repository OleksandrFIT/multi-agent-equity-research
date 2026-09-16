# A2 — Richer point-in-time evidence — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`). Tasks tagged **[CONTROLLER-LIVE]** need Ollama/SEC and a long run — the controller runs them.

**Goal:** Robust, richer historical fundamentals (margins, FCF, current ratio) + a `business` filing section, so agents get more point-in-time signal — and it stays measurable in the backtest. Also fixes `company_facts` crashes (KO/XOM/GOOGL).

**Architecture:** Defensive numeric extraction + more fields in `EdgarProvider.company_facts`; new margin/ratio metrics in `compute_fundamental_metrics`; prompt conventions for them; a third filing section; gauges for the new metrics. Then re-measure via the A3 backtest.

**Two repos / two branches:**
- **Part A** — backend `/Users/oleksandr/Documents/LLM/Multi-Agent Equity`, branch `feature/a2-evidence`.
- **Part B** — client `/Users/oleksandr/Documents/LLM/Multi-Agent Equity Client`, branch `feature/a2-evidence-ui`.

**Attribution:** commit ONLY as the repo author, NO `Co-Authored-By`. Client repo: no Claude/AI references.

---

## Part A — Backend

### Task A1: Robust + richer `company_facts`

**Files:**
- Modify: `equity_research/data/edgar.py`
- Modify: `tests/test_edgar_asof.py`

- [ ] **Step 1: Append a `_num` test to `tests/test_edgar_asof.py`**

```python
def test_num_is_defensive():
    import math

    from equity_research.data.edgar import _num

    assert _num({"a": 5}, "a") == 5.0
    assert math.isnan(_num({"a": None}, "a"))
    assert math.isnan(_num({}, "missing"))
    assert math.isnan(_num({"a": "x"}, "a"))
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_edgar_asof.py -v -k num_is_defensive` (import error).

- [ ] **Step 3: Update `equity_research/data/edgar.py`**

Add a module-level helper (near the top, after imports; `import math` if missing):

```python
def _num(m: dict, key: str) -> float:
    """Numeric field or NaN — never raises on missing/None/non-numeric values."""
    try:
        f = float(m.get(key))
    except (TypeError, ValueError):
        return float("nan")
    return f if math.isfinite(f) else float("nan")
```

Rewrite the tail of `company_facts` (from `m = financials.get_financial_metrics()` onward) to use `_num` everywhere and add the richer fields:

```python
        m = financials.get_financial_metrics()
        net_income = _num(m, "net_income")
        shares = _num(m, "shares_outstanding_diluted")
        eps_ttm = net_income / shares if math.isfinite(net_income) and math.isfinite(shares) and shares else float("nan")
        return {
            "net_income": net_income,
            "revenue": _num(m, "revenue"),
            "revenue_prev": _prior_revenue(financials),
            "equity": _num(m, "stockholders_equity"),
            "total_debt": _num(m, "total_liabilities"),
            "eps_ttm": eps_ttm,
            "operating_income": _num(m, "operating_income"),
            "free_cash_flow": _num(m, "free_cash_flow"),
            "current_ratio": _num(m, "current_ratio"),
        }
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_edgar_asof.py -v`.

- [ ] **Step 5: Commit**

```bash
git add equity_research/data/edgar.py tests/test_edgar_asof.py
git commit -m "feat(edgar): defensive numeric extraction + margin/FCF/liquidity facts"
```

---

### Task A2: New margin/ratio metrics

**Files:**
- Modify: `equity_research/analytics/fundamentals.py`
- Modify: `tests/test_fundamentals.py`

- [ ] **Step 1: Append tests to `tests/test_fundamentals.py`**

```python
def test_richer_margins_and_ratio():
    facts = dict(FACTS, operating_income=114000000000.0, free_cash_flow=99000000000.0, current_ratio=0.99)
    m = compute_fundamental_metrics(facts, price=196.0)
    assert round(m["operating_margin"], 4) == round(114000000000.0 / 383285000000.0, 4)
    assert round(m["net_margin"], 4) == round(96995000000.0 / 383285000000.0, 4)
    assert round(m["fcf_margin"], 4) == round(99000000000.0 / 383285000000.0, 4)
    assert m["current_ratio"] == 0.99


def test_richer_metrics_nan_when_missing():
    m = compute_fundamental_metrics(FACTS, price=196.0)  # FACTS has no op_income/fcf/current_ratio
    assert math.isnan(m["operating_margin"])
    assert math.isnan(m["fcf_margin"])
    assert math.isnan(m["current_ratio"])
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_fundamentals.py -v -k "richer"` (KeyError).

- [ ] **Step 3: Update `equity_research/analytics/fundamentals.py`** — add the new metrics to the returned dict (keep the existing four unchanged):

```python
def compute_fundamental_metrics(facts: dict[str, float], price: float) -> dict[str, float]:
    eps = facts.get("eps_ttm", 0.0)
    rev = facts.get("revenue", 0.0)
    rev_prev = facts.get("revenue_prev", 0.0)
    return {
        "pe": _safe_div(price, eps),
        "roe": _safe_div(facts.get("net_income", 0.0), facts.get("equity", 0.0)),
        "debt_to_equity": _safe_div(facts.get("total_debt", 0.0), facts.get("equity", 0.0)),
        "revenue_growth": _safe_div(rev - rev_prev, rev_prev) if rev_prev else math.nan,
        "operating_margin": _safe_div(facts.get("operating_income", math.nan), rev),
        "net_margin": _safe_div(facts.get("net_income", 0.0), rev),
        "fcf_margin": _safe_div(facts.get("free_cash_flow", math.nan), rev),
        "current_ratio": facts.get("current_ratio", math.nan),
    }
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_fundamentals.py tests/test_golden_metrics.py -v` (existing golden/edge tests still pass — they check only their expected keys).

- [ ] **Step 5: Commit**

```bash
git add equity_research/analytics/fundamentals.py tests/test_fundamentals.py
git commit -m "feat(analytics): operating/net/FCF margins + current ratio"
```

---

### Task A3: Prompt conventions for the new metrics

**Files:**
- Modify: `equity_research/agents/prompts.py`
- Modify: `tests/test_prompts.py`

- [ ] **Step 1: Append a test to `tests/test_prompts.py`**

```python
def test_prompt_formats_new_fundamental_metrics():
    e = Evidence(ticker="AAPL", as_of=date(2026, 9, 15),
                 metrics={"operating_margin": 0.30, "net_margin": 0.25, "fcf_margin": 0.26, "current_ratio": 0.99})
    prompt = build_judge_prompt("fundamentals", e)
    assert "Operating margin" in prompt and "30.0%" in prompt
    assert "Net margin" in prompt
    assert "Current ratio" in prompt and "0.99x" in prompt
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_prompts.py -v -k new_fundamental` (metrics rendered as bare `- key: value`).

- [ ] **Step 3: Add specs to `_METRIC_SPECS` in `equity_research/agents/prompts.py`**

```python
    "operating_margin": ("Operating margin", lambda v: f"{v:.1%}",
                         "operating income as a % of revenue; higher = more profitable operations"),
    "net_margin": ("Net margin", lambda v: f"{v:.1%}",
                   "net income as a % of revenue; higher = more profitable"),
    "fcf_margin": ("Free-cash-flow margin", lambda v: f"{v:.1%}",
                   "free cash flow as a % of revenue; higher = stronger cash generation"),
    "current_ratio": ("Current ratio", lambda v: f"{v:.2f}x",
                      "current assets over current liabilities; >1 = liquid, <1 = tighter liquidity"),
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_prompts.py -v`.

- [ ] **Step 5: Commit**

```bash
git add equity_research/agents/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): conventions for margins and current ratio"
```

---

### Task A4: Add the `business` filing section

**Files:**
- Modify: `equity_research/rag/filings.py`
- Modify: `tests/integration/test_filings_live.py`

- [ ] **Step 1: Add `business` to `fetch_filing_sections`** in `equity_research/rag/filings.py` — extend the loop:

```python
    for attr, name in [("risk_factors", "risk_factors"), ("management_discussion", "mda"), ("business", "business")]:
```

Update the docstring's section list to mention Item 1 (Business).

- [ ] **Step 2: Update the integration test** `tests/integration/test_filings_live.py` — allow the new section:

```python
    assert {s.section for s in sections} <= {"risk_factors", "mda", "business"}
```

- [ ] **Step 3: Verify it still imports/collects** — `uv run pytest tests/integration/test_filings_live.py --collect-only -q` (the live assertion runs in Task A6).

- [ ] **Step 4: Commit**

```bash
git add equity_research/rag/filings.py tests/integration/test_filings_live.py
git commit -m "feat(rag): include the Business (Item 1) filing section"
```

- [ ] **Step 5: Full suite** — `uv run pytest -q` → all green.

Merge `feature/a2-evidence` into `master` via **superpowers:finishing-a-development-branch** before Part B.

---

## Part B — Frontend

### Task B1: Gauges for the new metrics

**Files:**
- Modify: `src/components/MetricGauges.tsx`
- Modify: `src/components/MetricGauges.test.tsx`

- [ ] **Step 1: Append a test to `src/components/MetricGauges.test.tsx`**

```tsx
test('renders gauges for the new margin/ratio metrics', () => {
  render(<MetricGauges metrics={{ operating_margin: 0.3, net_margin: 0.25, fcf_margin: 0.26, current_ratio: 0.99 }} />)
  expect(screen.getByText('Op margin')).toBeInTheDocument()
  expect(screen.getByText('Net margin')).toBeInTheDocument()
  expect(screen.getByText('Current ratio')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run to verify fail** — `npm test -- src/components/MetricGauges.test.tsx` (labels absent).

- [ ] **Step 3: Add specs to `SPECS` in `src/components/MetricGauges.tsx`**

```ts
  operating_margin: { label: 'Op margin', min: 0, max: 0.5, format: pct, tone: 'var(--bull)' },
  net_margin: { label: 'Net margin', min: 0, max: 0.5, format: pct, tone: 'var(--bull)' },
  fcf_margin: { label: 'FCF margin', min: 0, max: 0.5, format: pct, tone: 'var(--bull)' },
  current_ratio: { label: 'Current ratio', min: 0, max: 3, format: num },
```

(`pct` and `num` formatters already exist in the file.)

- [ ] **Step 4: Run to verify pass + full suite + build** — `npm test && npm run build`.

- [ ] **Step 5: Commit**

```bash
git add src/components/MetricGauges.tsx src/components/MetricGauges.test.tsx
git commit -m "feat(ui): gauges for margins and current ratio"
```

Merge `feature/a2-evidence-ui` into `master` via **superpowers:finishing-a-development-branch**.

---

## Part C — [CONTROLLER-LIVE]: Live checks + re-measure

- [ ] **Step 1: Robustness** — confirm `company_facts` no longer crashes on the previously-failing tickers:

```bash
uv run python -c "
from datetime import date
from equity_research.config import Config
from equity_research.data.edgar import EdgarProvider
p = EdgarProvider(user_agent=Config.load('config.yaml').edgar_user_agent)
for t in ['KO','XOM','GOOGL','AAPL']:
    f = p.company_facts(t, date(2024,6,14))
    print(t, {k: round(v,3) for k,v in f.items()})
"
```
Expected: all four return a dict (some fields may be NaN) — no exception.

- [ ] **Step 2: Filing sections** — `uv run pytest -m integration tests/integration/test_filings_live.py` → passes and includes `business`.

- [ ] **Step 3: Re-measure** — re-run the A3 backtest (10×6) and record IC 21/63; compare to the post-Q3 baseline (IC ≈ −0.077 / −0.047). Report honestly whether richer evidence moved it.

- [ ] **Step 4: Golden** — re-capture `tests/fixtures/golden/verdicts_golden.json` (evidence changed → verdicts may shift), verify `uv run pytest -m integration tests/integration/test_verdicts_golden.py`, commit the golden.

- [ ] **Step 5: Live UI** — analyze AAPL, confirm the new gauges (Op/Net/FCF margin, Current ratio) render in the fundamentals card.

---

## Completion
Both branches merged (backend first). All commits authored as the repo owner, NO `Co-Authored-By`; client repo carries no Claude/AI references. Part C's measurement is reported; only the re-captured golden is committed.
