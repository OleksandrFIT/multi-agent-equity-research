# Phase 3A — As-of EDGAR (point-in-time fundamentals) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit attribution:** commit ONLY as the repo author (OleksandrFIT). Do NOT add any `Co-Authored-By` trailer.

**Goal:** Make `EdgarProvider.company_facts` point-in-time by accepting an `as_of` date and using only the most recent 10-K/10-Q filed on or before that date, so the backtest has no look-ahead in fundamentals.

**Architecture:** A pure `select_filing_asof(filings, as_of)` picks the newest eligible filing (unit-tested with fake filings); the thin `EdgarProvider.company_facts(ticker, as_of)` wrapper uses it and is verified live. `FactsSource.company_facts` gains the `as_of` parameter and `FundamentalsAgent.gather` threads it through. Live behavior is unchanged (`as_of = today` selects the latest filing).

**Tech Stack:** existing (edgartools, pydantic, pytest). No new deps.

**Scope:** Plan 3A only. Backtest engine + metrics + hit@k are Plan 3B. Spec: `docs/superpowers/specs/2026-09-15-phase-3-eval-backtest-design.md`.

## File Structure

- `equity_research/data/edgar.py` — add `select_filing_asof` + `NoFilingError`; change `FactsSource` and `EdgarProvider.company_facts` to take `as_of`.
- `equity_research/agents/fundamentals.py` — pass `as_of` into `company_facts`.
- `tests/test_edgar_asof.py` — pure filing-selection tests.
- `tests/test_agents.py` — update the fundamentals stub to the new signature.
- `tests/integration/test_edgar_asof_live.py` — live as-of check (marked).

Naming contract: `select_filing_asof(filings, as_of) -> filing`, `NoFilingError`, `FactsSource.company_facts(ticker, as_of)`.

---

## Task 1: Pure filing-as-of selection

**Files:**
- Modify: `equity_research/data/edgar.py`
- Test: `tests/test_edgar_asof.py`

- [ ] **Step 1: Write the failing test**

`tests/test_edgar_asof.py`:
```python
from datetime import date

import pytest

from equity_research.data.edgar import NoFilingError, select_filing_asof


class _Filing:
    def __init__(self, filing_date):
        self.filing_date = filing_date


def test_selects_latest_on_or_before_as_of():
    filings = [_Filing(date(2024, 2, 1)), _Filing(date(2024, 5, 1)), _Filing(date(2024, 8, 1))]
    chosen = select_filing_asof(filings, date(2024, 6, 15))
    assert chosen.filing_date == date(2024, 5, 1)


def test_includes_filing_on_exact_as_of():
    filings = [_Filing(date(2024, 5, 1)), _Filing(date(2024, 6, 15))]
    assert select_filing_asof(filings, date(2024, 6, 15)).filing_date == date(2024, 6, 15)


def test_raises_when_no_filing_before_as_of():
    filings = [_Filing(date(2024, 5, 1))]
    with pytest.raises(NoFilingError):
        select_filing_asof(filings, date(2024, 1, 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_edgar_asof.py -v`
Expected: FAIL — `select_filing_asof` / `NoFilingError` not defined.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/data/edgar.py`, add at the top (after the existing imports; `re` is already imported):
```python
class NoFilingError(ValueError):
    pass


def select_filing_asof(filings, as_of):
    """Return the most recent filing with filing_date <= as_of, or raise NoFilingError."""
    eligible = [f for f in filings if f.filing_date <= as_of]
    if not eligible:
        raise NoFilingError(f"no filing on or before {as_of}")
    return max(eligible, key=lambda f: f.filing_date)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_edgar_asof.py -v`
Expected: PASS (3 tests). Run full suite `uv run pytest -m "not integration" -p no:warnings -q`, confirm green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/data/edgar.py tests/test_edgar_asof.py
git commit -m "feat(edgar): pure select_filing_asof (point-in-time filing choice)"
```

---

## Task 2: Thread as_of through FactsSource → EdgarProvider → FundamentalsAgent

**Files:**
- Modify: `equity_research/data/edgar.py`
- Modify: `equity_research/agents/fundamentals.py`
- Modify: `tests/test_agents.py`
- Test: existing `tests/test_fundamentals.py` still passes (pure metric function unchanged).

- [ ] **Step 1: Update the failing test**

In `tests/test_agents.py`, the fundamentals test stubs `facts_source` with a one-arg `company_facts`. Update that stub to the new two-arg signature. Change:
```python
        facts_source=type("F", (), {"company_facts": staticmethod(lambda t: facts)})(),
```
to:
```python
        facts_source=type("F", (), {"company_facts": staticmethod(lambda t, as_of: facts)})(),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agents.py::test_fundamentals_agent_uses_facts_and_price -v`
Expected: FAIL — `FundamentalsAgent.gather` still calls `company_facts(ticker)` (one arg) while the stub now requires two; or (after the code change) the old stub signature mismatches. Either way this test drives the signature change.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/data/edgar.py`, change the `FactsSource` protocol:
```python
class FactsSource(Protocol):
    def company_facts(self, ticker: str, as_of: date) -> dict[str, float]: ...
```
(`date` is already imported.) Then change `EdgarProvider.company_facts` to take `as_of` and select the point-in-time filing. Replace the method body:
```python
    def company_facts(self, ticker: str, as_of: date) -> dict[str, float]:
        from edgar import Company, set_identity

        set_identity(self.user_agent)
        company = Company(ticker)
        filings = company.get_filings(form=["10-K", "10-Q"])
        filing = select_filing_asof(list(filings), as_of)
        financials = filing.obj().financials
        m = financials.get_financial_metrics()

        net_income = float(m["net_income"])
        diluted_shares = float(m["shares_outstanding_diluted"])
        eps_ttm = net_income / diluted_shares if diluted_shares else 0.0

        return {
            "net_income": net_income,
            "revenue": float(m["revenue"]),
            "revenue_prev": _prior_revenue(financials),
            "equity": float(m["stockholders_equity"]),
            "total_debt": float(m["total_liabilities"]),
            "eps_ttm": eps_ttm,
        }
```
NOTE (verify live in Task 3): `company.get_filings(form=[...])`, each filing's `.filing_date` (must be a `date`; if edgartools returns a string, convert in `select_filing_asof`'s caller or normalize here), and `filing.obj().financials` are edgartools-version-dependent. Adjust ONLY this method if the live API differs — the pure `select_filing_asof` and metric logic must not change.

In `equity_research/agents/fundamentals.py`, change `gather` to pass `as_of`:
```python
        facts = self.facts_source.company_facts(ticker, as_of)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agents.py tests/test_fundamentals.py -v`
Expected: PASS. `test_fundamentals.py` (pure `compute_fundamental_metrics`) is unaffected. Confirm `uv run python -c "import equity_research.data.edgar, equity_research.agents.fundamentals"` succeeds offline. Run full suite `uv run pytest -m "not integration" -p no:warnings -q`, confirm green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/data/edgar.py equity_research/agents/fundamentals.py tests/test_agents.py
git commit -m "feat(edgar): point-in-time company_facts(ticker, as_of) threaded through fundamentals"
```

---

## Task 3: Live verification (as-of correctness + live analyze unchanged)

**Files:**
- Create: `tests/integration/test_edgar_asof_live.py`

- [ ] **Step 1: Write the (marked) integration test**

`tests/integration/test_edgar_asof_live.py`:
```python
from datetime import date

import pytest


@pytest.mark.integration
def test_edgar_asof_returns_pit_facts():
    """Requires network (SEC). Run: uv run pytest -m integration."""
    from equity_research.data.edgar import EdgarProvider

    provider = EdgarProvider(user_agent="Equity Research test@example.com")
    facts_now = provider.company_facts("AAPL", date.today())
    facts_old = provider.company_facts("AAPL", date(2021, 6, 30))
    assert facts_now["revenue"] > 0
    assert facts_old["revenue"] > 0
    # point-in-time: an as_of in 2021 must not see the latest (much larger) revenue
    assert facts_old["revenue"] != facts_now["revenue"]
```

- [ ] **Step 2: Confirm it is deselected by default**

Run: `uv run pytest -m "not integration" -q`
Expected: this test is NOT run (deselected). Do not run `-m integration` until Step 3.

- [ ] **Step 3: Live verification**

Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run pytest -m integration -q`
Expected: PASS. If `company.get_filings(...)`, `filing.filing_date`, or `filing.obj().financials` differ from the wrapper's assumptions, fix ONLY `EdgarProvider.company_facts` (and, if `filing_date` is a string, normalize it to a `date` before `select_filing_asof`), then re-run. The pure `select_filing_asof` and `compute_fundamental_metrics` must not change.

Then confirm live `analyze` still works (as_of = today selects the latest filing):
Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run python -m equity_research.cli analyze AAPL`
Expected: report renders with a `fundamentals` opinion (not skipped); values match the pre-3A run.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_edgar_asof_live.py
git commit -m "test(edgar): live point-in-time as-of verification"
```
(Include any `EdgarProvider.company_facts` wrapper fix in this commit if one was needed.)

---

## Self-Review (completed during authoring)

- **Spec coverage (3A):** `FactsSource.company_facts(ticker, as_of)` + `EdgarProvider` filing-as-of selection (Tasks 1-2); pure `select_filing_asof` with NoFilingError fallback (Task 1) → fundamentals agent skipped when no filing (existing orchestrator try/except); `FundamentalsAgent.gather` threads `as_of` (Task 2); live behavior unchanged for `as_of=today` and live verification incl. two-date PIT check (Task 3). No-filing fallback (spec §2) is realized by `NoFilingError` propagating to the orchestrator.
- **Placeholder scan:** none. The thin `EdgarProvider.company_facts` body carries real code plus an explicit "verify live, adjust only this method" note (Phase 1 EdgarProvider pattern).
- **Type consistency:** `select_filing_asof(filings, as_of)`, `NoFilingError`, `FactsSource.company_facts(ticker, as_of)`, `FundamentalsAgent.gather` call site, and the returned facts dict keys (`net_income`/`revenue`/`revenue_prev`/`equity`/`total_debt`/`eps_ttm`) are consistent with the existing `compute_fundamental_metrics` consumer and `_prior_revenue` helper.
