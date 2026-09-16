# A3 — Meaningful backtest + per-record report — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`). Tasks tagged **[CONTROLLER-LIVE]** need a running Ollama and a long run — the controller runs them.

**Goal:** Per-record detail in the backtest report (ticker × date → verdict → forward returns), a cleaner/larger universe for a more meaningful IC, and a run that measures it.

**Architecture:** `eval/report.py` gains a `records` array + a markdown table. `config.yaml` gets a cleaner 10×6 universe/dates. Frontend `BacktestReportView` shows a per-record table. Then a background run measures IC.

**Tech Stack:** Python + pytest (backend); React + TS + Vitest (frontend).

**Two repos / two branches:**
- **Part A** — backend `/Users/oleksandr/Documents/LLM/Multi-Agent Equity`, branch `feature/a3-backtest`.
- **Part B** — client `/Users/oleksandr/Documents/LLM/Multi-Agent Equity Client`, branch `feature/a3-backtest-ui`.

**Attribution:** commit ONLY as the repo author, NO `Co-Authored-By`. Client repo: no Claude/AI references.

---

## Part A — Backend

### Task A1: `records` in the backtest report

**Files:**
- Modify: `equity_research/eval/report.py`
- Modify: `tests/test_backtest_report.py`

- [ ] **Step 1: Append failing tests to `tests/test_backtest_report.py`**

```python
def test_json_includes_per_record_rows():
    data = json.loads(render_backtest_json(_recs(), horizons=[21, 63]))
    assert len(data["records"]) == 2
    row = data["records"][0]
    assert row["ticker"] == "AAA" and row["verdict"] == "buy" and abs(row["score"] - 0.5) < 1e-9
    assert row["fwd_returns"]["21"] == 0.1 and row["fwd_returns"]["63"] == 0.2
    assert data["records"][1]["fwd_returns"]["63"] is None


def test_markdown_has_records_table():
    md = render_backtest_markdown(_recs(), horizons=[21, 63])
    assert "| Ticker |" in md
    assert "AAA" in md and "BBB" in md
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_backtest_report.py -v` (KeyError `records` / no table).

- [ ] **Step 3: Update `equity_research/eval/report.py`**

Add a record-rows helper and include it in JSON; add a markdown table.

```python
def _record_rows(records: list[BacktestRecord], horizons: list[int]) -> list[dict]:
    return [{"ticker": r.ticker, "as_of": str(r.as_of), "verdict": r.verdict, "score": r.score,
             "fwd_returns": {str(h): r.fwd_returns.get(h) for h in horizons}}
            for r in records]


def render_backtest_json(records: list[BacktestRecord], horizons: list[int]) -> str:
    return json.dumps({
        "n_records": len(records),
        "horizons": _horizon_dict(records, horizons),
        "records": _record_rows(records, horizons),
        "disclaimer": DISCLAIMER,
    }, indent=2)
```

Append the table to `render_backtest_markdown` (before the trailing `---`/disclaimer lines):

```python
    lines.append("## Records")
    lines.append("| Ticker | As of | Verdict | Score | " + " | ".join(f"fwd {h}d" for h in horizons) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(horizons))
    for r in records:
        fwds = " | ".join("n/a" if r.fwd_returns.get(h) is None else f"{r.fwd_returns[h]:+.2%}"
                          for h in horizons)
        lines.append(f"| {r.ticker} | {r.as_of} | {r.verdict} | {r.score:+.2f} | {fwds} |")
    lines.append("")
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_backtest_report.py -v` (all pass; existing tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add equity_research/eval/report.py tests/test_backtest_report.py
git commit -m "feat(eval): per-record rows in backtest report (json + markdown table)"
```

---

### Task A2: Cleaner/larger backtest universe

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Update the `backtest` block in `config.yaml`** — replace `universe` and `dates`:

```yaml
backtest:
  universe: [AAPL, MSFT, NVDA, META, GOOGL, AMZN, JNJ, PEP, CSCO, HD]
  dates: ["2023-06-15", "2023-09-15", "2023-12-15", "2024-03-15", "2024-06-14", "2024-09-13"]
  horizons: [21, 63]
  report_path: backtest_report.md
```

(Keep any other keys under `backtest` that already exist; only universe/dates change. Financials
excluded; tickers where `company_facts` fails simply skip the fundamentals agent — not a crash.)

- [ ] **Step 2: Sanity-load** — `uv run python -c "from equity_research.config import Config; c=Config.load('config.yaml'); print(len(c.backtest['universe']), len(c.backtest['dates']))"` → `10 6`.

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "config: cleaner 10x6 backtest universe for a more meaningful IC"
```

Merge `feature/a3-backtest` into `master` via **superpowers:finishing-a-development-branch** before Part B.

---

## Part B — Frontend

### Task B1: Per-record table in BacktestReportView

**Files:**
- Modify: `src/api/types.ts`
- Modify: `src/components/BacktestReport.tsx`
- Modify: `src/components/BacktestReport.test.tsx`

- [ ] **Step 1: Add types to `src/api/types.ts`**

```ts
export interface BacktestRecordRow {
  ticker: string
  as_of: string
  verdict: 'buy' | 'hold' | 'sell'
  score: number
  fwd_returns: Record<string, number | null>
}
```

And add `records?: BacktestRecordRow[]` to the `BacktestReport` interface (optional, so existing fixtures still typecheck).

- [ ] **Step 2: Update the test `src/components/BacktestReport.test.tsx`**

Add `records` to the fixture and assert the table renders. The file already mocks `./EquityCurve`. Add to the `report` object:

```tsx
  records: [
    { ticker: 'AAPL', as_of: '2024-03-15', verdict: 'buy', score: 0.42, fwd_returns: { '21': 0.05, '63': null } },
  ],
```

And a new test:

```tsx
test('renders the per-record table', () => {
  render(<BacktestReportView report={report} />)
  expect(screen.getByText('AAPL')).toBeInTheDocument()
  expect(screen.getByText('2024-03-15')).toBeInTheDocument()
})
```

- [ ] **Step 3: Run to verify fail** — `npm test -- src/components/BacktestReport.test.tsx` (no table yet).

- [ ] **Step 4: Update `src/components/BacktestReport.tsx`** — add the table after the horizons map. Add imports and a records block:

```tsx
import { verdictTone, toneVar } from '../ui/tokens'
```

Before the final disclaimer `<div>`, insert:

```tsx
      {report.records && report.records.length > 0 && (
        <Card title="Per-record detail">
          <div className="overflow-x-auto">
            <table className="w-full text-xs tnum">
              <thead>
                <tr style={{ color: 'var(--text-mut)' }}>
                  <th className="py-1 text-left">Ticker</th>
                  <th className="text-left">As of</th>
                  <th className="text-right">Verdict</th>
                  <th className="text-right">Score</th>
                  {Object.keys(report.horizons).map((h) => <th key={h} className="text-right">fwd {h}d</th>)}
                </tr>
              </thead>
              <tbody>
                {report.records.map((r, i) => (
                  <tr key={i} style={{ borderTop: '1px solid var(--border-soft)' }}>
                    <td className="py-1 font-semibold" style={{ color: 'var(--text)' }}>{r.ticker}</td>
                    <td style={{ color: 'var(--text-dim)' }}>{r.as_of}</td>
                    <td className="text-right uppercase" style={{ color: toneVar[verdictTone[r.verdict] ?? 'neutral'] }}>{r.verdict}</td>
                    <td className="text-right" style={{ color: 'var(--text)' }}>{r.score >= 0 ? '+' : ''}{r.score.toFixed(2)}</td>
                    {Object.keys(report.horizons).map((h) => {
                      const v = r.fwd_returns[h]
                      return (
                        <td key={h} className="text-right"
                          style={{ color: v == null ? 'var(--text-mut)' : v >= 0 ? 'var(--bull)' : 'var(--bear)' }}>
                          {v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
```

- [ ] **Step 5: Run to verify pass** — `npm test -- src/components/BacktestReport.test.tsx`, then full `npm test && npm run build`.

- [ ] **Step 6: Commit**

```bash
git add src/api/types.ts src/components/BacktestReport.tsx src/components/BacktestReport.test.tsx
git commit -m "feat(ui): per-record table in backtest report"
```

Merge `feature/a3-backtest-ui` into `master` via **superpowers:finishing-a-development-branch**.

---

## Part C — [CONTROLLER-LIVE]: Run the expanded backtest and measure

- [ ] **Step 1:** Run the expanded backtest (background; this is long — ~2-3 hours on 7B + PM):

```bash
uv run python -c "
from equity_research.cli import build_backtest_verdict, _full_close
from equity_research.config import Config
from equity_research.eval.backtest import run_backtest
from equity_research.eval.report import render_backtest_markdown
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient
from ollama import Client
from datetime import datetime
cfg = Config.load('config.yaml')
chat = Client(host=cfg.ollama_host, timeout=cfg.net['ollama_timeout']).chat
client = OllamaClient(model=cfg.model, cache=DiskCache(cfg.cache_dir), seed=cfg.seed, temperature=cfg.temperature, chat_fn=chat)
dates=[datetime.strptime(d,'%Y-%m-%d').date() for d in cfg.backtest['dates']]
rv = build_backtest_verdict(cfg, client)
recs = run_backtest(cfg.backtest['universe'], dates, cfg.backtest['horizons'], rv, _full_close)
open(cfg.backtest['report_path'],'w').write(render_backtest_markdown(recs, cfg.backtest['horizons']))
print('n =', len(recs))
"
```

- [ ] **Step 2:** Read `backtest_report.md`; record IC (21/63), hit rate, mean return; compare to the
prior IC ≈ −0.16. Report honestly whether Q1+Q3 moved it (and that n≈60 is still only indicative).

- [ ] **Step 3:** Verify the frontend table live — start backend + `npm run dev`, run the Backtest
tab (or reuse the report), confirm the per-record table renders with tickers/dates/verdicts/returns.
(Stop servers after.) No commit (measurement + visual check only).

---

## Completion
Both branches merged (backend first). All commits authored as the repo owner, NO `Co-Authored-By`; client repo carries no Claude/AI references. Part C is a measurement — its finding is reported, not committed (except the generated `backtest_report.md` if you choose to keep it).
