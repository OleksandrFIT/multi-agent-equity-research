# UI v2 — Plan B: frontend dark redesign + charts

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the web client into a dark fintech terminal with price/RSI charts (lightweight-charts), metric gauges fed by structured agent metrics, and redesigned verdict/agent/backtest views.

**Architecture:** One small backend change surfaces structured `metrics` on each `AgentOpinion` (so the frontend can draw exact gauges instead of parsing text). The frontend gets a dark design-system layer (`src/ui/`) of presentational primitives, thin lightweight-charts wrapper components, and rebuilt pages. The API client / SSE layer is unchanged except a new `getPrices` call. No agent/eval logic changes.

**Tech Stack:** Backend: Python + pydantic. Frontend: React 19 + TS + Tailwind v4 + lightweight-charts v4 + Vitest + Testing Library.

**Two repos / two branches:**
- **Part 0** runs in the backend repo `/Users/oleksandr/Documents/LLM/Multi-Agent Equity` on branch `feature/agent-metrics`.
- **Parts 1–9** run in the client repo `/Users/oleksandr/Documents/LLM/Multi-Agent Equity Client` on branch `feature/ui-v2-redesign`.

**Attribution (BOTH repos):** commit ONLY as the repo's configured author, NO `Co-Authored-By` trailer. In the **client repo**: no mention of Claude/AI/assistant anywhere in code, comments, commit messages, or docs — it must read as hand-written.

**Metric keys the frontend can rely on** (from backend agents):
- fundamentals: `pe`, `roe`, `debt_to_equity`, `revenue_growth`
- risk: `volatility`, `max_drawdown`, `beta`
- technical: `rsi14`, `sma50`, `sma200`, `trend_pct`

---

## Part 0 — Backend: structured metrics on AgentOpinion

### Task 0.1: `AgentOpinion.metrics` + orchestrator wiring

Surface each agent's gathered `Evidence.metrics` onto its opinion, dropping non-finite values so JSON stays valid.

**Files:**
- Modify: `equity_research/agents/base.py`
- Modify: `equity_research/orchestration/orchestrator.py`
- Test: `tests/test_orchestrator.py` (append)

- [ ] **Step 1: Write the failing test (append to `tests/test_orchestrator.py`)**

```python
def test_orchestrator_surfaces_agent_metrics_and_drops_nan():
    import math
    from datetime import date

    from equity_research.agents.base import AgentOpinion
    from equity_research.data.models import Evidence
    from equity_research.orchestration.aggregator import Aggregator, Verdict
    from equity_research.orchestration.orchestrator import Orchestrator

    class StubAgent:
        name = "fundamentals"
        def gather(self, ticker, as_of):
            return Evidence(ticker=ticker, as_of=as_of,
                            metrics={"pe": 20.0, "revenue_growth": math.nan})
        def judge(self, evidence):
            return AgentOpinion(agent="fundamentals", stance="bullish", score=0.5,
                                confidence=0.8, rationale="r")

    captured = {}
    def on_event(ev):
        if "opinion" in ev:
            captured[ev["agent"]] = ev["opinion"]

    class StubAgg:
        def aggregate(self, ticker, as_of, opinions, skipped, skip_reasons):
            return Verdict(ticker=ticker, as_of=as_of, verdict="buy", score=0.5,
                           confidence=0.8, narrative="n", opinions=opinions)

    orch = Orchestrator([StubAgent()], StubAgg())
    verdict = orch.run("AAPL", date(2026, 9, 15), on_event=on_event)
    op = captured["fundamentals"]
    assert op.metrics == {"pe": 20.0}          # nan dropped
    assert verdict.opinions[0].metrics == {"pe": 20.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator.py::test_orchestrator_surfaces_agent_metrics_and_drops_nan -v`
Expected: FAIL with `AttributeError: 'AgentOpinion' object has no attribute 'metrics'` (or assertion on `{}`).

- [ ] **Step 3a: Add the field in `equity_research/agents/base.py`**

Add to `AgentOpinion` (after `dropped_facts`):

```python
    metrics: dict[str, float] = Field(default_factory=dict)
```

- [ ] **Step 3b: Wire it in `equity_research/orchestration/orchestrator.py`**

Add `import math` at the top, and after `opinion = agent.judge(evidence)`:

```python
                opinion.metrics = {
                    k: float(v) for k, v in evidence.metrics.items()
                    if isinstance(v, (int, float)) and math.isfinite(v)
                }
```

(Place it before `opinions.append(opinion)`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: PASS (all orchestrator tests).

- [ ] **Step 5: Run full suite (no regressions)**

Run: `uv run pytest -q`
Expected: all pass. In particular confirm `/api/analyze` SSE still serializes: `uv run pytest tests/api -q`.

- [ ] **Step 6: Commit**

```bash
git add equity_research/agents/base.py equity_research/orchestration/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(agents): surface structured metrics on AgentOpinion"
```

- [ ] **Step 7: Live sanity check (JSON validity with NaN dropped)**

```bash
uv run uvicorn api.main:app --port 8000 --log-level warning &
sleep 3
curl -s "http://localhost:8000/api/analyze?ticker=AAPL" | grep -o '"metrics":{[^}]*}' | head
kill %1
```
Expected: `metrics` objects appear on agent events with finite numbers only (no literal `NaN` in the stream).

Merge `feature/agent-metrics` into `master` via **superpowers:finishing-a-development-branch** before starting the frontend (frontend types depend on this shape). Then in the frontend, the backend must be running for live checks.

---

## Part 1 — Frontend: dark design system

### Task 1.1: Dark theme tokens + base styles

**Files:**
- Modify: `src/index.css`

- [ ] **Step 1: Replace `src/index.css` contents**

```css
@import "tailwindcss";

:root {
  color-scheme: dark;
  --bg: #0a0f1c;
  --surface: #131a2a;
  --surface-2: #101827;
  --border: #263352;
  --border-soft: #1e293b;
  --text: #e2e8f0;
  --text-dim: #94a3b8;
  --text-mut: #64748b;
  --accent: #38bdf8;
  --bull: #34d399;
  --neutral: #fbbf24;
  --bear: #fb7185;
}

html, body, #root { min-height: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.tnum { font-variant-numeric: tabular-nums; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
@keyframes pulse-soft { 0%,100% { opacity: 1; } 50% { opacity: .45; } }
.animate-pulse-soft { animation: pulse-soft 1.2s ease-in-out infinite; }
@keyframes fade-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
.animate-fade-in { animation: fade-in .25s ease-out both; }
```

- [ ] **Step 2: Verify build compiles**

Run: `npm run build`
Expected: build succeeds (Tailwind v4 processes the file).

- [ ] **Step 3: Commit**

```bash
git add src/index.css
git commit -m "style: dark fintech theme tokens and base styles"
```

---

### Task 1.2: UI primitives (`src/ui/`)

Small presentational building blocks used across pages. All pure, all testable.

**Files:**
- Create: `src/ui/tokens.ts`
- Create: `src/ui/primitives.tsx`
- Create: `src/ui/primitives.test.tsx`

- [ ] **Step 1: Write the failing tests (`src/ui/primitives.test.tsx`)**

```tsx
import { render, screen } from '@testing-library/react'
import { Card, Pill, ScoreBar, Meter, Chip, StatTile } from './primitives'

test('Card renders children and title', () => {
  render(<Card title="Fundamentals"><span>body</span></Card>)
  expect(screen.getByText('Fundamentals')).toBeInTheDocument()
  expect(screen.getByText('body')).toBeInTheDocument()
})

test('Pill shows label and stance class', () => {
  const { container } = render(<Pill tone="bull">bullish</Pill>)
  expect(screen.getByText('bullish')).toBeInTheDocument()
  expect(container.querySelector('[data-tone="bull"]')).not.toBeNull()
})

test('ScoreBar places marker by score (-1..1)', () => {
  const { container } = render(<ScoreBar score={0.5} />)
  const marker = container.querySelector('[data-testid="score-marker"]') as HTMLElement
  expect(marker.style.left).toBe('75%')  // (0.5+1)/2*100
})

test('Meter width reflects fraction', () => {
  const { container } = render(<Meter value={0.8} />)
  const fill = container.querySelector('[data-testid="meter-fill"]') as HTMLElement
  expect(fill.style.width).toBe('80%')
})

test('StatTile shows label and value', () => {
  render(<StatTile label="IC" value="0.123" />)
  expect(screen.getByText('IC')).toBeInTheDocument()
  expect(screen.getByText('0.123')).toBeInTheDocument()
})

test('Chip renders text', () => {
  render(<Chip>RSI 67</Chip>)
  expect(screen.getByText('RSI 67')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- src/ui/primitives.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/ui/tokens.ts`**

```ts
export type Tone = 'bull' | 'neutral' | 'bear' | 'accent'

export const stanceTone: Record<string, Tone> = {
  bullish: 'bull', neutral: 'neutral', bearish: 'bear',
}
export const verdictTone: Record<string, Tone> = {
  buy: 'bull', hold: 'neutral', sell: 'bear',
}
export const toneVar: Record<Tone, string> = {
  bull: 'var(--bull)', neutral: 'var(--neutral)', bear: 'var(--bear)', accent: 'var(--accent)',
}
```

- [ ] **Step 4: Implement `src/ui/primitives.tsx`**

```tsx
import type { ReactNode } from 'react'
import { type Tone, toneVar } from './tokens'

export function Card({ title, accent, children }: { title?: ReactNode; accent?: Tone; children: ReactNode }) {
  return (
    <div
      className="rounded-xl border p-4"
      style={{
        background: 'var(--surface)',
        borderColor: 'var(--border)',
        borderLeft: accent ? `3px solid ${toneVar[accent]}` : undefined,
      }}
    >
      {title != null && <div className="mb-2 text-sm font-semibold text-[var(--text)]">{title}</div>}
      {children}
    </div>
  )
}

export function Pill({ tone, children }: { tone: Tone; children: ReactNode }) {
  return (
    <span
      data-tone={tone}
      className="rounded-md px-2 py-0.5 text-xs font-medium capitalize"
      style={{ color: toneVar[tone], background: 'color-mix(in srgb, ' + toneVar[tone] + ' 15%, transparent)' }}
    >
      {children}
    </span>
  )
}

export function ScoreBar({ score }: { score: number }) {
  const pct = ((Math.max(-1, Math.min(1, score)) + 1) / 2) * 100
  const tone: Tone = score > 0.05 ? 'bull' : score < -0.05 ? 'bear' : 'neutral'
  return (
    <div className="relative h-1.5 w-full rounded-full" style={{ background: 'var(--border-soft)' }}>
      <div className="absolute top-0 h-full w-px" style={{ left: '50%', background: 'var(--text-mut)' }} />
      <div
        data-testid="score-marker"
        className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{ left: `${pct}%`, background: toneVar[tone] }}
      />
    </div>
  )
}

export function Meter({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full" style={{ background: 'var(--border-soft)' }}>
      <div data-testid="meter-fill" className="h-full rounded-full" style={{ width: `${pct}%`, background: 'var(--accent)' }} />
    </div>
  )
}

export function Chip({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-md border px-2 py-0.5 text-xs tnum"
      style={{ borderColor: 'var(--border-soft)', color: 'var(--text-dim)' }}>
      {children}
    </span>
  )
}

export function StatTile({ label, value, tone }: { label: string; value: ReactNode; tone?: Tone }) {
  return (
    <div className="rounded-lg border p-3" style={{ background: 'var(--surface-2)', borderColor: 'var(--border-soft)' }}>
      <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-mut)' }}>{label}</div>
      <div className="mt-1 text-lg font-semibold tnum" style={{ color: tone ? toneVar[tone] : 'var(--text)' }}>{value}</div>
    </div>
  )
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npm test -- src/ui/primitives.test.tsx`
Expected: PASS (6 passed).

- [ ] **Step 6: Commit**

```bash
git add src/ui/tokens.ts src/ui/primitives.tsx src/ui/primitives.test.tsx
git commit -m "feat(ui): dark design-system primitives"
```

---

### Task 1.3: `Gauge` primitive (radial SVG)

**Files:**
- Create: `src/ui/Gauge.tsx`
- Create: `src/ui/Gauge.test.tsx`

- [ ] **Step 1: Write failing tests (`src/ui/Gauge.test.tsx`)**

```tsx
import { render, screen } from '@testing-library/react'
import { Gauge } from './Gauge'

test('Gauge shows label and formatted value', () => {
  render(<Gauge label="RSI" value={67} min={0} max={100} format={(v) => v.toFixed(0)} />)
  expect(screen.getByText('RSI')).toBeInTheDocument()
  expect(screen.getByText('67')).toBeInTheDocument()
})

test('Gauge clamps value into range without throwing', () => {
  const { container } = render(<Gauge label="Vol" value={5} min={0} max={1} format={(v) => v.toFixed(2)} />)
  expect(container.querySelector('svg')).not.toBeNull()
  expect(screen.getByText('5.00')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- src/ui/Gauge.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/ui/Gauge.tsx`**

```tsx
export function Gauge({
  label, value, min, max, format, tone = 'var(--accent)',
}: {
  label: string
  value: number
  min: number
  max: number
  format: (v: number) => string
  tone?: string
}) {
  const frac = Math.max(0, Math.min(1, (value - min) / (max - min || 1)))
  const r = 34
  const circ = Math.PI * r // half circle
  const dash = frac * circ
  return (
    <div className="flex flex-col items-center rounded-lg border p-3"
      style={{ background: 'var(--surface-2)', borderColor: 'var(--border-soft)' }}>
      <svg width="86" height="52" viewBox="0 0 86 52">
        <path d="M9 46 A34 34 0 0 1 77 46" fill="none" stroke="var(--border)" strokeWidth="7" strokeLinecap="round" />
        <path d="M9 46 A34 34 0 0 1 77 46" fill="none" stroke={tone} strokeWidth="7" strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`} />
      </svg>
      <div className="-mt-3 text-base font-semibold tnum" style={{ color: 'var(--text)' }}>{format(value)}</div>
      <div className="mt-0.5 text-xs uppercase tracking-wide" style={{ color: 'var(--text-mut)' }}>{label}</div>
    </div>
  )
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npm test -- src/ui/Gauge.test.tsx`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ui/Gauge.tsx src/ui/Gauge.test.tsx
git commit -m "feat(ui): radial Gauge primitive"
```

---

## Part 2 — Prices data + charts

### Task 2.1: `getPrices` client + `Prices` type

**Files:**
- Modify: `src/api/types.ts`
- Modify: `src/api/client.ts`
- Create: `src/api/prices.test.ts`

- [ ] **Step 1: Write failing test (`src/api/prices.test.ts`)**

```ts
import { getPrices } from './client'

test('getPrices requests the endpoint and returns parsed json', async () => {
  const body = { ticker: 'AAPL', period: '6M',
    candles: [{ time: '2026-01-02', value: 100 }], sma50: [], sma200: [], rsi: [] }
  const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(JSON.stringify(body), { status: 200 }) as any)
  const out = await getPrices('aapl', '6M')
  expect(spy).toHaveBeenCalledWith('/api/prices?ticker=AAPL&period=6M')
  expect(out.candles[0].value).toBe(100)
  spy.mockRestore()
})
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- src/api/prices.test.ts`
Expected: FAIL (`getPrices` not exported).

- [ ] **Step 3: Add the `Prices` type to `src/api/types.ts`**

```ts
export interface PricePoint { time: string; value: number }
export interface Prices {
  ticker: string
  period: string
  candles: PricePoint[]
  sma50: PricePoint[]
  sma200: PricePoint[]
  rsi: PricePoint[]
}
export type Period = '1M' | '3M' | '6M' | '1Y'
```

- [ ] **Step 4: Add `getPrices` to `src/api/client.ts`**

```ts
import type { Period, Prices } from './types'

export function getPrices(ticker: string, period: Period): Promise<Prices> {
  return getJSON<Prices>(`/api/prices?ticker=${encodeURIComponent(ticker.toUpperCase())}&period=${period}`)
}
```

(Add the import at the top with the other imports; `getJSON` is already defined in this file.)

- [ ] **Step 5: Run to verify pass**

Run: `npm test -- src/api/prices.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/api/types.ts src/api/client.ts src/api/prices.test.ts
git commit -m "feat(api): getPrices client and Prices types"
```

---

### Task 2.2: Install lightweight-charts

**Files:**
- Modify: `package.json`, `package-lock.json`

- [ ] **Step 1: Install pinned version**

Run: `npm install lightweight-charts@4.2.3`
Expected: adds `"lightweight-charts": "^4.2.3"` to dependencies.

- [ ] **Step 2: Verify it imports**

Run: `npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "build: add lightweight-charts"
```

---

### Task 2.3: `TimeframeTabs` primitive

**Files:**
- Create: `src/components/TimeframeTabs.tsx`
- Create: `src/components/TimeframeTabs.test.tsx`

- [ ] **Step 1: Write failing test**

```tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { TimeframeTabs } from './TimeframeTabs'

test('renders periods and reports selection', async () => {
  const onChange = vi.fn()
  render(<TimeframeTabs value="6M" onChange={onChange} />)
  expect(screen.getByRole('button', { name: '6M' })).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: '1Y' }))
  expect(onChange).toHaveBeenCalledWith('1Y')
})
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- src/components/TimeframeTabs.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/components/TimeframeTabs.tsx`**

```tsx
import type { Period } from '../api/types'

const PERIODS: Period[] = ['1M', '3M', '6M', '1Y']

export function TimeframeTabs({ value, onChange }: { value: Period; onChange: (p: Period) => void }) {
  return (
    <div className="flex gap-1">
      {PERIODS.map((p) => (
        <button
          key={p}
          onClick={() => onChange(p)}
          className="rounded-md px-2.5 py-1 text-xs font-medium"
          style={{
            background: p === value ? 'color-mix(in srgb, var(--accent) 20%, transparent)' : 'transparent',
            color: p === value ? 'var(--accent)' : 'var(--text-dim)',
          }}
        >
          {p}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npm test -- src/components/TimeframeTabs.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/TimeframeTabs.tsx src/components/TimeframeTabs.test.tsx
git commit -m "feat(ui): TimeframeTabs"
```

---

### Task 2.4: `PriceChart` (lightweight-charts, price + SMA overlays)

Thin wrapper around lightweight-charts. jsdom can't render canvas, so the test mocks the library and asserts the component mounts and feeds data; real rendering is verified live.

**Files:**
- Create: `src/components/PriceChart.tsx`
- Create: `src/components/PriceChart.test.tsx`

- [ ] **Step 1: Write failing test (mock the lib)**

```tsx
import { render } from '@testing-library/react'
import { PriceChart } from './PriceChart'

const addAreaSeries = vi.fn(() => ({ setData: vi.fn() }))
const addLineSeries = vi.fn(() => ({ setData: vi.fn() }))
vi.mock('lightweight-charts', () => ({
  createChart: () => ({
    addAreaSeries, addLineSeries,
    timeScale: () => ({ fitContent: vi.fn() }),
    applyOptions: vi.fn(), resize: vi.fn(), remove: vi.fn(),
  }),
  ColorType: { Solid: 'solid' },
}))

const prices = {
  ticker: 'AAPL', period: '6M' as const,
  candles: [{ time: '2026-01-02', value: 100 }, { time: '2026-01-03', value: 102 }],
  sma50: [{ time: '2026-01-03', value: 101 }],
  sma200: [],
  rsi: [],
}

test('mounts and feeds price + sma50 series (sma200 empty -> not drawn)', () => {
  render(<PriceChart prices={prices} />)
  expect(addAreaSeries).toHaveBeenCalledTimes(1)   // price
  expect(addLineSeries).toHaveBeenCalledTimes(1)   // sma50 only; sma200 empty
})
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- src/components/PriceChart.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/components/PriceChart.tsx`**

```tsx
import { useEffect, useRef } from 'react'
import { ColorType, createChart, type IChartApi } from 'lightweight-charts'
import type { Prices } from '../api/types'

const CHART_OPTS = {
  height: 260,
  layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#94a3b8' },
  grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
  rightPriceScale: { borderColor: '#263352' },
  timeScale: { borderColor: '#263352' },
}

export function PriceChart({ prices }: { prices: Prices }) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, { ...CHART_OPTS, width: ref.current.clientWidth })
    chartRef.current = chart
    const price = chart.addAreaSeries({ lineColor: '#38bdf8', topColor: 'rgba(56,189,248,0.25)', bottomColor: 'rgba(56,189,248,0)', lineWidth: 2 })
    price.setData(prices.candles)
    if (prices.sma50.length) chart.addLineSeries({ color: '#fbbf24', lineWidth: 1 }).setData(prices.sma50)
    if (prices.sma200.length) chart.addLineSeries({ color: '#fb7185', lineWidth: 1 }).setData(prices.sma200)
    chart.timeScale().fitContent()
    const ro = new ResizeObserver(() => ref.current && chart.applyOptions({ width: ref.current.clientWidth }))
    ro.observe(ref.current)
    return () => { ro.disconnect(); chart.remove(); chartRef.current = null }
  }, [prices])

  return <div ref={ref} className="w-full" />
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npm test -- src/components/PriceChart.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/PriceChart.tsx src/components/PriceChart.test.tsx
git commit -m "feat(charts): PriceChart with SMA overlays"
```

---

### Task 2.5: `RsiChart` (RSI subchart with 30/70 bands)

**Files:**
- Create: `src/components/RsiChart.tsx`
- Create: `src/components/RsiChart.test.tsx`

- [ ] **Step 1: Write failing test (mock the lib)**

```tsx
import { render } from '@testing-library/react'
import { RsiChart } from './RsiChart'

const setData = vi.fn()
const createPriceLine = vi.fn()
const addLineSeries = vi.fn(() => ({ setData, createPriceLine }))
vi.mock('lightweight-charts', () => ({
  createChart: () => ({
    addLineSeries,
    timeScale: () => ({ fitContent: vi.fn() }),
    applyOptions: vi.fn(), remove: vi.fn(),
  }),
  ColorType: { Solid: 'solid' },
}))

test('mounts, feeds rsi, and draws 30/70 bands', () => {
  render(<RsiChart rsi={[{ time: '2026-01-02', value: 55 }]} />)
  expect(addLineSeries).toHaveBeenCalled()
  expect(setData).toHaveBeenCalled()
  expect(createPriceLine).toHaveBeenCalledTimes(2)   // 30 and 70
})

test('renders nothing for empty rsi', () => {
  const { container } = render(<RsiChart rsi={[]} />)
  expect(container.textContent).toContain('no data')
})
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- src/components/RsiChart.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/components/RsiChart.tsx`**

```tsx
import { useEffect, useRef } from 'react'
import { ColorType, createChart, type IChartApi } from 'lightweight-charts'
import type { PricePoint } from '../api/types'

export function RsiChart({ rsi }: { rsi: PricePoint[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!ref.current || rsi.length === 0) return
    const chart = createChart(ref.current, {
      width: ref.current.clientWidth, height: 110,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
      rightPriceScale: { borderColor: '#263352' },
      timeScale: { borderColor: '#263352' },
    })
    chartRef.current = chart
    const line = chart.addLineSeries({ color: '#38bdf8', lineWidth: 1 })
    line.setData(rsi)
    line.createPriceLine({ price: 70, color: '#fb7185', lineWidth: 1, lineStyle: 2 })
    line.createPriceLine({ price: 30, color: '#34d399', lineWidth: 1, lineStyle: 2 })
    chart.timeScale().fitContent()
    const ro = new ResizeObserver(() => ref.current && chart.applyOptions({ width: ref.current.clientWidth }))
    ro.observe(ref.current)
    return () => { ro.disconnect(); chart.remove(); chartRef.current = null }
  }, [rsi])

  if (rsi.length === 0) return <div className="text-xs" style={{ color: 'var(--text-mut)' }}>no data</div>
  return <div ref={ref} className="w-full" />
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npm test -- src/components/RsiChart.test.tsx`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/components/RsiChart.tsx src/components/RsiChart.test.tsx
git commit -m "feat(charts): RsiChart with 30/70 bands"
```

---

### Task 2.6: `PricePanel` (fetch + tabs + chart + rsi + states)

Owns period state, fetches `/api/prices`, and composes the charts. Test mocks `getPrices`.

**Files:**
- Create: `src/components/PricePanel.tsx`
- Create: `src/components/PricePanel.test.tsx`

- [ ] **Step 1: Write failing test**

```tsx
import { render, screen } from '@testing-library/react'
import * as client from '../api/client'
import { PricePanel } from './PricePanel'

vi.mock('./PriceChart', () => ({ PriceChart: () => <div>price-chart</div> }))
vi.mock('./RsiChart', () => ({ RsiChart: () => <div>rsi-chart</div> }))

test('fetches prices and renders charts', async () => {
  vi.spyOn(client, 'getPrices').mockResolvedValue({
    ticker: 'AAPL', period: '6M',
    candles: [{ time: '2026-01-02', value: 100 }], sma50: [], sma200: [], rsi: [],
  })
  render(<PricePanel ticker="AAPL" />)
  expect(await screen.findByText('price-chart')).toBeInTheDocument()
  expect(screen.getByText('rsi-chart')).toBeInTheDocument()
})
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- src/components/PricePanel.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/components/PricePanel.tsx`**

```tsx
import { useEffect, useState } from 'react'
import { getPrices } from '../api/client'
import type { Period, Prices } from '../api/types'
import { Card } from '../ui/primitives'
import { PriceChart } from './PriceChart'
import { RsiChart } from './RsiChart'
import { TimeframeTabs } from './TimeframeTabs'

export function PricePanel({ ticker }: { ticker: string }) {
  const [period, setPeriod] = useState<Period>('6M')
  const [prices, setPrices] = useState<Prices | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    setError(null)
    getPrices(ticker, period)
      .then((p) => { if (alive) setPrices(p) })
      .catch((e) => { if (alive) setError((e as Error).message) })
    return () => { alive = false }
  }, [ticker, period])

  return (
    <Card title={
      <div className="flex items-center justify-between">
        <span>{ticker} price</span>
        <TimeframeTabs value={period} onChange={setPeriod} />
      </div>
    }>
      {error && <div className="text-xs" style={{ color: 'var(--bear)' }}>chart unavailable — {error}</div>}
      {!error && !prices && <div className="h-[260px] animate-pulse-soft rounded-lg" style={{ background: 'var(--surface-2)' }} />}
      {!error && prices && prices.candles.length === 0 && <div className="text-xs" style={{ color: 'var(--text-mut)' }}>no data</div>}
      {!error && prices && prices.candles.length > 0 && (
        <div className="space-y-2">
          <PriceChart prices={prices} />
          <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-mut)' }}>RSI (14)</div>
          <RsiChart rsi={prices.rsi} />
        </div>
      )}
    </Card>
  )
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npm test -- src/components/PricePanel.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/components/PricePanel.tsx src/components/PricePanel.test.tsx
git commit -m "feat(charts): PricePanel (fetch + timeframe + rsi + states)"
```

---

## Part 3 — Metric gauges

### Task 3.1: `MetricGauges` (map known metric keys to gauges)

**Files:**
- Create: `src/components/MetricGauges.tsx`
- Create: `src/components/MetricGauges.test.tsx`

- [ ] **Step 1: Write failing test**

```tsx
import { render, screen } from '@testing-library/react'
import { MetricGauges } from './MetricGauges'

test('renders gauges for known metric keys only', () => {
  render(<MetricGauges metrics={{ pe: 24, roe: 0.3, volatility: 0.22, unknown_key: 9 }} />)
  expect(screen.getByText('P/E')).toBeInTheDocument()
  expect(screen.getByText('ROE')).toBeInTheDocument()
  expect(screen.getByText('Volatility')).toBeInTheDocument()
  expect(screen.queryByText(/unknown_key/i)).toBeNull()
})

test('renders nothing when no known metrics', () => {
  const { container } = render(<MetricGauges metrics={{ foo: 1 }} />)
  expect(container.firstChild).toBeNull()
})
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- src/components/MetricGauges.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/components/MetricGauges.tsx`**

```tsx
import { Gauge } from '../ui/Gauge'

interface Spec { label: string; min: number; max: number; format: (v: number) => string; tone?: string }

const pct = (v: number) => `${(v * 100).toFixed(0)}%`
const num = (v: number) => v.toFixed(1)

// Only metrics with a spec are drawn; everything else is left to key-facts chips.
const SPECS: Record<string, Spec> = {
  pe: { label: 'P/E', min: 0, max: 50, format: num },
  roe: { label: 'ROE', min: 0, max: 0.5, format: pct, tone: 'var(--bull)' },
  debt_to_equity: { label: 'D/E', min: 0, max: 3, format: num },
  revenue_growth: { label: 'Rev growth', min: -0.2, max: 0.5, format: pct },
  volatility: { label: 'Volatility', min: 0, max: 1, format: pct, tone: 'var(--neutral)' },
  max_drawdown: { label: 'Max drawdown', min: 0, max: 1, format: pct, tone: 'var(--bear)' },
  beta: { label: 'Beta', min: 0, max: 2.5, format: num },
  rsi14: { label: 'RSI', min: 0, max: 100, format: (v) => v.toFixed(0) },
  trend_pct: { label: 'Trend', min: -30, max: 30, format: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(0)}%` },
}

export function MetricGauges({ metrics }: { metrics: Record<string, number> }) {
  const items = Object.entries(metrics)
    .map(([k, v]) => [SPECS[k], v] as const)
    .filter(([spec]) => spec != null)
  if (items.length === 0) return null
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {items.map(([spec, v]) => (
        <Gauge key={spec!.label} label={spec!.label} value={v} min={spec!.min} max={spec!.max}
          format={spec!.format} tone={spec!.tone} />
      ))}
    </div>
  )
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npm test -- src/components/MetricGauges.test.tsx`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/components/MetricGauges.tsx src/components/MetricGauges.test.tsx
git commit -m "feat(ui): MetricGauges mapping structured metrics to gauges"
```

---

## Part 4 — Redesigned cards

### Task 4.1: Redesign `AgentCard`

Add `metrics` to the `AgentOpinion` type first, then restyle the card (accent border by stance, Pill, ScoreBar, Meter, key-facts Chips, gauges).

**Files:**
- Modify: `src/api/types.ts`
- Modify: `src/components/AgentCard.tsx`
- Modify: `src/components/AgentCard.test.tsx`

- [ ] **Step 1: Extend the type in `src/api/types.ts`**

Add to `AgentOpinion` (after `dropped_facts`). Optional so existing test fixtures that omit it still typecheck; the backend always sends it (possibly `{}`):

```ts
  metrics?: Record<string, number>
```

- [ ] **Step 2: Update the test (`src/components/AgentCard.test.tsx`)**

Replace file with:

```tsx
import { render, screen } from '@testing-library/react'
import { AgentCard } from './AgentCard'

test('renders an opinion agent with stance, rationale, key facts', () => {
  render(<AgentCard event={{ agent: 'technical', opinion: {
    agent: 'technical', stance: 'bullish', score: 0.7, confidence: 0.8,
    rationale: 'golden cross', key_facts: ['RSI 67'], dropped_facts: [], metrics: { rsi14: 67 } } }} />)
  expect(screen.getByText(/technical/i)).toBeInTheDocument()
  expect(screen.getByText(/bullish/i)).toBeInTheDocument()
  expect(screen.getByText(/golden cross/i)).toBeInTheDocument()
  expect(screen.getByText('RSI 67')).toBeInTheDocument()
})

test('renders a skipped agent', () => {
  render(<AgentCard event={{ agent: 'sentiment', skipped: true, reason: 'no news' }} />)
  expect(screen.getByText(/sentiment/i)).toBeInTheDocument()
  expect(screen.getByText(/no news/i)).toBeInTheDocument()
})
```

- [ ] **Step 3: Run to verify fail**

Run: `npm test -- src/components/AgentCard.test.tsx`
Expected: FAIL (type/rendering mismatch until rewrite).

- [ ] **Step 4: Rewrite `src/components/AgentCard.tsx`**

```tsx
import type { AgentEvent } from '../api/types'
import { stanceTone } from '../ui/tokens'
import { Card, Pill, ScoreBar, Meter, Chip } from '../ui/primitives'
import { MetricGauges } from './MetricGauges'

export function AgentCard({ event }: { event: AgentEvent }) {
  if ('skipped' in event) {
    return (
      <div className="rounded-xl border p-3 opacity-60"
        style={{ background: 'var(--surface-2)', borderColor: 'var(--border-soft)' }}>
        <div className="font-medium capitalize" style={{ color: 'var(--text-dim)' }}>{event.agent}</div>
        <div className="text-sm" style={{ color: 'var(--text-mut)' }}>skipped — {event.reason}</div>
      </div>
    )
  }
  const o = event.opinion
  const tone = stanceTone[o.stance] ?? 'neutral'
  return (
    <div className="animate-fade-in">
      <Card accent={tone} title={
        <div className="flex items-center justify-between">
          <span className="capitalize">{o.agent}</span>
          <Pill tone={tone}>{o.stance}</Pill>
        </div>
      }>
        <p className="text-sm" style={{ color: 'var(--text-dim)' }}>{o.rationale}</p>
        <div className="mt-3 grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-2 text-xs" style={{ color: 'var(--text-mut)' }}>
          <span>score</span><div className="flex items-center gap-2"><ScoreBar score={o.score} /><span className="tnum" style={{ color: 'var(--text)' }}>{o.score >= 0 ? '+' : ''}{o.score.toFixed(2)}</span></div>
          <span>confidence</span><div className="flex items-center gap-2"><Meter value={o.confidence} /><span className="tnum" style={{ color: 'var(--text)' }}>{Math.round(o.confidence * 100)}%</span></div>
        </div>
        {o.metrics && Object.keys(o.metrics).length > 0 && <div className="mt-3"><MetricGauges metrics={o.metrics} /></div>}
        {o.key_facts.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">{o.key_facts.map((f) => <Chip key={f}>{f}</Chip>)}</div>
        )}
      </Card>
    </div>
  )
}
```

- [ ] **Step 5: Run to verify pass**

Run: `npm test -- src/components/AgentCard.test.tsx`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/api/types.ts src/components/AgentCard.tsx src/components/AgentCard.test.tsx
git commit -m "feat(ui): redesign AgentCard (accent, gauges, chips)"
```

---

### Task 4.2: Redesign `VerdictCard`

**Files:**
- Modify: `src/components/VerdictCard.tsx`

The existing `src/components/VerdictCard.test.tsx` asserts only visible text (ticker `AAPL`, `/BUY/i`, `/Elevated risk/i`, `/not advice/i`) with `opinions: []`. The rewrite below still renders all of those, so **the test needs no changes** — just confirm it stays green.

- [ ] **Step 1: Rewrite `src/components/VerdictCard.tsx`**

```tsx
import type { Verdict } from '../api/types'
import { verdictTone, toneVar } from '../ui/tokens'
import { Meter } from '../ui/primitives'
import { AgentCard } from './AgentCard'

export function VerdictCard({ verdict }: { verdict: Verdict }) {
  const tone = verdictTone[verdict.verdict] ?? 'neutral'
  return (
    <div className="space-y-4 animate-fade-in">
      <div className="rounded-xl border p-5" style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-2xl font-bold" style={{ color: 'var(--text)' }}>{verdict.ticker}</h2>
          <span className="rounded-lg px-3 py-1 text-sm font-bold uppercase"
            style={{ color: toneVar[tone], background: `color-mix(in srgb, ${toneVar[tone]} 18%, transparent)` }}>
            {verdict.verdict}
          </span>
          <span className="tnum text-sm" style={{ color: 'var(--text-dim)' }}>
            score {verdict.score >= 0 ? '+' : ''}{verdict.score.toFixed(2)}
          </span>
        </div>
        <div className="mt-3 flex items-center gap-3 text-xs" style={{ color: 'var(--text-mut)' }}>
          <span>confidence</span>
          <div className="w-40"><Meter value={verdict.confidence} /></div>
          <span className="tnum" style={{ color: 'var(--text)' }}>{Math.round(verdict.confidence * 100)}%</span>
        </div>
        <p className="mt-4 border-l-2 pl-3 text-sm leading-relaxed"
          style={{ borderColor: 'var(--accent)', color: 'var(--text-dim)' }}>{verdict.narrative}</p>
        {verdict.caution && (
          <div className="mt-3 rounded-lg border p-2 text-sm"
            style={{ borderColor: 'var(--neutral)', color: 'var(--neutral)', background: 'color-mix(in srgb, var(--neutral) 10%, transparent)' }}>
            ⚠ {verdict.caution}
          </div>
        )}
      </div>
      <div className="space-y-3">
        {verdict.opinions.map((o) => <AgentCard key={o.agent} event={{ agent: o.agent, opinion: o }} />)}
      </div>
      {verdict.skipped_agents.length > 0 && (
        <div className="text-xs" style={{ color: 'var(--text-mut)' }}>
          Skipped: {verdict.skipped_agents.map((a) => `${a}${verdict.skip_reasons[a] ? ` (${verdict.skip_reasons[a]})` : ''}`).join(', ')}
        </div>
      )}
      <div className="border-t pt-2 text-xs" style={{ borderColor: 'var(--border-soft)', color: 'var(--text-mut)' }}>{verdict.disclaimer}</div>
    </div>
  )
}
```

- [ ] **Step 2: Run to verify pass**

Run: `npm test -- src/components/VerdictCard.test.tsx`
Expected: PASS (unchanged test).

- [ ] **Step 3: Commit**

```bash
git add src/components/VerdictCard.tsx
git commit -m "feat(ui): redesign VerdictCard"
```

---

## Part 5 — Shell, pages, streaming polish

### Task 5.1: App-shell + redesigned nav + HealthBanner + TickerInput

**Files:**
- Modify: `src/App.tsx`, `src/components/HealthBanner.tsx`, `src/components/TickerInput.tsx`

- [ ] **Step 1: Rewrite `src/App.tsx`**

```tsx
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { HealthBanner } from './components/HealthBanner'
import { Analyze } from './pages/Analyze'
import { Backtest } from './pages/Backtest'
import { Ingest } from './pages/Ingest'

const tabs = [{ to: '/', label: 'Analyze' }, { to: '/backtest', label: 'Backtest' }, { to: '/ingest', label: 'Ingest' }]

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen">
        <header className="sticky top-0 z-10 border-b" style={{ background: 'var(--surface-2)', borderColor: 'var(--border)' }}>
          <div className="mx-auto flex max-w-4xl flex-wrap items-center gap-4 px-4 py-3">
            <span className="text-sm font-bold tracking-wide" style={{ color: 'var(--accent)' }}>◆ Equity Terminal</span>
            <nav className="flex gap-1 text-sm">
              {tabs.map((t) => (
                <NavLink key={t.to} to={t.to} end={t.to === '/'}
                  className="rounded-md px-3 py-1"
                  style={({ isActive }) => ({
                    color: isActive ? 'var(--accent)' : 'var(--text-dim)',
                    borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                  })}>
                  {t.label}
                </NavLink>
              ))}
            </nav>
            <div className="ml-auto"><HealthBanner /></div>
          </div>
        </header>
        <main className="mx-auto max-w-4xl px-4 py-6">
          <Routes>
            <Route path="/" element={<Analyze />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/ingest" element={<Ingest />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
```

- [ ] **Step 2: Rewrite `src/components/HealthBanner.tsx` as an inline chip**

```tsx
import { useEffect, useState } from 'react'
import { getJSON } from '../api/client'
import type { Health } from '../api/types'

export function HealthBanner() {
  const [h, setH] = useState<Health | null>(null)
  useEffect(() => {
    getJSON<Health>('/api/health').then(setH).catch(() => setH({ ok: false, model: '', ollama_reachable: false }))
  }, [])
  const ok = h?.ollama_reachable ?? false
  const tone = ok ? 'var(--bull)' : 'var(--bear)'
  return (
    <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-dim)' }} title={h?.model || ''}>
      <span className="h-2 w-2 rounded-full" style={{ background: tone }} />
      {ok ? h?.model || 'online' : 'ollama offline'}
    </span>
  )
}
```

- [ ] **Step 3: Rewrite `src/components/TickerInput.tsx` (dark styling, same API)**

```tsx
import { useState } from 'react'

export function TickerInput({ onSubmit, disabled }: { onSubmit: (t: string) => void; disabled?: boolean }) {
  const [t, setT] = useState('')
  return (
    <form onSubmit={(e) => { e.preventDefault(); if (t.trim()) onSubmit(t.trim().toUpperCase()) }} className="flex gap-2">
      <input value={t} onChange={(e) => setT(e.target.value)} placeholder="Ticker (e.g. AAPL)"
        className="tnum rounded-lg border px-3 py-2 text-sm outline-none"
        style={{ background: 'var(--surface-2)', borderColor: 'var(--border)', color: 'var(--text)' }} />
      <button disabled={disabled}
        className="rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
        style={{ background: 'var(--accent)', color: '#04121f' }}>Go</button>
    </form>
  )
}
```

- [ ] **Step 4: Update `src/App.smoke.test.tsx`** (old text "Equity Research" is gone)

```tsx
import { render, screen } from '@testing-library/react'
import App from './App'

test('renders the app shell brand', () => {
  render(<App />)
  expect(screen.getByText(/equity terminal/i)).toBeInTheDocument()
})
```

- [ ] **Step 5: Run the full test suite**

Run: `npm test`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/App.tsx src/components/HealthBanner.tsx src/components/TickerInput.tsx src/App.smoke.test.tsx
git commit -m "feat(ui): dark app-shell, nav, health chip, ticker input"
```

---

### Task 5.2: Analyze page — progress-rail + price panel + streaming polish

**Files:**
- Create: `src/components/ProgressRail.tsx`
- Modify: `src/pages/Analyze.tsx`

The existing `src/pages/Analyze.test.tsx` only asserts the ticker input is present and never calls `run()` (so no EventSource, and `PricePanel` is not mounted because `ticker` starts `null`). The rewrite below keeps the ticker input, so **the test needs no changes** — confirm it stays green.

- [ ] **Step 1: Implement `src/components/ProgressRail.tsx`**

```tsx
const AGENTS = ['fundamentals', 'technical', 'sentiment', 'risk']

export function ProgressRail({ done, running }: { done: Set<string>; running: boolean }) {
  return (
    <div className="flex flex-wrap gap-2">
      {AGENTS.map((a) => {
        const isDone = done.has(a)
        const isRunning = running && !isDone
        return (
          <span key={a}
            className={`rounded-md border px-2.5 py-1 text-xs capitalize ${isRunning ? 'animate-pulse-soft' : ''}`}
            style={{
              borderColor: isDone ? 'var(--bull)' : 'var(--border-soft)',
              color: isDone ? 'var(--bull)' : 'var(--text-mut)',
              background: isDone ? 'color-mix(in srgb, var(--bull) 12%, transparent)' : 'transparent',
            }}>
            {isDone ? '✓ ' : ''}{a}
          </span>
        )
      })}
    </div>
  )
}
```

- [ ] **Step 2: Rewrite `src/pages/Analyze.tsx`**

```tsx
import { useMemo, useRef, useState } from 'react'
import { streamSSE } from '../api/client'
import type { AgentEvent, Verdict } from '../api/types'
import { AgentCard } from '../components/AgentCard'
import { VerdictCard } from '../components/VerdictCard'
import { TickerInput } from '../components/TickerInput'
import { ProgressRail } from '../components/ProgressRail'
import { PricePanel } from '../components/PricePanel'

export function Analyze() {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [ticker, setTicker] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  const done = useMemo(() => new Set(events.map((e) => e.agent)), [events])

  const run = (t: string) => {
    esRef.current?.close()
    setEvents([]); setVerdict(null); setError(null); setRunning(true); setTicker(t)
    esRef.current = streamSSE(`/api/analyze?ticker=${encodeURIComponent(t)}`, {
      agent: (d) => setEvents((prev) => [...prev, d]),
      verdict: (d) => setVerdict(d),
      error: (d) => setError(d.message ?? 'error'),
    }, () => setRunning(false))
  }

  return (
    <div className="space-y-4">
      <TickerInput onSubmit={run} disabled={running} />
      {error && <div className="text-sm" style={{ color: 'var(--bear)' }}>{error}</div>}
      {ticker && <PricePanel ticker={ticker} />}
      {(running || events.length > 0) && !verdict && <ProgressRail done={done} running={running} />}
      {!verdict && <div className="space-y-3">{events.map((e, i) => <AgentCard key={i} event={e} />)}</div>}
      {verdict && <VerdictCard verdict={verdict} />}
    </div>
  )
}
```

- [ ] **Step 3: Run Analyze tests**

Run: `npm test -- src/pages/Analyze.test.tsx`
Expected: PASS (unchanged test — still finds the ticker input).

- [ ] **Step 4: Commit**

```bash
git add src/components/ProgressRail.tsx src/pages/Analyze.tsx
git commit -m "feat(ui): Analyze page — progress rail, price panel, polish"
```

---

### Task 5.3: Backtest page — StatTiles + equity curve + forward-return

Replace the inline-SVG `LongShortCurve` usage with a lightweight-charts equity curve and StatTiles.

**Files:**
- Create: `src/components/EquityCurve.tsx`
- Create: `src/components/EquityCurve.test.tsx`
- Modify: `src/components/BacktestReport.tsx`
- Modify: `src/components/BacktestReport.test.tsx` (keep behavioral assertions)
- Modify: `src/pages/Backtest.tsx`

- [ ] **Step 1: Write failing test for `EquityCurve` (mock the lib)**

```tsx
import { render, screen } from '@testing-library/react'
import { EquityCurve } from './EquityCurve'

const setData = vi.fn()
const addLineSeries = vi.fn(() => ({ setData }))
vi.mock('lightweight-charts', () => ({
  createChart: () => ({ addLineSeries, timeScale: () => ({ fitContent: vi.fn() }), applyOptions: vi.fn(), remove: vi.fn() }),
  ColorType: { Solid: 'solid' },
}))

test('feeds cumulative curve', () => {
  render(<EquityCurve curve={[0.1, -0.05, 0.2]} />)
  expect(addLineSeries).toHaveBeenCalled()
  expect(setData).toHaveBeenCalled()
})

test('empty curve shows no data', () => {
  render(<EquityCurve curve={[]} />)
  expect(screen.getByText(/no data/i)).toBeInTheDocument()
})
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- src/components/EquityCurve.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `src/components/EquityCurve.tsx`**

The backend `long_short_curve` is a numeric array (per-step values); index it as sequential points.

```tsx
import { useEffect, useRef } from 'react'
import { ColorType, createChart, type IChartApi } from 'lightweight-charts'

export function EquityCurve({ curve }: { curve: number[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!ref.current || curve.length === 0) return
    const chart = createChart(ref.current, {
      width: ref.current.clientWidth, height: 140,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#1e293b' }, horzLines: { color: '#1e293b' } },
      rightPriceScale: { borderColor: '#263352' },
      timeScale: { borderColor: '#263352' },
    })
    chartRef.current = chart
    const s = chart.addLineSeries({ color: '#38bdf8', lineWidth: 2 })
    s.setData(curve.map((v, i) => ({ time: (i + 1) as unknown as string, value: v })))
    chart.timeScale().fitContent()
    const ro = new ResizeObserver(() => ref.current && chart.applyOptions({ width: ref.current.clientWidth }))
    ro.observe(ref.current)
    return () => { ro.disconnect(); chart.remove(); chartRef.current = null }
  }, [curve])

  if (curve.length === 0) return <div className="text-xs" style={{ color: 'var(--text-mut)' }}>no data</div>
  return <div ref={ref} className="w-full" />
}
```

Note: lightweight-charts accepts integer indices as `time` for a plain sequential line; the cast keeps TypeScript happy.

- [ ] **Step 4: Run to verify pass**

Run: `npm test -- src/components/EquityCurve.test.tsx`
Expected: PASS.

- [ ] **Step 5: Rewrite `src/components/BacktestReport.tsx`**

```tsx
import type { BacktestReport } from '../api/types'
import { StatTile } from '../ui/primitives'
import { Card } from '../ui/primitives'
import { EquityCurve } from './EquityCurve'

const pct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`

export function BacktestReportView({ report }: { report: BacktestReport }) {
  return (
    <div className="space-y-4">
      <div className="text-sm" style={{ color: 'var(--text-mut)' }}>{report.n_records} verdicts</div>
      {Object.entries(report.horizons).map(([h, m]) => {
        const meanKeys = Object.keys(m.mean_return)
        return (
          <Card key={h} title={`Horizon ${h} trading days (n=${m.n})`}>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <StatTile label="IC" value={m.ic === null ? 'n/a' : m.ic.toFixed(3)}
                tone={m.ic != null && m.ic > 0 ? 'bull' : m.ic != null && m.ic < 0 ? 'bear' : undefined} />
              {meanKeys.map((k) => (
                <StatTile key={k} label={`Mean ${k}`} value={pct(m.mean_return[k])}
                  tone={m.mean_return[k] >= 0 ? 'bull' : 'bear'} />
              ))}
              {Object.entries(m.hit_rate).map(([k, v]) => (
                <StatTile key={`hit-${k}`} label={`Hit ${k}`} value={`${Math.round(v * 100)}%`} />
              ))}
            </div>
            <div className="mt-3 text-xs uppercase tracking-wide" style={{ color: 'var(--text-mut)' }}>Long-short curve</div>
            <div className="mt-1"><EquityCurve curve={m.long_short_curve} /></div>
          </Card>
        )
      })}
      <div className="text-xs" style={{ color: 'var(--text-mut)' }}>{report.disclaimer}</div>
    </div>
  )
}
```

- [ ] **Step 6: Rewrite `src/components/BacktestReport.test.tsx`**

The old assertion on "information coefficient" no longer applies (now an `IC` StatTile). Mock `./EquityCurve` to keep the chart lib out of this test. Replace the file with:

```tsx
import { render, screen } from '@testing-library/react'
import { BacktestReportView } from './BacktestReport'

vi.mock('./EquityCurve', () => ({ EquityCurve: () => <div>curve</div> }))

const report = {
  n_records: 2, disclaimer: 'small sample',
  horizons: {
    '21': { n: 2, hit_rate: { buy: 0.5 }, mean_return: { buy: 0.03 }, ic: 0.2, long_short_curve: [0.1, -0.1] },
    '63': { n: 2, hit_rate: {}, mean_return: {}, ic: null, long_short_curve: [] },
  },
}

test('renders horizons, IC tile, and curve', () => {
  render(<BacktestReportView report={report} />)
  expect(screen.getByText(/Horizon 21 trading days/i)).toBeInTheDocument()
  expect(screen.getByText(/Horizon 63 trading days/i)).toBeInTheDocument()
  expect(screen.getAllByText('IC').length).toBeGreaterThan(0)
  expect(screen.getByText('0.200')).toBeInTheDocument()   // ic formatted
  expect(screen.getAllByText('curve').length).toBeGreaterThan(0)
})
```

- [ ] **Step 7: Rewrite `src/pages/Backtest.tsx` (dark button + progress card)**

```tsx
import { useState } from 'react'
import { streamSSE } from '../api/client'
import type { BacktestReport } from '../api/types'
import { BacktestReportView } from '../components/BacktestReport'

export function Backtest() {
  const [progress, setProgress] = useState<string[]>([])
  const [report, setReport] = useState<BacktestReport | null>(null)
  const [running, setRunning] = useState(false)

  const run = () => {
    setProgress([]); setReport(null); setRunning(true)
    streamSSE('/api/backtest', {
      progress: (d) => setProgress((p) => [...p, `${d.ticker} @ ${d.as_of}`]),
      report: (d) => setReport(d),
      error: (d) => setProgress((p) => [...p, `error: ${d.message}`]),
    }, () => setRunning(false))
  }

  return (
    <div className="space-y-4">
      <button onClick={run} disabled={running}
        className="rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-50"
        style={{ background: 'var(--accent)', color: '#04121f' }}>
        Run backtest
      </button>
      {running && <div className="text-sm" style={{ color: 'var(--text-dim)' }}>Running… {progress.length} done</div>}
      {progress.length > 0 && !report && (
        <ul className="tnum space-y-0.5 text-xs" style={{ color: 'var(--text-mut)' }}>{progress.map((p, i) => <li key={i}>{p}</li>)}</ul>
      )}
      {report && <BacktestReportView report={report} />}
    </div>
  )
}
```

- [ ] **Step 8: Delete the now-unused `LongShortCurve`**

Remove `src/components/LongShortCurve.tsx` and `src/components/LongShortCurve.test.tsx` (replaced by `EquityCurve`). Confirm nothing else imports them: `grep -rn LongShortCurve src`.

- [ ] **Step 9: Run full test suite**

Run: `npm test`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add src/components/EquityCurve.tsx src/components/EquityCurve.test.tsx src/components/BacktestReport.tsx src/components/BacktestReport.test.tsx src/pages/Backtest.tsx
git rm src/components/LongShortCurve.tsx src/components/LongShortCurve.test.tsx
git commit -m "feat(ui): backtest StatTiles + equity curve chart"
```

---

### Task 5.4: Ingest page dark styling

**Files:**
- Modify: `src/pages/Ingest.tsx`

- [ ] **Step 1: Rewrite `src/pages/Ingest.tsx`**

```tsx
import { useState } from 'react'
import { postJSON } from '../api/client'
import { TickerInput } from '../components/TickerInput'
import { Card } from '../ui/primitives'

export function Ingest() {
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const run = async (ticker: string) => {
    setBusy(true); setMsg(null)
    try {
      const r = await postJSON<{ ticker: string; ingested: number }>('/api/ingest', { ticker })
      setMsg(`Ingested ${r.ingested} news items for ${r.ticker}`)
    } catch (e) {
      setMsg(`Error: ${(e as Error).message}`)
    } finally { setBusy(false) }
  }
  return (
    <div className="space-y-3">
      <Card title="Ingest news + filings">
        <TickerInput onSubmit={run} disabled={busy} />
        {msg && <div className="mt-3 text-sm" style={{ color: 'var(--text-dim)' }}>{msg}</div>}
      </Card>
    </div>
  )
}
```

- [ ] **Step 2: Run tests + build**

Run: `npm test && npm run build`
Expected: all pass, build succeeds.

- [ ] **Step 3: Commit**

```bash
git add src/pages/Ingest.tsx
git commit -m "feat(ui): dark Ingest page"
```

---

## Part 6 — Live verification

### Task 6.1: End-to-end browser check

- [ ] **Step 1: Start backend (in the backend repo) and frontend dev server**

Backend: `uv run uvicorn api.main:app --port 8000`
Frontend: `npm run dev` (Vite proxies `/api` → 8000).

- [ ] **Step 2: In the browser, verify:**
  - Dark shell renders; nav active-tab underline; health chip green (or "ollama offline" red).
  - Analyze `AAPL`: price chart renders with SMA overlays; timeframe tabs switch (1M/3M/6M/1Y) and refetch; RSI subchart shows the 30/70 bands; progress-rail pulses while agents stream, marks ✓ as each arrives; agent cards fade in with accent border, score bar, confidence meter, gauges, key-fact chips; verdict card shows badge + confidence meter + narrative + caution.
  - Backtest: StatTiles populate per horizon; equity curve renders.
  - Ingest: form works, result message shows.
  - Resize to ~400px width: no horizontal scroll; charts resize; grids stack.

- [ ] **Step 3: Stop both servers.** No commit (verification only).

---

## Completion

- Merge `feature/ui-v2-redesign` (client repo) via **superpowers:finishing-a-development-branch**.
- Part 0's `feature/agent-metrics` (backend repo) is merged before frontend work begins.
- All commits authored as the repo owner with NO `Co-Authored-By`; client repo carries no Claude/AI references.
