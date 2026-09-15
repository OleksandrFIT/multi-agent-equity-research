# Phase 4B — React+TS frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit attribution:** commit ONLY as the repo author (OleksandrFIT). Do NOT add any `Co-Authored-By` trailer, and put NO reference to Claude/AI anywhere in code, comments, README, or commit messages — the repo must read as hand-written.
>
> **Working directory for ALL tasks:** `/Users/oleksandr/Documents/LLM/Multi-Agent Equity Client` (separate from the Python repo). Its own git repo; remote `git@github.com:OleksandrFIT/Multi-Agent-Equity-Web-Client.git`.

**Goal:** A React + TypeScript (Vite) web UI that consumes the FastAPI backend: Analyze (SSE per-agent stream → verdict), Backtest (SSE progress → report + long-short curve), and Ingest.

**Architecture:** Presentational components (AgentCard, VerdictCard, BacktestReport, LongShortCurve) are pure and unit-tested with Vitest + Testing Library. A thin `api` layer (`types.ts`, `client.ts`) wraps `fetch` + `EventSource`. Pages wire the client to components; verified live against a running backend. Tailwind for styling; react-router for the 3 routes. Dev proxy `/api → http://localhost:8000`.

**Tech Stack:** Node 18+/npm, Vite, React 18, TypeScript, Tailwind v4, react-router-dom, Vitest, @testing-library/react, jsdom.

**Backend contract (from Phase 4A, do not change):**
- `GET /api/health` → `{ok, model, ollama_reachable}`
- `GET /api/analyze?ticker=X` → SSE: `event: agent` data `{agent, opinion}` or `{agent, skipped, reason}`; `event: verdict` data `Verdict`; `event: error` data `{message}`
- `POST /api/ingest {ticker}` → `{ticker, ingested}`
- `GET /api/backtest` → SSE: `event: progress` data `{ticker, as_of}`; `event: report` data `BacktestReport`; `event: error`
- `GET /api/backtest/config` → `{universe, dates, horizons}`

## File Structure (in the web client repo)

- `index.html`, `package.json`, `tsconfig*.json`, `vite.config.ts` — scaffold + config.
- `src/main.tsx`, `src/App.tsx` (router), `src/index.css` (Tailwind), `src/test-setup.ts`.
- `src/api/types.ts` — TS mirrors of the JSON contract.
- `src/api/client.ts` — `getJSON`, `postJSON`, `streamSSE`.
- `src/components/AgentCard.tsx`, `VerdictCard.tsx`, `BacktestReport.tsx`, `LongShortCurve.tsx`, `HealthBanner.tsx`, `TickerInput.tsx`.
- `src/pages/Analyze.tsx`, `Backtest.tsx`, `Ingest.tsx`.
- `src/components/*.test.tsx` — Vitest tests.

---

## Task 1: Scaffold Vite + React + TS + Tailwind + router + Vitest

**Files:** whole scaffold (generated) + config edits + `src/App.smoke.test.tsx`.

- [ ] **Step 1: Scaffold and install**

Run (in the working dir, which is empty):
```bash
npm create vite@latest . -- --template react-ts
npm install
npm install react-router-dom
npm install -D tailwindcss @tailwindcss/vite vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom
git init
```

- [ ] **Step 2: Configure Tailwind, proxy, Vitest**

Replace `vite.config.ts` with:
```ts
/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
  test: { environment: 'jsdom', globals: true, setupFiles: './src/test-setup.ts' },
})
```

Replace `src/index.css` with:
```css
@import "tailwindcss";
```

Create `src/test-setup.ts`:
```ts
import '@testing-library/jest-dom'
```

Create `.gitignore` (if `npm create` didn't already — ensure these are present):
```
node_modules
dist
*.local
.DS_Store
```

Add test scripts to `package.json` `"scripts"` (keep existing dev/build/preview):
```json
    "test": "vitest run",
    "test:watch": "vitest"
```

- [ ] **Step 3: Write the smoke test + minimal App**

Replace `src/App.tsx`:
```tsx
export default function App() {
  return <h1>Equity Research</h1>
}
```

Create `src/App.smoke.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import App from './App'

test('renders app title', () => {
  render(<App />)
  expect(screen.getByText('Equity Research')).toBeInTheDocument()
})
```

- [ ] **Step 4: Run the test**

Run: `npm test`
Expected: 1 passed. Also confirm `npm run build` succeeds (TypeScript compiles).

- [ ] **Step 5: Commit + wire remote**

```bash
git add -A
git commit -m "chore: scaffold Vite + React + TS + Tailwind + Vitest"
git remote add origin git@github.com:OleksandrFIT/Multi-Agent-Equity-Web-Client.git
```
(Do not push yet — push in Task 6. Author must be OleksandrFIT; if `git config user.name/email` are unset in this repo, set them to `OleksandrFIT` / `dragan.sasha2002@gmail.com` first.)

---

## Task 2: API types + client

**Files:** `src/api/types.ts`, `src/api/client.ts`, `src/api/client.test.ts`.

- [ ] **Step 1: Write the failing test**

`src/api/client.test.ts`:
```ts
import { getJSON } from './client'

test('getJSON parses a JSON response', async () => {
  const fake = { ok: true } as Response
  ;(fake as any).json = async () => ({ model: 'qwen2.5:7b' })
  vi.stubGlobal('fetch', vi.fn(async () => fake))
  const data = await getJSON<{ model: string }>('/api/health')
  expect(data.model).toBe('qwen2.5:7b')
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test`
Expected: FAIL — `./client` not found.

- [ ] **Step 3: Write minimal implementation**

`src/api/types.ts`:
```ts
export type Stance = 'bullish' | 'neutral' | 'bearish'

export interface AgentOpinion {
  agent: string
  stance: Stance
  score: number
  confidence: number
  rationale: string
  key_facts: string[]
  dropped_facts: string[]
}

export interface Verdict {
  ticker: string
  as_of: string
  verdict: 'buy' | 'hold' | 'sell'
  score: number
  confidence: number
  narrative: string
  opinions: AgentOpinion[]
  disclaimer: string
  caution: string | null
  skipped_agents: string[]
  skip_reasons: Record<string, string>
}

export type AgentEvent =
  | { agent: string; opinion: AgentOpinion }
  | { agent: string; skipped: true; reason: string }

export interface HorizonMetrics {
  n: number
  hit_rate: Record<string, number>
  mean_return: Record<string, number>
  ic: number | null
  long_short_curve: number[]
}
export interface BacktestReport {
  n_records: number
  horizons: Record<string, HorizonMetrics>
  disclaimer: string
}
export interface BacktestConfig { universe: string[]; dates: string[]; horizons: number[] }
export interface Health { ok: boolean; model: string; ollama_reachable: boolean }
```

`src/api/client.ts`:
```ts
export async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`GET ${url} failed: ${r.status}`)
  return r.json()
}

export async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`POST ${url} failed: ${r.status}`)
  return r.json()
}

export interface SSEHandlers {
  [event: string]: (data: any) => void
}

// Opens an SSE stream. `handlers` maps event names to callbacks; `onDone` is
// called when a terminal event ('verdict' | 'report' | 'error') arrives, and the
// stream is closed. The browser fires a spurious 'error' on normal close — we
// only treat it as a failure if it carries data (our backend error event).
export function streamSSE(
  url: string,
  handlers: SSEHandlers,
  onDone?: () => void,
): EventSource {
  const es = new EventSource(url)
  const terminal = new Set(['verdict', 'report', 'error'])
  for (const [event, cb] of Object.entries(handlers)) {
    es.addEventListener(event, (e: MessageEvent) => {
      cb(JSON.parse(e.data))
      if (terminal.has(event)) {
        es.close()
        onDone?.()
      }
    })
  }
  es.onerror = () => {
    // normal stream close after a terminal event; ignore unless still open early
    if (es.readyState === EventSource.CLOSED) return
    es.close()
    handlers['error']?.({ message: 'connection lost' })
    onDone?.()
  }
  return es
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test`
Expected: PASS. `npm run build` compiles.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: api types + fetch/SSE client"
```

---

## Task 3: AgentCard + VerdictCard

**Files:** `src/components/AgentCard.tsx`, `VerdictCard.tsx`, `AgentCard.test.tsx`, `VerdictCard.test.tsx`.

- [ ] **Step 1: Write the failing tests**

`src/components/AgentCard.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import { AgentCard } from './AgentCard'

test('renders an opinion agent', () => {
  render(<AgentCard event={{ agent: 'technical', opinion: {
    agent: 'technical', stance: 'bullish', score: 0.7, confidence: 0.8,
    rationale: 'golden cross', key_facts: ['RSI 67'], dropped_facts: [] } }} />)
  expect(screen.getByText(/technical/i)).toBeInTheDocument()
  expect(screen.getByText(/bullish/i)).toBeInTheDocument()
  expect(screen.getByText(/golden cross/i)).toBeInTheDocument()
})

test('renders a skipped agent', () => {
  render(<AgentCard event={{ agent: 'sentiment', skipped: true, reason: 'no news' }} />)
  expect(screen.getByText(/sentiment/i)).toBeInTheDocument()
  expect(screen.getByText(/no news/i)).toBeInTheDocument()
})
```

`src/components/VerdictCard.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import { VerdictCard } from './VerdictCard'

const v = {
  ticker: 'AAPL', as_of: '2026-09-15', verdict: 'buy' as const, score: 0.42,
  confidence: 0.7, narrative: 'Looks strong.', opinions: [], disclaimer: 'not advice',
  caution: 'Elevated risk', skipped_agents: ['sentiment'],
  skip_reasons: { sentiment: 'no news' },
}

test('renders verdict headline, caution and disclaimer', () => {
  render(<VerdictCard verdict={v} />)
  expect(screen.getByText('AAPL')).toBeInTheDocument()
  expect(screen.getByText(/BUY/i)).toBeInTheDocument()
  expect(screen.getByText(/Elevated risk/i)).toBeInTheDocument()
  expect(screen.getByText(/not advice/i)).toBeInTheDocument()
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npm test`
Expected: FAIL — components not found.

- [ ] **Step 3: Write minimal implementation**

`src/components/AgentCard.tsx`:
```tsx
import type { AgentEvent } from '../api/types'

const stanceColor: Record<string, string> = {
  bullish: 'text-green-700', bearish: 'text-red-700', neutral: 'text-gray-600',
}

export function AgentCard({ event }: { event: AgentEvent }) {
  if ('skipped' in event) {
    return (
      <div className="rounded-xl border border-gray-200 p-3 opacity-70">
        <div className="font-medium capitalize">{event.agent}</div>
        <div className="text-sm text-gray-500">skipped — {event.reason}</div>
      </div>
    )
  }
  const o = event.opinion
  return (
    <div className="rounded-xl border border-gray-200 p-3">
      <div className="flex items-center justify-between">
        <span className="font-medium capitalize">{o.agent}</span>
        <span className={`text-sm ${stanceColor[o.stance] ?? ''}`}>
          {o.stance} ({o.score >= 0 ? '+' : ''}{o.score.toFixed(2)}, {Math.round(o.confidence * 100)}%)
        </span>
      </div>
      <p className="mt-1 text-sm text-gray-700">{o.rationale}</p>
      {o.key_facts.length > 0 && (
        <div className="mt-1 text-xs text-gray-500">{o.key_facts.join(' · ')}</div>
      )}
    </div>
  )
}
```

`src/components/VerdictCard.tsx`:
```tsx
import type { Verdict } from '../api/types'
import { AgentCard } from './AgentCard'

const verdictColor: Record<string, string> = {
  buy: 'bg-green-100 text-green-800', hold: 'bg-gray-100 text-gray-800', sell: 'bg-red-100 text-red-800',
}

export function VerdictCard({ verdict }: { verdict: Verdict }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <h2 className="text-2xl font-semibold">{verdict.ticker}</h2>
        <span className={`rounded-lg px-3 py-1 text-sm font-medium uppercase ${verdictColor[verdict.verdict]}`}>
          {verdict.verdict}
        </span>
        <span className="text-sm text-gray-500">
          score {verdict.score >= 0 ? '+' : ''}{verdict.score.toFixed(2)} · confidence {Math.round(verdict.confidence * 100)}%
        </span>
      </div>
      <p className="text-gray-800">{verdict.narrative}</p>
      <div className="space-y-2">
        {verdict.opinions.map((o) => (
          <AgentCard key={o.agent} event={{ agent: o.agent, opinion: o }} />
        ))}
      </div>
      {verdict.caution && (
        <div className="rounded-lg bg-amber-50 p-2 text-sm text-amber-800">⚠ {verdict.caution}</div>
      )}
      {verdict.skipped_agents.length > 0 && (
        <div className="text-xs text-gray-500">
          Skipped: {verdict.skipped_agents.map((a) => `${a}${verdict.skip_reasons[a] ? ` (${verdict.skip_reasons[a]})` : ''}`).join(', ')}
        </div>
      )}
      <div className="border-t pt-2 text-xs text-gray-400">{verdict.disclaimer}</div>
    </div>
  )
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `npm test`
Expected: PASS. `npm run build` compiles.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: AgentCard + VerdictCard"
```

---

## Task 4: BacktestReport + LongShortCurve

**Files:** `src/components/BacktestReport.tsx`, `LongShortCurve.tsx`, `BacktestReport.test.tsx`, `LongShortCurve.test.tsx`.

- [ ] **Step 1: Write the failing tests**

`src/components/LongShortCurve.test.tsx`:
```tsx
import { render } from '@testing-library/react'
import { LongShortCurve } from './LongShortCurve'

test('renders a polyline with one point per value', () => {
  const { container } = render(<LongShortCurve curve={[0.1, -0.1, 0.2]} />)
  const poly = container.querySelector('polyline')
  expect(poly).not.toBeNull()
  expect(poly!.getAttribute('points')!.trim().split(/\s+/).length).toBe(3)
})

test('renders empty state for no data', () => {
  const { container } = render(<LongShortCurve curve={[]} />)
  expect(container.querySelector('polyline')).toBeNull()
})
```

`src/components/BacktestReport.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import { BacktestReportView } from './BacktestReport'

const report = {
  n_records: 2, disclaimer: 'small sample',
  horizons: {
    '21': { n: 2, hit_rate: { buy: 0.5 }, mean_return: { buy: 0.03 }, ic: 0.2, long_short_curve: [0.1, -0.1] },
    '63': { n: 2, hit_rate: {}, mean_return: {}, ic: null, long_short_curve: [] },
  },
}

test('renders horizons and IC', () => {
  render(<BacktestReportView report={report} />)
  expect(screen.getByText(/21/)).toBeInTheDocument()
  expect(screen.getByText(/63/)).toBeInTheDocument()
  expect(screen.getByText(/information coefficient/i)).toBeInTheDocument()
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npm test`
Expected: FAIL — components not found.

- [ ] **Step 3: Write minimal implementation**

`src/components/LongShortCurve.tsx`:
```tsx
export function LongShortCurve({ curve }: { curve: number[] }) {
  if (curve.length === 0) return <div className="text-sm text-gray-400">no data</div>
  const w = 320, h = 80, pad = 4
  const min = Math.min(0, ...curve), max = Math.max(0, ...curve)
  const range = max - min || 1
  const x = (i: number) => pad + (i * (w - 2 * pad)) / Math.max(1, curve.length - 1)
  const y = (v: number) => h - pad - ((v - min) / range) * (h - 2 * pad)
  const points = curve.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const zeroY = y(0)
  return (
    <svg width={w} height={h} className="rounded border border-gray-200">
      <line x1={pad} y1={zeroY} x2={w - pad} y2={zeroY} stroke="#e5e7eb" />
      <polyline points={points} fill="none" stroke="#2563eb" strokeWidth={1.5} />
    </svg>
  )
}
```

`src/components/BacktestReport.tsx`:
```tsx
import type { BacktestReport } from '../api/types'
import { LongShortCurve } from './LongShortCurve'

const pct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`

export function BacktestReportView({ report }: { report: BacktestReport }) {
  return (
    <div className="space-y-4">
      <div className="text-sm text-gray-500">{report.n_records} verdicts</div>
      {Object.entries(report.horizons).map(([h, m]) => (
        <div key={h} className="rounded-xl border border-gray-200 p-3">
          <div className="font-medium">Horizon {h} trading days (n={m.n})</div>
          <div className="text-sm text-gray-700">
            Information coefficient: {m.ic === null ? 'n/a' : m.ic.toFixed(3)}
          </div>
          <div className="text-sm text-gray-700">
            Hit rate: {Object.entries(m.hit_rate).map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(', ') || 'n/a'}
          </div>
          <div className="text-sm text-gray-700">
            Mean return: {Object.entries(m.mean_return).map(([k, v]) => `${k} ${pct(v)}`).join(', ') || 'n/a'}
          </div>
          <div className="mt-2"><LongShortCurve curve={m.long_short_curve} /></div>
        </div>
      ))}
      <div className="text-xs text-gray-400">{report.disclaimer}</div>
    </div>
  )
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `npm test`
Expected: PASS. `npm run build` compiles.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: BacktestReport + LongShortCurve (inline SVG)"
```

---

## Task 5: Pages + router + HealthBanner

**Files:** `src/components/TickerInput.tsx`, `HealthBanner.tsx`, `src/pages/Analyze.tsx`, `Backtest.tsx`, `Ingest.tsx`, `src/App.tsx`.

- [ ] **Step 1: Write a light failing test**

`src/pages/Analyze.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react'
import { Analyze } from './Analyze'

test('analyze page shows a ticker input', () => {
  render(<Analyze />)
  expect(screen.getByPlaceholderText(/ticker/i)).toBeInTheDocument()
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm test`
Expected: FAIL — page not found.

- [ ] **Step 3: Write minimal implementation**

`src/components/TickerInput.tsx`:
```tsx
import { useState } from 'react'

export function TickerInput({ onSubmit, disabled }: { onSubmit: (t: string) => void; disabled?: boolean }) {
  const [t, setT] = useState('')
  return (
    <form onSubmit={(e) => { e.preventDefault(); if (t.trim()) onSubmit(t.trim().toUpperCase()) }} className="flex gap-2">
      <input value={t} onChange={(e) => setT(e.target.value)} placeholder="Ticker (e.g. AAPL)"
        className="rounded-lg border border-gray-300 px-3 py-2" />
      <button disabled={disabled} className="rounded-lg bg-blue-600 px-4 py-2 text-white disabled:opacity-50">Go</button>
    </form>
  )
}
```

`src/components/HealthBanner.tsx`:
```tsx
import { useEffect, useState } from 'react'
import { getJSON } from '../api/client'
import type { Health } from '../api/types'

export function HealthBanner() {
  const [h, setH] = useState<Health | null>(null)
  useEffect(() => { getJSON<Health>('/api/health').then(setH).catch(() => setH({ ok: false, model: '', ollama_reachable: false })) }, [])
  if (h && !h.ollama_reachable) {
    return <div className="bg-red-100 p-2 text-sm text-red-800">Ollama is not reachable — start it and reload.</div>
  }
  return null
}
```

`src/pages/Analyze.tsx`:
```tsx
import { useRef, useState } from 'react'
import { streamSSE } from '../api/client'
import type { AgentEvent, Verdict } from '../api/types'
import { AgentCard } from '../components/AgentCard'
import { VerdictCard } from '../components/VerdictCard'
import { TickerInput } from '../components/TickerInput'

export function Analyze() {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  const run = (ticker: string) => {
    esRef.current?.close()
    setEvents([]); setVerdict(null); setError(null); setRunning(true)
    esRef.current = streamSSE(`/api/analyze?ticker=${encodeURIComponent(ticker)}`, {
      agent: (d) => setEvents((prev) => [...prev, d]),
      verdict: (d) => setVerdict(d),
      error: (d) => setError(d.message ?? 'error'),
    }, () => setRunning(false))
  }

  return (
    <div className="space-y-4">
      <TickerInput onSubmit={run} disabled={running} />
      {error && <div className="text-red-700">{error}</div>}
      {!verdict && events.map((e, i) => <AgentCard key={i} event={e} />)}
      {running && <div className="text-sm text-gray-500">Analyzing… (agents run one by one)</div>}
      {verdict && <VerdictCard verdict={verdict} />}
    </div>
  )
}
```

`src/pages/Backtest.tsx`:
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
      <button onClick={run} disabled={running} className="rounded-lg bg-blue-600 px-4 py-2 text-white disabled:opacity-50">
        Run backtest
      </button>
      {running && <div className="text-sm text-gray-500">Running… {progress.length} done</div>}
      {progress.length > 0 && !report && (
        <ul className="text-xs text-gray-500">{progress.map((p, i) => <li key={i}>{p}</li>)}</ul>
      )}
      {report && <BacktestReportView report={report} />}
    </div>
  )
}
```

`src/pages/Ingest.tsx`:
```tsx
import { useState } from 'react'
import { postJSON } from '../api/client'
import { TickerInput } from '../components/TickerInput'

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
      <TickerInput onSubmit={run} disabled={busy} />
      {msg && <div className="text-sm text-gray-700">{msg}</div>}
    </div>
  )
}
```

`src/App.tsx`:
```tsx
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import { HealthBanner } from './components/HealthBanner'
import { Analyze } from './pages/Analyze'
import { Backtest } from './pages/Backtest'
import { Ingest } from './pages/Ingest'

export default function App() {
  return (
    <BrowserRouter>
      <div className="mx-auto max-w-3xl p-6">
        <header className="mb-4 flex items-center gap-4">
          <h1 className="text-xl font-semibold">Equity Research</h1>
          <nav className="flex gap-3 text-sm text-blue-600">
            <Link to="/">Analyze</Link>
            <Link to="/backtest">Backtest</Link>
            <Link to="/ingest">Ingest</Link>
          </nav>
        </header>
        <HealthBanner />
        <Routes>
          <Route path="/" element={<Analyze />} />
          <Route path="/backtest" element={<Backtest />} />
          <Route path="/ingest" element={<Ingest />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
```
Note: `src/App.smoke.test.tsx` from Task 1 asserted the `<h1>Equity Research</h1>` text — it still passes (the header still renders that text). If it fails due to the router, update the smoke test to wrap-free assert `getByText('Equity Research')` (still present).

- [ ] **Step 4: Run to verify they pass**

Run: `npm test`
Expected: PASS (all component/page tests). `npm run build` compiles with no TS errors.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: Analyze/Backtest/Ingest pages + router + health banner"
```

---

## Task 6: Live run + README + push

**Files:** `README.md`.

- [ ] **Step 1: Start backend + frontend**

In the Python repo: `DISABLE_PANDERA_IMPORT_WARNING=True uv run uvicorn api.main:app --port 8000` (background).
In the web client dir: `npm run dev` (background; Vite serves on http://localhost:5173).

- [ ] **Step 2: Manual verification (report what you see)**

Open http://localhost:5173. Verify: the Analyze page loads; entering `AAPL` streams agent cards one by one, then shows the verdict; the Backtest page runs and renders metrics + the curve; the Ingest page returns a news count; the health banner appears only when Ollama is down. Stop both servers afterward.

- [ ] **Step 3: Write `README.md`**

```markdown
# Multi-Agent Equity Research — Web Client

React + TypeScript (Vite) UI for the equity-research API.

## Run
    npm install
    npm run dev        # http://localhost:5173

Requires the API running on http://localhost:8000 (see the backend repo).
Dev requests to `/api` are proxied to the backend.

## Build / test
    npm run build
    npm test

## Screens
- Analyze — enter a ticker; agents stream in one by one, then a verdict.
- Backtest — run the historical backtest; metrics per horizon + long-short curve.
- Ingest — pull recent news for a ticker into the vector store.
```

- [ ] **Step 4: Commit + push**

```bash
git add -A
git commit -m "docs: readme"
git push -u origin main
```
(If the default branch is `master`, push `master`; the repo was created empty so the local branch name wins. Author must be OleksandrFIT; no Claude/AI text anywhere.)

---

## Self-Review (completed during authoring)

- **Spec coverage (4B):** Vite+React+TS+Tailwind+router scaffold (Task 1); api types + fetch/SSE client with EventSource close-on-terminal + spurious-error handling (Task 2); AgentCard/VerdictCard (Task 3); BacktestReport + inline-SVG LongShortCurve (Task 4); Analyze/Backtest/Ingest pages + HealthBanner + router (Task 5); live verify + README + push to the separate web-client repo (Task 6). Consumes exactly the Phase 4A contract; inline-SVG curve (no chart dep). Error handling: HealthBanner + SSE error events + POST try/catch (spec §4).
- **Placeholder scan:** none in code. Scaffold uses generator commands (can't reproduce every generated file verbatim) plus concrete config/file contents — standard for a scaffolding task.
- **Type consistency:** `AgentEvent`/`Verdict`/`BacktestReport`/`Health` in `types.ts` match the 4A JSON; `getJSON/postJSON/streamSSE` signatures are used consistently by HealthBanner/pages; component prop names (`event`, `verdict`, `report`, `curve`) match across component + test files.
