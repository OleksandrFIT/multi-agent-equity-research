# P1 — Analyze UX implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A ticker-tape of 50 stocks (price + %change) above Analyze, and LLM-based ticker resolution that corrects typos (APPL→AAPL) or says "not found".

**Architecture:** Two new backend endpoints — `/api/quotes` (batch daily-cached quotes) and `/api/resolve` (validate-or-LLM-correct a ticker). Frontend adds a marquee `TickerTape` and a resolve step before analyze.

**Tech Stack:** FastAPI + yfinance (backend); React + TS + Vitest (frontend).

**Two repos / two branches:**
- **Part A** — backend repo `/Users/oleksandr/Documents/LLM/Multi-Agent Equity`, branch `feature/p1-quotes-resolve`.
- **Part B** — client repo `/Users/oleksandr/Documents/LLM/Multi-Agent Equity Client`, branch `feature/p1-analyze-ux`.

**Attribution:** commit ONLY as the repo author, NO `Co-Authored-By`. Client repo: no Claude/AI/assistant references anywhere.

---

## Part A — Backend

### Task A1: `fetch_quotes` adapter

**Files:**
- Modify: `equity_research/data/adapters.py`
- Test: `tests/test_adapters_quotes.py` (new)

- [ ] **Step 1: Write the failing test (`tests/test_adapters_quotes.py`)**

```python
import pandas as pd

from equity_research.data import adapters


def test_fetch_quotes_maps_price_and_change(monkeypatch):
    idx = pd.date_range("2026-09-10", periods=3, freq="D")
    frame = pd.DataFrame({("Close", "AAPL"): [100.0, 110.0, 121.0],
                          ("Close", "MSFT"): [200.0, 200.0, 210.0]}, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    monkeypatch.setattr(adapters, "_yf_download", lambda tickers: frame)
    out = adapters.fetch_quotes(["AAPL", "MSFT"])
    assert out["AAPL"] == {"price": 121.0, "change_pct": 10.0}   # (121-110)/110
    assert out["MSFT"] == {"price": 210.0, "change_pct": 5.0}


def test_fetch_quotes_skips_missing(monkeypatch):
    idx = pd.date_range("2026-09-10", periods=1, freq="D")  # only one row -> no prev close
    frame = pd.DataFrame({("Close", "AAPL"): [100.0]}, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    monkeypatch.setattr(adapters, "_yf_download", lambda tickers: frame)
    assert adapters.fetch_quotes(["AAPL"]) == {}
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_adapters_quotes.py -v`
Expected: FAIL (`fetch_quotes` / `_yf_download` missing).

- [ ] **Step 3: Add to `equity_research/data/adapters.py`**

```python
def _yf_download(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(tickers, period="5d", auto_adjust=True, progress=False)


def fetch_quotes(tickers: list[str]) -> dict[str, dict]:
    """Last price and daily % change per ticker (batch). Missing tickers are omitted."""
    df = _yf_download(tickers)
    if df is None or df.empty:
        return {}
    close = df["Close"]
    out: dict[str, dict] = {}
    for t in tickers:
        try:
            col = close[t] if hasattr(close, "columns") and t in close.columns else close
        except Exception:
            continue
        s = col.dropna() if hasattr(col, "dropna") else col
        if len(s) < 2:
            continue
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        if prev == 0:
            continue
        out[t] = {"price": round(last, 2), "change_pct": round((last - prev) / prev * 100, 2)}
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_adapters_quotes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add equity_research/data/adapters.py tests/test_adapters_quotes.py
git commit -m "feat(data): fetch_quotes batch adapter"
```

---

### Task A2: `core.quotes` (daily cache) + `resolve_ticker` + `core.resolve`

**Files:**
- Modify: `api/core.py`
- Test: `tests/api/test_quotes_resolve.py` (new)

- [ ] **Step 1: Write the failing tests (`tests/api/test_quotes_resolve.py`)**

```python
import api.core as core
from equity_research.data.resolve import resolve_ticker


def test_resolve_valid_ticker_passes_through():
    out = resolve_ticker("aapl", llm_guess=lambda q: "ZZZ", price_ok=lambda t: t == "AAPL")
    assert out == {"input": "aapl", "resolved": "AAPL", "corrected": False}


def test_resolve_corrects_typo_via_llm():
    out = resolve_ticker("APPL", llm_guess=lambda q: "AAPL", price_ok=lambda t: t == "AAPL")
    assert out == {"input": "APPL", "resolved": "AAPL", "corrected": True}


def test_resolve_none_when_llm_none_or_invalid():
    assert resolve_ticker("ZZZZ", llm_guess=lambda q: None, price_ok=lambda t: False)["resolved"] is None
    assert resolve_ticker("ZZZZ", llm_guess=lambda q: "QQQQ", price_ok=lambda t: t == "AAPL")["resolved"] is None


def test_quotes_caches_by_day(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(tickers):
        calls["n"] += 1
        return {t: {"price": 1.0, "change_pct": 0.5} for t in tickers}

    monkeypatch.setattr("equity_research.data.adapters.fetch_quotes", fake_fetch)
    core._QUOTES_CACHE.clear()
    a = core.quotes(["AAPL", "MSFT"])
    b = core.quotes(["AAPL", "MSFT"])  # served from cache
    assert calls["n"] == 1
    assert a == b == [{"ticker": "AAPL", "price": 1.0, "change_pct": 0.5},
                      {"ticker": "MSFT", "price": 1.0, "change_pct": 0.5}]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/api/test_quotes_resolve.py -v`
Expected: FAIL (`equity_research.data.resolve` and `core.quotes`/`core._QUOTES_CACHE` missing).

- [ ] **Step 3a: Create `equity_research/data/resolve.py`**

```python
from __future__ import annotations

from typing import Callable


def resolve_ticker(query: str, llm_guess: Callable[[str], str | None],
                   price_ok: Callable[[str], bool]) -> dict:
    """Resolve a raw query to a valid ticker.

    Valid as-typed -> pass through. Else the LLM proposes a candidate which is
    accepted only if it has price data (guards against hallucinated symbols).
    Otherwise resolved is None (unknown).
    """
    q = query.strip().upper()
    if not q:
        return {"input": query, "resolved": None, "corrected": False}
    if price_ok(q):
        return {"input": query, "resolved": q, "corrected": False}
    cand = llm_guess(query)
    cand = cand.strip().upper() if cand else None
    if cand and cand != q and price_ok(cand):
        return {"input": query, "resolved": cand, "corrected": True}
    return {"input": query, "resolved": None, "corrected": False}
```

- [ ] **Step 3b: Add to `api/core.py`**

```python
_QUOTES_CACHE: dict = {}  # {iso_date: {ticker: {price, change_pct} | None}}


def quotes(tickers: list[str]) -> list[dict]:
    from datetime import date

    from equity_research.data import adapters

    key = date.today().isoformat()
    cache = _QUOTES_CACHE.setdefault(key, {})
    missing = [t for t in tickers if t not in cache]
    if missing:
        try:
            got = adapters.fetch_quotes(missing)
        except Exception:
            got = {}
        for t in missing:
            cache[t] = got.get(t)  # store None for failures so we don't refetch all day
    return [{"ticker": t, **cache[t]} for t in tickers if cache.get(t)]


_RESOLVE_PROMPT = (
    "The user typed '{q}' as a US stock ticker or company. "
    "Reply with ONLY the single most likely valid US stock ticker symbol in uppercase, "
    "or NONE if you cannot tell. No other words."
)


def resolve(query: str) -> dict:
    from datetime import date

    from equity_research.config import Config
    from equity_research.data.adapters import fetch_stooq, fetch_yfinance
    from equity_research.data.prices import PriceProvider
    from equity_research.data.resolve import resolve_ticker
    from equity_research.llm.cache import DiskCache
    from equity_research.llm.ollama_client import OllamaClient
    from equity_research.util.resilient import resilient

    cfg = Config.load(CONFIG_PATH)
    prices = PriceProvider(fetch_yfinance=resilient(fetch_yfinance, cfg.net),
                           fetch_stooq=resilient(fetch_stooq, cfg.net))

    def price_ok(t: str) -> bool:
        try:
            prices.history(t, date.today())
            return True
        except Exception:
            return False

    from ollama import Client

    chat_fn = Client(host=cfg.ollama_host, timeout=cfg.net["ollama_timeout"]).chat
    client = OllamaClient(model=cfg.model, cache=DiskCache(cfg.cache_dir),
                          seed=cfg.seed, temperature=cfg.temperature, chat_fn=chat_fn)

    def llm_guess(q: str) -> str | None:
        raw = client.generate_text(_RESOLVE_PROMPT.format(q=q)).strip().upper()
        tok = raw.split()[0] if raw else ""
        return None if tok in ("", "NONE") else tok

    return resolve_ticker(query, llm_guess, price_ok)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/api/test_quotes_resolve.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add equity_research/data/resolve.py api/core.py tests/api/test_quotes_resolve.py
git commit -m "feat(api): quotes daily cache and LLM ticker resolution seam"
```

---

### Task A3: `/api/quotes` + `/api/resolve` endpoints

**Files:**
- Modify: `api/main.py`
- Test: `tests/api/test_quotes_resolve.py` (append)

- [ ] **Step 1: Append endpoint tests**

```python
from fastapi.testclient import TestClient

import api.main as main


def test_quotes_endpoint(monkeypatch):
    monkeypatch.setattr(core, "quotes", lambda tickers: [{"ticker": tickers[0], "price": 1.0, "change_pct": 2.0}])
    client = TestClient(main.app)
    resp = client.get("/api/quotes?tickers=aapl,msft")
    assert resp.status_code == 200
    assert resp.json()[0]["ticker"] == "AAPL"


def test_resolve_endpoint(monkeypatch):
    monkeypatch.setattr(core, "resolve", lambda q: {"input": q, "resolved": "AAPL", "corrected": True})
    client = TestClient(main.app)
    assert client.get("/api/resolve?query=APPL").json()["resolved"] == "AAPL"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/api/test_quotes_resolve.py -v -k endpoint`
Expected: FAIL (404).

- [ ] **Step 3: Add routes to `api/main.py`** (after `/api/prices`)

```python
@app.get("/api/quotes")
def quotes(tickers: str):
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return core.quotes(syms)


@app.get("/api/resolve")
def resolve(query: str):
    return core.resolve(query)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/api/test_quotes_resolve.py -v`
Expected: all pass.

- [ ] **Step 5: Full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add api/main.py tests/api/test_quotes_resolve.py
git commit -m "feat(api): /api/quotes and /api/resolve endpoints"
```

---

### Task A4: Live verification (backend)

- [ ] Start API: `uv run uvicorn api.main:app --port 8000`
- [ ] `curl -s "http://localhost:8000/api/quotes?tickers=AAPL,MSFT,NVDA" | python -m json.tool` → 3 objects with price + change_pct.
- [ ] `curl -s "http://localhost:8000/api/resolve?query=APPL"` → `{"input":"APPL","resolved":"AAPL","corrected":true}` (LLM correction, validated).
- [ ] `curl -s "http://localhost:8000/api/resolve?query=AAPL"` → `resolved":"AAPL","corrected":false`.
- [ ] `curl -s "http://localhost:8000/api/resolve?query=ZZZZ"` → `"resolved":null`.
- [ ] Stop server. No commit.

Merge `feature/p1-quotes-resolve` into `master` via **superpowers:finishing-a-development-branch** before Part B.

---

## Part B — Frontend

### Task B1: TOP50 list + client + types

**Files:**
- Create: `src/ui/top50.ts`
- Modify: `src/api/types.ts`
- Modify: `src/api/client.ts`
- Create: `src/api/quotes.test.ts`

- [ ] **Step 1: Create `src/ui/top50.ts`**

```ts
// Curated large-cap universe for the ticker tape.
export const TOP50 = [
  'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM',
  'V', 'XOM', 'UNH', 'MA', 'JNJ', 'PG', 'HD', 'COST', 'ORCL', 'MRK',
  'ABBV', 'CVX', 'KO', 'PEP', 'ADBE', 'WMT', 'BAC', 'CRM', 'NFLX', 'AMD',
  'TMO', 'MCD', 'CSCO', 'ACN', 'ABT', 'DHR', 'LIN', 'WFC', 'DIS', 'TXN',
  'PM', 'VZ', 'INTC', 'QCOM', 'IBM', 'CAT', 'GE', 'NKE', 'NOW', 'TMUS',
]
```

- [ ] **Step 2: Add types to `src/api/types.ts`**

```ts
export interface Quote { ticker: string; price: number; change_pct: number }
export interface Resolve { input: string; resolved: string | null; corrected: boolean }
```

- [ ] **Step 3: Add client calls to `src/api/client.ts`**

```ts
import type { Quote, Resolve } from './types'

export function getQuotes(tickers: string[]): Promise<Quote[]> {
  return getJSON<Quote[]>(`/api/quotes?tickers=${encodeURIComponent(tickers.join(','))}`)
}

export function getResolve(query: string): Promise<Resolve> {
  return getJSON<Resolve>(`/api/resolve?query=${encodeURIComponent(query)}`)
}
```

- [ ] **Step 4: Write test (`src/api/quotes.test.ts`)**

```ts
import { getQuotes, getResolve } from './client'

test('getQuotes joins tickers', async () => {
  const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(JSON.stringify([{ ticker: 'AAPL', price: 1, change_pct: 2 }]), { status: 200 }) as any)
  const out = await getQuotes(['AAPL', 'MSFT'])
  expect(spy).toHaveBeenCalledWith('/api/quotes?tickers=AAPL%2CMSFT')
  expect(out[0].ticker).toBe('AAPL')
  spy.mockRestore()
})

test('getResolve passes the query', async () => {
  const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(JSON.stringify({ input: 'APPL', resolved: 'AAPL', corrected: true }), { status: 200 }) as any)
  const out = await getResolve('APPL')
  expect(out.resolved).toBe('AAPL')
  spy.mockRestore()
})
```

- [ ] **Step 5: Run + commit**

Run: `npm test -- src/api/quotes.test.ts` → PASS.

```bash
git add src/ui/top50.ts src/api/types.ts src/api/client.ts src/api/quotes.test.ts
git commit -m "feat(api): quotes and resolve client + top50 list"
```

---

### Task B2: `TickerTape` marquee

**Files:**
- Create: `src/components/TickerTape.tsx`
- Create: `src/components/TickerTape.test.tsx`

- [ ] **Step 1: Write failing test**

```tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import * as client from '../api/client'
import { TickerTape } from './TickerTape'

test('renders quotes and reports a click', async () => {
  vi.spyOn(client, 'getQuotes').mockResolvedValue([
    { ticker: 'AAPL', price: 210.5, change_pct: 1.2 },
    { ticker: 'MSFT', price: 400.0, change_pct: -0.8 },
  ])
  const onSelect = vi.fn()
  render(<TickerTape onSelect={onSelect} />)
  const aapl = await screen.findAllByText(/AAPL/)
  await userEvent.click(aapl[0])
  expect(onSelect).toHaveBeenCalledWith('AAPL')
})
```

- [ ] **Step 2: Run to verify fail** — `npm test -- src/components/TickerTape.test.tsx` (module not found).

- [ ] **Step 3: Implement `src/components/TickerTape.tsx`**

```tsx
import { useEffect, useState } from 'react'
import { getQuotes } from '../api/client'
import type { Quote } from '../api/types'
import { TOP50 } from '../ui/top50'

export function TickerTape({ onSelect }: { onSelect: (t: string) => void }) {
  const [quotes, setQuotes] = useState<Quote[] | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let alive = true
    getQuotes(TOP50)
      .then((q) => { if (alive) setQuotes(q) })
      .catch(() => { if (alive) setFailed(true) })
    return () => { alive = false }
  }, [])

  if (failed) return <div className="text-xs" style={{ color: 'var(--text-mut)' }}>quotes unavailable</div>
  if (!quotes || quotes.length === 0) return <div className="h-8" />

  const row = (key: string) => (
    <div key={key} className="flex shrink-0 gap-4 pr-4" aria-hidden={key === 'b'}>
      {quotes.map((q) => {
        const up = q.change_pct >= 0
        return (
          <button key={key + q.ticker} onClick={() => onSelect(q.ticker)}
            className="flex shrink-0 items-center gap-2 rounded-md px-2 py-1 text-xs tnum"
            style={{ background: 'var(--surface-2)', border: '1px solid var(--border-soft)' }}>
            <span className="font-semibold" style={{ color: 'var(--text)' }}>{q.ticker}</span>
            <span style={{ color: 'var(--text-dim)' }}>${q.price.toFixed(2)}</span>
            <span style={{ color: up ? 'var(--bull)' : 'var(--bear)' }}>
              {up ? '▲' : '▼'} {Math.abs(q.change_pct).toFixed(2)}%
            </span>
          </button>
        )
      })}
    </div>
  )

  return (
    <div className="tape-mask overflow-hidden rounded-lg" style={{ border: '1px solid var(--border-soft)' }}>
      <div className="tape-track flex w-max">
        {row('a')}
        {row('b')}
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Add the marquee CSS to `src/index.css`**

```css
.tape-track { animation: tape-scroll 60s linear infinite; }
@keyframes tape-scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.tape-track:hover { animation-play-state: paused; }
@media (prefers-reduced-motion: reduce) {
  .tape-track { animation: none; }
  .tape-mask { overflow-x: auto; }
}
```

- [ ] **Step 5: Run to verify pass** — `npm test -- src/components/TickerTape.test.tsx` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/components/TickerTape.tsx src/components/TickerTape.test.tsx src/index.css
git commit -m "feat(ui): scrolling ticker tape"
```

---

### Task B3: Wire tape + resolve into Analyze

**Files:**
- Modify: `src/pages/Analyze.tsx`
- Modify: `src/pages/Analyze.test.tsx` (append)

- [ ] **Step 1: Append a failing test to `src/pages/Analyze.test.tsx`**

```tsx
import { vi } from 'vitest'
import userEvent from '@testing-library/user-event'
import * as client from '../api/client'

vi.mock('../components/TickerTape', () => ({ TickerTape: () => <div>tape</div> }))
vi.mock('../components/PricePanel', () => ({ PricePanel: () => <div>price-panel</div> }))

test('corrects a mistyped ticker before analyzing', async () => {
  vi.spyOn(client, 'getResolve').mockResolvedValue({ input: 'APPL', resolved: 'AAPL', corrected: true })
  const stream = vi.spyOn(client, 'streamSSE').mockReturnValue({ close() {} } as any)
  render(<Analyze />)
  await userEvent.type(screen.getByPlaceholderText(/ticker/i), 'APPL')
  await userEvent.click(screen.getByRole('button', { name: /go/i }))
  expect(await screen.findByText(/AAPL/)).toBeInTheDocument()  // correction note
  expect(stream).toHaveBeenCalledWith(expect.stringContaining('ticker=AAPL'), expect.anything(), expect.anything())
})

test('shows not-found for an unresolvable query', async () => {
  vi.spyOn(client, 'getResolve').mockResolvedValue({ input: 'ZZZZ', resolved: null, corrected: false })
  render(<Analyze />)
  await userEvent.type(screen.getByPlaceholderText(/ticker/i), 'ZZZZ')
  await userEvent.click(screen.getByRole('button', { name: /go/i }))
  expect(await screen.findByText(/not.*found|no such|не знайшл/i)).toBeInTheDocument()
})
```

(If the existing top-of-file test lacks these imports/mocks, merge them in without removing the existing `test('analyze page shows a ticker input')`.)

- [ ] **Step 2: Run to verify fail** — `npm test -- src/pages/Analyze.test.tsx`.

- [ ] **Step 3: Update `src/pages/Analyze.tsx`**

Add the tape at top, and a resolve step in `run`. Replace the component with:

```tsx
import { useMemo, useRef, useState } from 'react'
import { getResolve, streamSSE } from '../api/client'
import type { AgentEvent, Verdict } from '../api/types'
import { AgentCard } from '../components/AgentCard'
import { VerdictCard } from '../components/VerdictCard'
import { TickerInput } from '../components/TickerInput'
import { ProgressRail } from '../components/ProgressRail'
import { PricePanel } from '../components/PricePanel'
import { TickerTape } from '../components/TickerTape'

export function Analyze() {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [verdict, setVerdict] = useState<Verdict | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [ticker, setTicker] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  const done = useMemo(() => new Set(events.map((e) => e.agent)), [events])

  const analyze = (t: string) => {
    esRef.current?.close()
    setEvents([]); setVerdict(null); setError(null); setRunning(true); setTicker(t)
    esRef.current = streamSSE(`/api/analyze?ticker=${encodeURIComponent(t)}`, {
      agent: (d) => setEvents((prev) => [...prev, d]),
      verdict: (d) => setVerdict(d),
      error: (d) => setError(d.message ?? 'error'),
    }, () => setRunning(false))
  }

  const run = async (query: string) => {
    setNotice(null); setError(null)
    let r
    try {
      r = await getResolve(query)
    } catch {
      setError('Could not verify the ticker — try again.')
      return
    }
    if (!r.resolved) {
      setTicker(null); setVerdict(null); setEvents([])
      setNotice(`No such stock found for “${query}” — the name may be incorrect.`)
      return
    }
    if (r.corrected) setNotice(`Showing ${r.resolved} (you typed ${r.input}).`)
    analyze(r.resolved)
  }

  return (
    <div className="space-y-4">
      <TickerTape onSelect={run} />
      <TickerInput onSubmit={run} disabled={running} />
      {notice && <div className="text-sm" style={{ color: 'var(--accent)' }}>{notice}</div>}
      {error && <div className="text-sm" style={{ color: 'var(--bear)' }}>{error}</div>}
      {ticker && <PricePanel ticker={ticker} />}
      {(running || events.length > 0) && !verdict && <ProgressRail done={done} running={running} />}
      {!verdict && <div className="space-y-3">{events.map((e, i) => <AgentCard key={i} event={e} />)}</div>}
      {verdict && <VerdictCard verdict={verdict} />}
    </div>
  )
}
```

- [ ] **Step 4: Run to verify pass** — `npm test -- src/pages/Analyze.test.tsx` → PASS.

- [ ] **Step 5: Full suite + build** — `npm test && npm run build` → all pass.

- [ ] **Step 6: Commit**

```bash
git add src/pages/Analyze.tsx src/pages/Analyze.test.tsx
git commit -m "feat(ui): ticker tape + resolve-before-analyze flow"
```

---

### Task B4: Live verification (frontend)

- [ ] Start backend + `npm run dev`.
- [ ] Tape scrolls RTL with 50 stocks, price + colored %change; hover pauses; click a tile → analyzes it.
- [ ] Type `APPL` → notice "Showing AAPL (you typed APPL)" then normal analysis of AAPL.
- [ ] Type `ZZZZ` → "No such stock found…", no analysis.
- [ ] Type `AAPL` → analyzes directly (no notice).
- [ ] Mobile width: tape and layout usable, no horizontal page scroll (tape scrolls within its own container).
- [ ] Stop servers. No commit.

---

## Completion
Merge both branches via **superpowers:finishing-a-development-branch** (backend first). All commits authored as the repo owner, NO `Co-Authored-By`; client repo carries no Claude/AI references.
