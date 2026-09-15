# Analysis integrity fixes (A–E) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the system from producing confident verdicts when data is missing (unknown/invalid ticker, transient failures), make sentiment news ticker-relevant, harden embed calls, and add a timeframe loading state.

**Architecture:** Backend gains a `Verdict.status` contract (`ok|unknown_ticker|insufficient_data`); a pre-flight price check short-circuits unknown tickers; the aggregator requires a price-based directional agent; sentiment filters news by company relevance; the vector store retries embed ops. Frontend renders the new statuses and a timeframe loader.

**Tech Stack:** Python + pydantic + pytest (backend); React + TS + Vitest (frontend).

**Two repos / two branches:**
- **Part 0** — backend repo `/Users/oleksandr/Documents/LLM/Multi-Agent Equity`, branch `feature/analysis-integrity`.
- **Part 1** — client repo `/Users/oleksandr/Documents/LLM/Multi-Agent Equity Client`, branch `feature/analysis-integrity-ui`.

**Attribution (BOTH):** commit ONLY as the repo author, NO `Co-Authored-By`. Client repo: no Claude/AI/assistant/"generated" references anywhere.

---

## Part 0 — Backend

### Task 0.1: `Verdict.status` + aggregator insufficient-data rule (B)

**Files:**
- Modify: `equity_research/orchestration/aggregator.py`
- Test: `tests/test_aggregator.py` (append)

- [ ] **Step 1: Write the failing test (append to `tests/test_aggregator.py`)**

```python
def test_insufficient_data_when_no_price_directional_agent():
    from datetime import date

    from equity_research.agents.base import AgentOpinion
    from equity_research.config import Config
    from equity_research.orchestration.aggregator import Aggregator

    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".c",
                 edgar_user_agent="x", weights={"fundamentals": 1, "technical": 1, "sentiment": 1})

    class FakeClient:
        def generate_text(self, prompt):  # narrative not used on this path
            return "n"

    agg = Aggregator(cfg, FakeClient())
    # only sentiment produced an opinion; fundamentals+technical skipped
    sentiment = AgentOpinion(agent="sentiment", stance="neutral", score=0.0, confidence=0.8, rationale="r")
    verdict = agg.aggregate("AAPL", date(2026, 9, 15), [sentiment],
                            skipped=["fundamentals", "technical"], skip_reasons={})
    assert verdict.status == "insufficient_data"
    assert verdict.confidence == 0.0
    assert verdict.opinions == [sentiment]  # what ran is still shown


def test_status_ok_when_price_directional_present():
    from datetime import date

    from equity_research.agents.base import AgentOpinion
    from equity_research.config import Config
    from equity_research.orchestration.aggregator import Aggregator

    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".c",
                 edgar_user_agent="x", weights={"fundamentals": 1, "technical": 1, "sentiment": 1})

    class FakeClient:
        def generate_text(self, prompt):
            return "narrative"

    agg = Aggregator(cfg, FakeClient())
    tech = AgentOpinion(agent="technical", stance="bullish", score=0.6, confidence=0.8, rationale="r")
    verdict = agg.aggregate("AAPL", date(2026, 9, 15), [tech], skipped=[], skip_reasons={})
    assert verdict.status == "ok"
    assert verdict.verdict == "buy"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_aggregator.py -v -k "insufficient or status_ok"`
Expected: FAIL (`Verdict` has no `status`).

- [ ] **Step 3a: Add `status` to `Verdict` in `equity_research/orchestration/aggregator.py`**

Add the import if missing (`Literal` is already imported) and the field (after `verdict`):

```python
    status: Literal["ok", "unknown_ticker", "insufficient_data"] = "ok"
```

- [ ] **Step 3b: Add the price-directional rule in `Aggregator.aggregate`**

Add near the top of the module (after `DIRECTIONAL_AGENTS`):

```python
PRICE_DIRECTIONAL = {"fundamentals", "technical"}
```

In `aggregate`, replace the current empty-directional early return:

```python
        if not directional:
            return Verdict(
                ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                confidence=0.0, narrative="No agent produced an opinion; no data available.",
                opinions=[o for o in [risk_op] if o], skipped_agents=skipped,
                skip_reasons=skip_reasons,
            )
```

with a check that requires a price-based directional agent:

```python
        price_directional = [o for o in directional if o.agent in PRICE_DIRECTIONAL]
        if not price_directional:
            return Verdict(
                ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                confidence=0.0, status="insufficient_data",
                narrative="Insufficient data: no price-based analysis available for this ticker.",
                opinions=opinions, skipped_agents=skipped, skip_reasons=skip_reasons,
            )
```

(`opinions` is the full input list, so any sentiment/risk opinions that ran are still shown. The normal weighted path below is unchanged and implicitly `status="ok"`.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_aggregator.py -v`
Expected: all pass (update any pre-existing test that asserted the old "no agent produced an opinion" narrative for the sentiment-only / empty case — the new behavior is `insufficient_data`; adjust its assertion to the new narrative/status).

- [ ] **Step 5: Commit**

```bash
git add equity_research/orchestration/aggregator.py tests/test_aggregator.py
git commit -m "feat(aggregator): insufficient_data status when no price-based agent"
```

---

### Task 0.2: Unknown-ticker pre-flight (A)

**Files:**
- Modify: `equity_research/cli.py`
- Test: `tests/test_cli.py` (append)

- [ ] **Step 1: Write the failing test (append to `tests/test_cli.py`)**

```python
def test_unknown_ticker_verdict_short_circuits():
    from datetime import date

    from equity_research.cli import _unknown_ticker_verdict
    from equity_research.data.prices import PriceValidationError

    class FakePrices:
        def history(self, ticker, as_of):
            raise PriceValidationError(f"no price history for {ticker}")

    v = _unknown_ticker_verdict(FakePrices(), "APPL", date(2026, 9, 15))
    assert v is not None
    assert v.status == "unknown_ticker"
    assert v.opinions == []


def test_known_ticker_preflight_returns_none():
    from datetime import date

    from equity_research.cli import _unknown_ticker_verdict

    class FakePrices:
        def history(self, ticker, as_of):
            return ("df", None)  # any non-raising result

    assert _unknown_ticker_verdict(FakePrices(), "AAPL", date(2026, 9, 15)) is None
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_cli.py -v -k "unknown_ticker or preflight"`
Expected: FAIL (`_unknown_ticker_verdict` not defined).

- [ ] **Step 3a: Add the helper in `equity_research/cli.py`**

Add near the top (after imports). `Verdict` and `PriceValidationError` must be importable here — add imports if missing:

```python
from equity_research.data.prices import PriceValidationError
from equity_research.orchestration.aggregator import Verdict


def _unknown_ticker_verdict(prices, ticker: str, as_of) -> "Verdict | None":
    """Return an unknown-ticker Verdict if there is no price history, else None.

    A ticker with no price data (e.g. a typo like APPL) cannot be valued; we
    short-circuit before running any agent so we don't surface generic,
    misleading news for a symbol that does not exist.
    """
    try:
        prices.history(ticker, as_of)
        return None
    except PriceValidationError:
        return Verdict(
            ticker=ticker, as_of=as_of, verdict="hold", score=0.0, confidence=0.0,
            status="unknown_ticker",
            narrative=f"No price data for {ticker}; it may be an unknown or delisted ticker.",
            opinions=[],
        )
```

- [ ] **Step 3b: Wire it into `analyze_ticker`** — immediately before `orch = Orchestrator(...)` / `return orch.run(...)`:

```python
    unknown = _unknown_ticker_verdict(prices, ticker, as_of)
    if unknown is not None:
        return unknown
    orch = Orchestrator(agents=agents, aggregator=Aggregator(cfg, client))
    return orch.run(ticker, as_of, on_event=on_event)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add equity_research/cli.py tests/test_cli.py
git commit -m "feat(analyze): short-circuit unknown tickers before running agents"
```

---

### Task 0.3: Sentiment news relevance filter (C)

**Files:**
- Create: `equity_research/data/company.py`
- Modify: `equity_research/agents/sentiment.py`
- Modify: `equity_research/cli.py` (wire `name_fn`)
- Test: `tests/test_sentiment_agent.py` (append)

- [ ] **Step 1: Write the failing tests (append to `tests/test_sentiment_agent.py`)**

```python
def test_sentiment_filters_news_by_company_relevance():
    from datetime import date

    from equity_research.agents.sentiment import SentimentAgent

    class FakeRetriever:
        def retrieve(self, ticker, query, as_of, k, candidate_k):
            return ["Apple unveils new iPhone", "Generic market wrap: indexes edge lower"]

    agent = SentimentAgent(retriever=FakeRetriever(), ingest_fn=lambda t: None,
                           client=None, name_fn=lambda t: "Apple Inc")
    ev = agent.gather("AAPL", date(2026, 9, 15))
    assert ev.context == ["Apple unveils new iPhone"]  # generic wrap dropped


def test_sentiment_keeps_all_when_no_company_name():
    from datetime import date

    from equity_research.agents.sentiment import SentimentAgent

    class FakeRetriever:
        def retrieve(self, ticker, query, as_of, k, candidate_k):
            return ["Generic market wrap", "Another headline"]

    agent = SentimentAgent(retriever=FakeRetriever(), ingest_fn=lambda t: None,
                           client=None, name_fn=lambda t: None)
    ev = agent.gather("AAPL", date(2026, 9, 15))
    assert ev.context == ["Generic market wrap", "Another headline"]  # no name -> no filtering
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_sentiment_agent.py -v -k "relevance or company"`
Expected: FAIL (`SentimentAgent` has no `name_fn`).

- [ ] **Step 3a: Create `equity_research/data/company.py`**

```python
from __future__ import annotations


def company_name(ticker: str) -> str | None:
    """Best-effort company name for a ticker via yfinance; None on any failure.

    Used only to filter news for relevance, so it must never raise.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        name = info.get("shortName") or info.get("longName")
        return name or None
    except Exception:
        return None
```

- [ ] **Step 3b: Add the filter + `name_fn` in `equity_research/agents/sentiment.py`**

Add a module-level helper and thread `name_fn` through:

```python
def _filter_relevant(texts: list[str], ticker: str, name: str | None) -> list[str]:
    """Keep only news mentioning the ticker or company name.

    When the company name is unavailable we cannot reliably tell relevance, so
    we keep everything rather than risk dropping valid news.
    """
    if not name:
        return texts
    needles = [ticker.lower(), name.split()[0].lower()]
    return [t for t in texts if any(n in t.lower() for n in needles)]
```

Update `__init__` signature to accept `name_fn=None` (store it), and in `gather`, after retrieval:

```python
    def gather(self, ticker: str, as_of: date) -> Evidence:
        self.ingest_fn(ticker)
        texts = self.retriever.retrieve(ticker, _QUERY, as_of, self.k, self.candidate_k)
        name = None
        if self.name_fn is not None:
            try:
                name = self.name_fn(ticker)
            except Exception:
                name = None
        texts = _filter_relevant(texts, ticker, name)
        notes = [] if texts else ["no news retrieved"]
        return Evidence(ticker=ticker, as_of=as_of, metrics={}, context=texts, notes=notes)
```

Constructor (add `name_fn` param, keep others):

```python
    def __init__(self, retriever, ingest_fn, client, k=6, candidate_k=20, name_fn=None):
        self.retriever = retriever
        self.ingest_fn = ingest_fn
        self.client = client
        self.k = k
        self.candidate_k = candidate_k
        self.name_fn = name_fn
```

- [ ] **Step 3c: Wire `name_fn` in `equity_research/cli.py`** — where `SentimentAgent(...)` is constructed, add `name_fn=company_name` (import `from equity_research.data.company import company_name`).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_sentiment_agent.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add equity_research/data/company.py equity_research/agents/sentiment.py equity_research/cli.py tests/test_sentiment_agent.py
git commit -m "feat(sentiment): filter news by ticker/company relevance"
```

---

### Task 0.4: Vector-store embed resilience (D)

**Files:**
- Modify: `equity_research/rag/chroma_store.py`
- Modify: `equity_research/cli.py` and `api/core.py` (pass `net`)
- Test: `tests/test_rag_store.py` (append)

- [ ] **Step 1: Write the failing test (append to `tests/test_rag_store.py`)**

```python
def test_chroma_store_retries_embed_ops():
    from equity_research.rag.chroma_store import ChromaVectorStore

    class FlakyDB:
        def __init__(self):
            self.calls = 0

        def max_marginal_relevance_search(self, query, k, filter):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection reset by peer")

            class Doc:
                page_content = "ok"
                metadata = {"id": "1"}

            return [Doc()]

    store = ChromaVectorStore.__new__(ChromaVectorStore)  # bypass real Chroma init
    store._db = FlakyDB()
    store._net = {"data_timeout": 5, "data_attempts": 3, "data_base_delay": 0}
    out = store.mmr_search("q", {"doc_type": {"$eq": "news"}}, k=1)
    assert out == [("ok", {"id": "1"})]
    assert store._db.calls == 2  # retried once


def test_chroma_store_no_net_calls_once():
    from equity_research.rag.chroma_store import ChromaVectorStore

    class DB:
        def __init__(self):
            self.calls = 0

        def get(self, ids):
            self.calls += 1
            return {"ids": ["a"]}

    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._db = DB()
    store._net = None
    assert store.existing_ids(["a"]) == {"a"}
    assert store._db.calls == 1
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_rag_store.py -v -k "retries or no_net"`
Expected: FAIL (`_net` / retry behavior absent).

- [ ] **Step 3: Update `equity_research/rag/chroma_store.py`**

```python
from __future__ import annotations

from equity_research.util.resilient import resilient_call


class ChromaVectorStore:
    """Thin Chroma wrapper implementing the VectorStore protocol.

    When a `net` config is supplied, embed-backed operations are retried, so a
    transient Ollama embedding failure (e.g. a reset /tokenize call) does not
    abort an agent.
    """

    def __init__(self, persist_dir: str, embed_model: str, collection: str = "news", net: dict | None = None):
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        self._db = Chroma(
            collection_name=collection,
            persist_directory=persist_dir,
            embedding_function=OllamaEmbeddings(model=embed_model),
        )
        self._net = net

    def _call(self, fn):
        if self._net is None:
            return fn()
        return resilient_call(fn, timeout=self._net["data_timeout"],
                              attempts=self._net["data_attempts"], base_delay=self._net["data_base_delay"])

    def existing_ids(self, ids: list[str]) -> set[str]:
        return self._call(lambda: set(self._db.get(ids=ids).get("ids", [])))

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        self._call(lambda: self._db.add_texts(texts=texts, metadatas=metadatas, ids=ids))

    def mmr_search(self, query: str, where: dict, k: int) -> list[tuple[str, dict]]:
        return self._call(lambda: [(d.page_content, d.metadata)
                                   for d in self._db.max_marginal_relevance_search(query, k=k, filter=where)])
```

- [ ] **Step 3b: Pass `net` at construction sites**

In `equity_research/cli.py` (`analyze_ticker`) and `api/core.py` (`ingest_ticker`), change `ChromaVectorStore(persist_dir=..., embed_model=...)` to also pass `net=cfg.net`. (Search: `grep -rn "ChromaVectorStore(" equity_research api` and add `net=cfg.net` to each call that has a `cfg` in scope.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_rag_store.py -v`
Expected: all pass.

- [ ] **Step 5: Full suite**

Run: `uv run pytest -q`
Expected: all pass (fix any aggregator test that asserted old empty-directional behavior).

- [ ] **Step 6: Commit**

```bash
git add equity_research/rag/chroma_store.py equity_research/cli.py api/core.py tests/test_rag_store.py
git commit -m "feat(rag): retry embed operations on transient failures"
```

---

### Task 0.5: Live verification (backend)

- [ ] Start API: `uv run uvicorn api.main:app --port 8000 --log-level warning`
- [ ] Unknown ticker: `curl -s "http://localhost:8000/api/analyze?ticker=APPL"` → the `verdict` event has `"status": "unknown_ticker"` and empty `opinions` (agents did NOT run — no `event: agent` lines).
- [ ] Known ticker: `curl -s "http://localhost:8000/api/analyze?ticker=AAPL"` → agents stream, verdict `"status": "ok"`.
- [ ] Stop the server. No commit.

Merge `feature/analysis-integrity` into `master` via **superpowers:finishing-a-development-branch** before the frontend part (frontend consumes `status`).

---

## Part 1 — Frontend

### Task 1.1: `Verdict.status` type + VerdictCard branches (A/B display)

**Files:**
- Modify: `src/api/types.ts`
- Modify: `src/components/VerdictCard.tsx`
- Modify: `src/components/VerdictCard.test.tsx` (append cases)

- [ ] **Step 1: Add `status` to the `Verdict` type in `src/api/types.ts`**

Add (after `verdict`):

```ts
  status?: 'ok' | 'unknown_ticker' | 'insufficient_data'
```

(Optional so existing fixtures without it still typecheck; the backend always sends it.)

- [ ] **Step 2: Append failing tests to `src/components/VerdictCard.test.tsx`**

```tsx
test('renders unknown-ticker state without a verdict badge', () => {
  render(<VerdictCard verdict={{ ...v, status: 'unknown_ticker', narrative: 'No price data for ZZZZ.', opinions: [] }} />)
  expect(screen.getByText(/unknown or delisted|no price data|unknown ticker/i)).toBeInTheDocument()
  expect(screen.queryByText(/^BUY$/i)).toBeNull()
})

test('renders insufficient-data state', () => {
  render(<VerdictCard verdict={{ ...v, status: 'insufficient_data', opinions: [] }} />)
  expect(screen.getByText(/insufficient data/i)).toBeInTheDocument()
})
```

(`v` is the existing fixture at the top of the file; it has `verdict: 'buy'`. Spreading overrides `status`.)

- [ ] **Step 3: Run to verify fail**

Run: `npm test -- src/components/VerdictCard.test.tsx`
Expected: FAIL (no status handling yet).

- [ ] **Step 4: Add status branches at the top of `VerdictCard` (before the normal return) in `src/components/VerdictCard.tsx`**

```tsx
  if (verdict.status === 'unknown_ticker' || verdict.status === 'insufficient_data') {
    const heading = verdict.status === 'unknown_ticker' ? 'Unknown ticker' : 'Insufficient data'
    return (
      <div className="space-y-3 animate-fade-in">
        <div className="rounded-xl border p-5" style={{ background: 'var(--surface)', borderColor: 'var(--neutral)' }}>
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold" style={{ color: 'var(--text)' }}>{verdict.ticker}</h2>
            <span className="rounded-lg px-3 py-1 text-sm font-bold uppercase"
              style={{ color: 'var(--neutral)', background: 'color-mix(in srgb, var(--neutral) 18%, transparent)' }}>
              {heading}
            </span>
          </div>
          <p className="mt-3 text-sm" style={{ color: 'var(--text-dim)' }}>{verdict.narrative}</p>
          {verdict.skipped_agents.length > 0 && (
            <div className="mt-3 text-xs" style={{ color: 'var(--text-mut)' }}>
              {verdict.skipped_agents.map((a) => `${a}${verdict.skip_reasons[a] ? ` (${verdict.skip_reasons[a]})` : ''}`).join(', ')}
            </div>
          )}
        </div>
        {verdict.opinions.map((o) => <AgentCard key={o.agent} event={{ agent: o.agent, opinion: o }} />)}
      </div>
    )
  }
```

- [ ] **Step 5: Run to verify pass**

Run: `npm test -- src/components/VerdictCard.test.tsx`
Expected: PASS (existing "buy" test still green — it has no `status`, so it takes the normal path via `status ?? 'ok'`).

- [ ] **Step 6: Commit**

```bash
git add src/api/types.ts src/components/VerdictCard.tsx src/components/VerdictCard.test.tsx
git commit -m "feat(ui): render unknown-ticker and insufficient-data states"
```

---

### Task 1.2: PricePanel timeframe loader (E)

**Files:**
- Modify: `src/components/PricePanel.tsx`
- Modify: `src/components/PricePanel.test.tsx` (append)

- [ ] **Step 1: Append failing test to `src/components/PricePanel.test.tsx`**

```tsx
test('shows a loading indicator while fetching', async () => {
  let resolve: (v: any) => void = () => {}
  vi.spyOn(client, 'getPrices').mockReturnValue(new Promise((r) => { resolve = r }))
  render(<PricePanel ticker="AAPL" />)
  expect(screen.getByTestId('prices-loading')).toBeInTheDocument()
  resolve({ ticker: 'AAPL', period: '6M', candles: [{ time: '2026-01-02', value: 1 }], sma50: [], sma200: [], rsi: [] })
  expect(await screen.findByText('price-chart')).toBeInTheDocument()
})
```

(The existing mocks of `./PriceChart` and `./RsiChart` at the top of this test file remain.)

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- src/components/PricePanel.test.tsx`
Expected: FAIL (no `prices-loading` testid).

- [ ] **Step 3: Add a `loading` flag to `src/components/PricePanel.tsx`**

Track loading around the fetch and render a thin indicator; keep any existing chart visible but dimmed while refetching:

```tsx
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let alive = true
    setError(null); setLoading(true)
    getPrices(ticker, period)
      .then((p) => { if (alive) setPrices(p) })
      .catch((e) => { if (alive) setError((e as Error).message) })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [ticker, period])
```

Add the indicator inside the Card body (above the chart block); when reloading with an existing chart, dim it:

```tsx
      {loading && <div data-testid="prices-loading" className="h-0.5 w-full animate-pulse-soft rounded" style={{ background: 'var(--accent)' }} />}
      {!error && !prices && !loading && <div className="h-[260px] rounded-lg" style={{ background: 'var(--surface-2)' }} />}
```

And wrap the existing chart block so it dims while loading:

```tsx
      {!error && prices && prices.candles.length > 0 && (
        <div className="space-y-2" style={{ opacity: loading ? 0.5 : 1 }}>
          ...existing PriceChart + RSI...
        </div>
      )}
```

(Keep the existing error and empty-`candles` branches.)

- [ ] **Step 4: Run to verify pass**

Run: `npm test -- src/components/PricePanel.test.tsx`
Expected: PASS.

- [ ] **Step 5: Full suite + build**

Run: `npm test && npm run build`
Expected: all pass, build succeeds.

- [ ] **Step 6: Commit**

```bash
git add src/components/PricePanel.tsx src/components/PricePanel.test.tsx
git commit -m "feat(ui): loading indicator on timeframe change"
```

---

### Task 1.3: Live verification (frontend)

- [ ] Start backend (`uv run uvicorn api.main:app --port 8000`) and `npm run dev`.
- [ ] `APPL` (typo): price panel shows "no data"; verdict area shows **Unknown ticker** panel, no buy/hold/sell badge, no agent cards.
- [ ] `AAPL`: normal flow (charts, agents, verdict). Switch timeframe → loading indicator flashes, chart updates.
- [ ] (If Ollama drops an embed) Fundamentals no longer skips on a single transient reset. Stop servers. No commit.

---

## Completion
- Merge each branch via **superpowers:finishing-a-development-branch**; backend first (frontend depends on `status`).
- All commits authored as the repo owner, NO `Co-Authored-By`; client repo carries no Claude/AI references.
