# Phase 4A — FastAPI backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit attribution:** commit ONLY as the repo author (OleksandrFIT). Do NOT add any `Co-Authored-By` trailer.

**Goal:** A thin FastAPI layer over the existing `equity_research` core exposing health, streaming analyze (SSE, per-agent progress), ingest, and streaming backtest — with an orchestrator `on_event` hook as the only core change.

**Architecture:** `Orchestrator.run` gains an optional `on_event` callback (called per agent). `api/core.py` holds thin real wiring (reuses `cli.analyze_ticker`, `build_backtest_verdict`) behind seam functions tests monkeypatch. `api/main.py` is the FastAPI app; SSE endpoints bridge the synchronous core to a stream via a background thread + queue. All endpoints are unit-tested with FastAPI `TestClient` and injected fakes (no LLM/network).

**Tech Stack:** existing + FastAPI + uvicorn (new optional `web` extra) + httpx (TestClient dep). No frontend here (Plan 4B, separate repo).

**Scope:** Plan 4A only. Spec: `docs/superpowers/specs/2026-09-15-phase-4-web-ui-design.md`.

## File Structure

- `equity_research/orchestration/orchestrator.py` — add `on_event` param.
- `equity_research/cli.py` — `analyze_ticker` passes `on_event` through.
- `api/__init__.py`, `api/core.py` — seam wiring (health/run_analyze/ingest/backtest/config).
- `api/sse.py` — `sse_event` formatter.
- `api/main.py` — FastAPI app + endpoints + CORS.
- `pyproject.toml` — `web` extra.
- `tests/test_orchestrator.py`, `tests/api/test_endpoints.py`.

Naming contract: `Orchestrator.run(ticker, as_of, on_event=None)`, `on_event({"agent", "opinion"} | {"agent", "skipped", "reason"})`, `api.core.health/run_analyze/ingest_ticker/run_backtest_records/backtest_config`, `sse_event(event, data)`.

---

## Task 1: Orchestrator on_event hook (+ cli passthrough)

**Files:**
- Modify: `equity_research/orchestration/orchestrator.py`
- Modify: `equity_research/cli.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_orchestrator.py`:
```python
def test_on_event_called_per_agent(tmp_path):
    agents = [StubAgent("fundamentals", 0.5), StubAgent("technical", 0.0, fail=True)]
    orch = Orchestrator(agents=agents, aggregator=_agg(tmp_path))
    events = []
    orch.run("AAPL", as_of=date(2026, 9, 15), on_event=events.append)
    assert events[0]["agent"] == "fundamentals" and "opinion" in events[0]
    assert events[1]["agent"] == "technical" and events[1]["skipped"] is True
    assert "data unavailable" in events[1]["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_orchestrator.py::test_on_event_called_per_agent -v`
Expected: FAIL — `run` has no `on_event`.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/orchestration/orchestrator.py`, change `run` to accept and fire `on_event`:
```python
    def run(self, ticker: str, as_of: date, on_event=None) -> Verdict:
        opinions: list[AgentOpinion] = []
        skipped: list[str] = []
        skip_reasons: dict[str, str] = {}
        for agent in self.agents:
            try:
                evidence = agent.gather(ticker, as_of)
                opinion = agent.judge(evidence)
                opinions.append(opinion)
                if on_event is not None:
                    on_event({"agent": agent.name, "opinion": opinion})
            except Exception as exc:
                logger.exception("agent %s failed during run", agent.name)
                skipped.append(agent.name)
                reason = f"{type(exc).__name__}: {exc}"
                skip_reasons[agent.name] = reason
                if on_event is not None:
                    on_event({"agent": agent.name, "skipped": True, "reason": reason})
        return self.aggregator.aggregate(ticker, as_of, opinions, skipped, skip_reasons)
```

In `equity_research/cli.py`, change `analyze_ticker` to accept and forward `on_event`. Change the signature and the final `orch.run` call:
```python
def analyze_ticker(ticker: str, as_of: date, cfg_path: str, on_event=None) -> Verdict:
```
and:
```python
    return orch.run(ticker, as_of, on_event=on_event)
```
(Everything else in `analyze_ticker` unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_orchestrator.py tests/test_cli.py -v`
Expected: PASS (new + existing; existing `run(ticker, as_of)` calls still work with `on_event=None`). Full suite `uv run pytest -m "not integration" -p no:warnings -q` green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/orchestration/orchestrator.py equity_research/cli.py tests/test_orchestrator.py
git commit -m "feat(orchestration): on_event hook for per-agent streaming"
```

---

## Task 2: web deps + api package + health endpoint

**Files:**
- Modify: `pyproject.toml`
- Create: `api/__init__.py` (empty)
- Create: `api/core.py`
- Create: `api/sse.py`
- Create: `api/main.py`
- Create: `tests/api/__init__.py` (empty)
- Test: `tests/api/test_endpoints.py`

- [ ] **Step 1: Write the failing test**

`tests/api/test_endpoints.py`:
```python
from fastapi.testclient import TestClient

import api.core as core
import api.main as main


def test_health(monkeypatch):
    monkeypatch.setattr(core, "health", lambda: {"ok": True, "model": "qwen2.5:7b", "ollama_reachable": True})
    client = TestClient(main.app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["ollama_reachable"] is True
    assert resp.json()["model"] == "qwen2.5:7b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_endpoints.py -v`
Expected: FAIL — `api` package / fastapi missing.

- [ ] **Step 3: Write minimal implementation**

In `pyproject.toml`, add a `web` optional extra (keep `dev`, `rerank`):
```toml
web = ["fastapi>=0.110", "uvicorn>=0.29", "httpx>=0.27"]
```
Then `uv sync --extra dev --extra web`.

`api/__init__.py`: empty. `tests/api/__init__.py`: empty.

`api/sse.py`:
```python
from __future__ import annotations

import json


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
```

`api/core.py` (thin real wiring; verified live in Task 6 — reuses existing functions):
```python
from __future__ import annotations

from datetime import date, datetime

CONFIG_PATH = "config.yaml"


def health() -> dict:
    from equity_research.config import Config

    cfg = Config.load(CONFIG_PATH)
    reachable = True
    try:
        from ollama import Client

        Client(host=cfg.ollama_host, timeout=5).list()
    except Exception:
        reachable = False
    return {"ok": reachable, "model": cfg.model, "ollama_reachable": reachable}


def run_analyze(ticker: str, on_event) -> "object":
    from equity_research.cli import analyze_ticker

    return analyze_ticker(ticker.upper(), date.today(), CONFIG_PATH, on_event=on_event)


def ingest_ticker(ticker: str) -> int:
    from equity_research.config import Config
    from equity_research.data.adapters import fetch_news
    from equity_research.rag.chroma_store import ChromaVectorStore
    from equity_research.rag.ingest import ingest_news
    from equity_research.rag.store import NewsStore
    from equity_research.util.resilient import resilient

    cfg = Config.load(CONFIG_PATH)
    store = NewsStore(ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"], embed_model=cfg.rag["embed_model"]))
    return ingest_news(resilient(fetch_news, cfg.net), store, ticker.upper())


def backtest_config() -> dict:
    from equity_research.config import Config

    cfg = Config.load(CONFIG_PATH)
    return {k: cfg.backtest[k] for k in ("universe", "dates", "horizons")}


def run_backtest_records(on_progress):
    from equity_research.config import Config
    from equity_research.eval.backtest import run_backtest
    from equity_research.cli import _full_close, build_backtest_verdict
    from equity_research.llm.cache import DiskCache
    from equity_research.llm.ollama_client import OllamaClient

    cfg = Config.load(CONFIG_PATH)
    from ollama import Client

    chat_fn = Client(host=cfg.ollama_host, timeout=cfg.net["ollama_timeout"]).chat
    client = OllamaClient(model=cfg.model, cache=DiskCache(cfg.cache_dir),
                          seed=cfg.seed, temperature=cfg.temperature, chat_fn=chat_fn)
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in cfg.backtest["dates"]]
    run_verdict = build_backtest_verdict(cfg, client)

    def run_verdict_progress(ticker, as_of):
        on_progress({"ticker": ticker, "as_of": str(as_of)})
        return run_verdict(ticker, as_of)

    records = run_backtest(cfg.backtest["universe"], dates, cfg.backtest["horizons"],
                           run_verdict_progress, _full_close)
    return records, cfg.backtest["horizons"]
```
NOTE (verify live in Task 6): `cli._full_close` and `cli.build_backtest_verdict` must be importable — they are module-level in `equity_research/cli.py` from Phase 3B.

`api/main.py`:
```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import api.core as core

app = FastAPI(title="Multi-Agent Equity Research API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return core.health()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/api/test_endpoints.py -v`
Expected: PASS (1 test). Confirm `uv run python -c "import api.main"` works offline. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock api tests/api
git commit -m "feat(api): FastAPI app skeleton + core seams + health"
```

---

## Task 3: analyze SSE endpoint

**Files:**
- Modify: `api/main.py`
- Test: `tests/api/test_endpoints.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/api/test_endpoints.py`:
```python
def test_analyze_streams_agents_then_verdict(monkeypatch):
    from datetime import date

    from equity_research.agents.base import AgentOpinion
    from equity_research.orchestration.aggregator import Verdict

    def fake_run(ticker, on_event):
        on_event({"agent": "technical",
                  "opinion": AgentOpinion(agent="technical", stance="bullish", score=0.5,
                                          confidence=0.8, rationale="r")})
        on_event({"agent": "sentiment", "skipped": True, "reason": "no news"})
        return Verdict(ticker=ticker, as_of=date(2026, 9, 15), verdict="buy", score=0.5,
                       confidence=0.8, narrative="n", opinions=[], skipped_agents=["sentiment"])

    monkeypatch.setattr(core, "run_analyze", fake_run)
    client = TestClient(main.app)
    body = client.get("/api/analyze?ticker=AAPL").text
    assert "event: agent" in body
    assert "technical" in body and "skipped" in body
    assert "event: verdict" in body
    assert '"verdict": "buy"' in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_endpoints.py::test_analyze_streams_agents_then_verdict -v`
Expected: FAIL — no `/api/analyze` route.

- [ ] **Step 3: Write minimal implementation**

In `api/main.py`, add the imports and the SSE endpoint:
```python
import json
import queue
import threading

from fastapi.responses import StreamingResponse

from api.sse import sse_event
```
```python
@app.get("/api/analyze")
def analyze(ticker: str):
    q: "queue.Queue" = queue.Queue()

    def on_event(ev: dict):
        if ev.get("skipped"):
            q.put(("agent", {"agent": ev["agent"], "skipped": True, "reason": ev["reason"]}))
        else:
            q.put(("agent", {"agent": ev["agent"], "opinion": ev["opinion"].model_dump(mode="json")}))

    def worker():
        try:
            verdict = core.run_analyze(ticker, on_event)
            q.put(("verdict", json.loads(verdict.model_dump_json())))
        except Exception as exc:  # surfaces to the client as an SSE error event
            q.put(("error", {"message": str(exc)}))
        finally:
            q.put((None, None))

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            event, data = q.get()
            if event is None:
                break
            yield sse_event(event, data)

    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/api/test_endpoints.py -v`
Expected: PASS (health + analyze). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/api/test_endpoints.py
git commit -m "feat(api): SSE /api/analyze (per-agent stream + verdict)"
```

---

## Task 4: ingest + backtest/config endpoints

**Files:**
- Modify: `api/main.py`
- Test: `tests/api/test_endpoints.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/api/test_endpoints.py`:
```python
def test_ingest(monkeypatch):
    monkeypatch.setattr(core, "ingest_ticker", lambda ticker: 7)
    client = TestClient(main.app)
    resp = client.post("/api/ingest", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    assert resp.json() == {"ticker": "AAPL", "ingested": 7}


def test_backtest_config(monkeypatch):
    monkeypatch.setattr(core, "backtest_config",
                        lambda: {"universe": ["AAPL"], "dates": ["2024-03-15"], "horizons": [21, 63]})
    client = TestClient(main.app)
    resp = client.get("/api/backtest/config")
    assert resp.json()["horizons"] == [21, 63]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_endpoints.py::test_ingest tests/api/test_endpoints.py::test_backtest_config -v`
Expected: FAIL — routes missing.

- [ ] **Step 3: Write minimal implementation**

In `api/main.py`, add a request model and the two routes:
```python
from pydantic import BaseModel


class IngestRequest(BaseModel):
    ticker: str


@app.post("/api/ingest")
def ingest(req: IngestRequest):
    n = core.ingest_ticker(req.ticker)
    return {"ticker": req.ticker.upper(), "ingested": n}


@app.get("/api/backtest/config")
def backtest_config():
    return core.backtest_config()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/api/test_endpoints.py -v`
Expected: PASS (health + analyze + ingest + backtest_config). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/api/test_endpoints.py
git commit -m "feat(api): /api/ingest + /api/backtest/config"
```

---

## Task 5: backtest SSE endpoint

**Files:**
- Modify: `api/main.py`
- Test: `tests/api/test_endpoints.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/api/test_endpoints.py`:
```python
def test_backtest_streams_progress_then_report(monkeypatch):
    from datetime import date

    from equity_research.eval.backtest import BacktestRecord

    def fake_run(on_progress):
        on_progress({"ticker": "AAPL", "as_of": "2024-03-15"})
        recs = [BacktestRecord("AAPL", date(2024, 3, 15), "buy", 0.5, {21: 0.1, 63: 0.2})]
        return recs, [21, 63]

    monkeypatch.setattr(core, "run_backtest_records", fake_run)
    client = TestClient(main.app)
    body = client.get("/api/backtest").text
    assert "event: progress" in body and "AAPL" in body
    assert "event: report" in body
    assert '"21"' in body  # horizon key in the report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_endpoints.py::test_backtest_streams_progress_then_report -v`
Expected: FAIL — no `/api/backtest` route.

- [ ] **Step 3: Write minimal implementation**

In `api/main.py`, add the backtest SSE endpoint (reuse the report renderer):
```python
@app.get("/api/backtest")
def backtest():
    q: "queue.Queue" = queue.Queue()

    def on_progress(ev: dict):
        q.put(("progress", ev))

    def worker():
        try:
            from equity_research.eval.report import render_backtest_json

            records, horizons = core.run_backtest_records(on_progress)
            q.put(("report", json.loads(render_backtest_json(records, horizons))))
        except Exception as exc:
            q.put(("error", {"message": str(exc)}))
        finally:
            q.put((None, None))

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            event, data = q.get()
            if event is None:
                break
            yield sse_event(event, data)

    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/api/test_endpoints.py -v`
Expected: PASS (all 5 API tests). Full suite `uv run pytest -m "not integration" -p no:warnings -q` green.

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/api/test_endpoints.py
git commit -m "feat(api): SSE /api/backtest (progress + report)"
```

---

## Task 6: Live verification + run docs

**Files:**
- Create: `api/README.md`

- [ ] **Step 1: Full unit suite**

Run: `uv run pytest -m "not integration" -p no:warnings -q`
Expected: all PASS.

- [ ] **Step 2: Start the server**

Run (background): `DISABLE_PANDERA_IMPORT_WARNING=True uv run uvicorn api.main:app --port 8000 &`
Then wait ~2s and probe health:
Run: `curl -s http://localhost:8000/api/health`
Expected: JSON `{"ok": ..., "model": "qwen2.5:7b", "ollama_reachable": true}` (true if Ollama running).

- [ ] **Step 3: Live analyze SSE (real, slow)**

Run: `curl -sN "http://localhost:8000/api/analyze?ticker=AAPL" | head -40`
Expected: a stream of `event: agent` blocks (fundamentals/technical/sentiment/risk as they finish) then `event: verdict`. If `core.run_analyze` / `cli._full_close` / `build_backtest_verdict` imports fail, fix `api/core.py` only. Stop the server afterward (`kill %1` or find the pid).

- [ ] **Step 4: Write `api/README.md`**

```markdown
# API (FastAPI)

Backend for the web UI. Reuses the `equity_research` core.

## Run
    uv sync --extra web
    uv run uvicorn api.main:app --port 8000

Requires Ollama running with the configured model and `nomic-embed-text` pulled.

## Endpoints
- `GET /api/health`
- `GET /api/analyze?ticker=AAPL`  (SSE: `agent` events, then `verdict`)
- `POST /api/ingest` `{ "ticker": "AAPL" }`
- `GET /api/backtest`  (SSE: `progress` events, then `report`)
- `GET /api/backtest/config`

CORS allows `http://localhost:5173` (the Vite dev origin) by default.
```

- [ ] **Step 5: Commit**

```bash
git add api/README.md
git commit -m "docs(api): run instructions"
```
(Include any `api/core.py` wiring fix from Step 3 in this commit if one was needed.)

---

## Self-Review (completed during authoring)

- **Spec coverage (4A):** orchestrator `on_event` hook (Task 1); FastAPI app + `/health` + CORS (Task 2); SSE `/api/analyze` with per-agent + verdict events via queue/thread bridge (Task 3); `/api/ingest` + `/api/backtest/config` (Task 4); SSE `/api/backtest` progress+report (Task 5); live verification + README (Task 6). Streaming hook is the only core change; endpoints reuse `analyze_ticker(on_event=...)` and `build_backtest_verdict`. Error handling: SSE `error` events + `/health` (spec §4).
- **Placeholder scan:** none. `api/core.py` carries real wiring plus an explicit "verify live, fix only this file" note (matches the EdgarProvider pattern).
- **Type consistency:** `Orchestrator.run(ticker, as_of, on_event=None)`, the `on_event` payload shapes (`{agent, opinion}` / `{agent, skipped, reason}`), `api.core.health/run_analyze/ingest_ticker/backtest_config/run_backtest_records`, `sse_event(event, data)`, and reused `Verdict`/`AgentOpinion`/`BacktestRecord`/`render_backtest_json` all match existing signatures. Endpoints reference `core.<fn>` (attribute lookup) so tests monkeypatch cleanly.
