from __future__ import annotations

import json
import queue
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import api.core as core
from api.sse import sse_event

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


@app.get("/api/prices")
def prices(ticker: str, period: str = "6M"):
    return core.prices(ticker, period)


@app.get("/api/quotes")
def quotes(tickers: str):
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    return core.quotes(syms)


@app.get("/api/resolve")
def resolve(query: str):
    return core.resolve(query)


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
        except Exception as exc:  # surfaced to the client as an SSE error event
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


class IngestRequest(BaseModel):
    ticker: str


@app.post("/api/ingest")
def ingest(req: IngestRequest):
    n = core.ingest_ticker(req.ticker)
    return {"ticker": req.ticker.upper(), "ingested": n}


class NewsRequest(BaseModel):
    query: str


@app.post("/api/news")
def news(req: NewsRequest):
    return core.news(req.query)


@app.get("/api/backtest/config")
def backtest_config():
    return core.backtest_config()


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
