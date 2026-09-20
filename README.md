# Multi-Agent Equity Research

A research-grade, **multi-agent equity analysis system** — a "mini hedge fund" that runs a panel of
specialised LLM analyst agents over a stock, reconciles their views into a single buy/hold/sell
verdict, and ships with a **real evaluation harness** to measure whether those verdicts have any
predictive value. Everything runs **locally** on an open-weights LLM (Ollama); no paid APIs.

It is built for **correctness and honesty over hype**: point-in-time data (no look-ahead),
reproducible LLM output, grounded facts, and a backtest that reports the system's information
coefficient plainly — including when the edge is not there.

> ⚠️ Automated research aid, **not investment advice**. Figures can be incomplete or wrong.

---

## What it does

Given a ticker and an as-of date, the system:

1. **Gathers point-in-time evidence** — prices, SEC 10-K fundamentals, filing text, and news.
2. **Runs four specialised agents**, each an LLM constrained to a narrow judgment:
   - **Fundamentals** — valuation & financial health (P/E, ROE, D/E, margins, FCF, current ratio) from the latest 10-K on/before the date, plus retrieved filing sections.
   - **Technical** — trend & momentum (SMA-50/200 cross, RSI, 3-month trend).
   - **Sentiment** — tone of recent, ticker-relevant news (RAG-retrieved).
   - **Risk** — volatility, beta, max drawdown (used as a confidence gate, not a direction).
3. **Reconciles** the opinions via an **LLM portfolio-manager** blended with a deterministic
   weighted score (the mechanical score stays the anchor), gated by risk.
4. **Self-critiques** each opinion against its evidence (lowers confidence when a rationale is
   weak) and **calibrates** confidence by the class's measured historical hit-rate.
5. Returns a **Verdict** (direction, score, confidence, narrative, per-agent opinions, disclaimers)
   — streamed to the web client token-by-token via SSE.

## Why it's interesting (engineering highlights)

- **Code-driven orchestration ("Approach A")** — a deterministic orchestrator with *constrained*
  LLM judgment per agent, rather than a free-form tool-calling loop. Predictable, testable, debuggable.
- **Point-in-time correctness / no look-ahead** — `PriceProvider.history(ticker, as_of)` filters
  prices to the date; `EdgarProvider.company_facts(ticker, as_of)` selects the latest 10-K ≤ as_of.
  The backtest is genuinely historical.
- **Reproducible, structured LLM output** — grammar-constrained JSON (`format=schema`) + retry +
  `jsonschema` validation, fixed seed and temperature 0, and a content-addressed disk cache. Same
  input → same output.
- **Grounding guardrail** — cited facts are checked against the actual metrics/context; fabricated
  numbers are dropped, and prompt-injection in retrieved text is fenced as untrusted data.
- **RAG** — Chroma vector store, MMR retrieval, an optional cross-encoder re-ranker, and a
  ParentDocument scheme for long 10-K sections (child chunks indexed, whole sections stored).
- **Resilient data layer** — yfinance primary with a Stooq cross-check, EDGAR via `edgartools`,
  retries with exponential backoff and per-call timeouts; every field degrades to `NaN` rather than
  crashing.
- **Evaluation harness** — a historical backtest computing hit-rate by class, mean forward return by
  class, a hand-rolled Spearman **information coefficient**, and a long-short curve at 21/63-day
  horizons, plus a **golden set** (hand-verified fundamentals, retrieval relevance pairs, end-to-end
  verdict baselines) and a retrieval `hit@k` / MRR eval.
- **Streaming API** — FastAPI + Server-Sent Events bridging the synchronous core to the browser.
- **210 tests** (deterministic unit tests run by default; live integration tests behind a marker).

## Architecture

```
                         ┌──────────────────────────────────────────┐
  ticker, as_of  ─────▶  │              Orchestrator                 │
                         │  gather → judge → self-critique (opt)     │
                         └───────┬───────────┬───────────┬──────────┘
                                 │           │           │
                    ┌────────────▼┐ ┌────────▼───┐ ┌─────▼──────┐ ┌──────────┐
                    │Fundamentals │ │ Technical  │ │ Sentiment  │ │  Risk    │
                    │  (EDGAR+RAG)│ │ (indicators)│ │ (news RAG) │ │ (vol/β)  │
                    └────────────┬┘ └────────┬───┘ └─────┬──────┘ └────┬─────┘
                                 └───────────┴─────┬─────┴─────────────┘
                                                   ▼
                            ┌──────────────────────────────────────┐
                            │  Aggregator: weighted score (anchor)  │
                            │  + LLM portfolio-manager blend        │
                            │  + risk gate + confidence calibration │
                            └───────────────────┬──────────────────┘
                                                 ▼
                                             Verdict (SSE)
```

Data sources: **yfinance** (prices, quotes, news) with a **Stooq** cross-check · **SEC EDGAR**
(10-K fundamentals + filing text) · **Chroma** (vectors) · **Ollama** (LLM + embeddings).

## Tech stack

**Python** · pydantic · pandas / numpy · pandera (data validation) · **FastAPI** + SSE · **Ollama**
(`qwen2.5` chat, `nomic-embed-text` embeddings) · **LangChain + Chroma** (RAG) · `edgartools`
(SEC) · yfinance / pandas-datareader · sentence-transformers (cross-encoder re-rank) · Typer (CLI)
· pytest. Packaged with `uv` / hatchling.

## Repository layout

```
equity_research/
  agents/         judge, grounding, self-critique, per-agent prompts + the four agents
  analytics/      fundamentals metrics, technical indicators, risk, price series
  data/           PriceProvider (yfinance+stooq), EdgarProvider, ticker resolver, adapters
  orchestration/  Orchestrator, Aggregator, LLM portfolio-manager
  rag/            Chroma store, chunking, retrievers, re-ranker, ParentDocument, filings
  eval/           backtest, metrics (IC/hit-rate/mean-return), forward returns, calibration, report
  llm/            Ollama client (grammar-constrained JSON, retry, disk cache)
  util/           resilient retry/timeout wrapper
api/              FastAPI app + SSE (analyze / backtest / prices / quotes / resolve / news)
tests/            210 tests; tests/integration/* behind the `integration` marker
docs/             design specs & implementation plans
```

## Quickstart

**Prerequisites:** [Ollama](https://ollama.com) running locally, and [uv](https://docs.astral.sh/uv/).

```bash
# 1. models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 2. install
uv sync

# 3. configure — set your SEC user agent (name + email) in config.yaml (edgar_user_agent)

# 4. analyze a ticker (CLI)
uv run python -m equity_research.cli analyze AAPL

# 5. ingest news + filings into the vector store
uv run python -m equity_research.cli ingest AAPL

# 6. run the backtest (writes backtest_report.md)
uv run python -m equity_research.cli backtest

# 7. serve the API (for the web client)
uv run uvicorn api.main:app --port 8000
```

Tests:

```bash
uv run pytest                     # fast deterministic suite (210)
uv run pytest -m integration      # live tests (network + Ollama)
```

## Configuration

`config.yaml` drives everything: model & seed, agent weights, risk thresholds, RAG parameters,
network timeouts/retries, the backtest universe/dates, and optional layers —
`judge_model` / `narrative_model` (route judging to a stronger model), `pm_enabled` (LLM
portfolio-manager), `self_critique_enabled`, and `calibration_enabled`.

## Evaluation — an honest note

The backtest is deliberately included so the system can be judged, not just admired. On a 10-ticker
× 6-date (2023–2024) run the information coefficient sits near zero / slightly negative — i.e. the
verdicts show **no proven predictive edge** over that (bull-market) period, and the system's
value/mean-reversion lean underperformed momentum. That finding is surfaced plainly rather than
hidden: the goal of the harness is exactly this kind of measurable, reproducible feedback. The
confidence-calibration layer then makes the reported confidence honest about that track record.

## License

For research and educational use. Not investment advice.
