# Multi-Agent Equity Research

![tests](https://img.shields.io/badge/tests-210%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.13-blue)
![llm](https://img.shields.io/badge/LLM-local%20(Ollama)-6aa84f)
![status](https://img.shields.io/badge/status-research%20project-lightgrey)

A multi-agent equity analysis system — a "mini hedge fund" that runs a panel of specialised LLM
analyst agents over a stock, reconciles their views into a single **buy / hold / sell** verdict, and
ships with an **evaluation harness** that measures whether those verdicts have any predictive value.
Runs fully locally on an open-weights model via Ollama; **no paid APIs**. The LLM client is
**provider-agnostic by design** — a hosted model (OpenAI, Anthropic, …) can be wired behind the same
interface (see [LLM backend](#llm-backend)).

Built for **correctness over hype**: point-in-time data (no look-ahead), reproducible LLM output,
grounded facts, and a backtest that reports the system's information coefficient plainly — including
when the edge is not there.

> ⚠️ Automated research aid, **not investment advice**. Figures can be incomplete or wrong.

---

## Results at a glance

Backtest: **10 tickers × 6 as-of dates** (2023-06 … 2024-09), **60 verdicts**, forward horizons
21 / 63 trading days. Full report: [`backtest_report.md`](backtest_report.md).

| Metric | 21 d | 63 d |
|---|---|---|
| Information coefficient (Spearman, score vs fwd return) | **−0.036** | **−0.192** |
| Hit-rate — BUY | 62 % | 56 % |
| Hit-rate — SELL | 33 % | 14 % |
| Mean fwd return — BUY / HOLD / SELL | +1.65 % / +2.80 % / +2.22 % | +3.08 % / +8.90 % / +10.61 % |
| Naive long–short (BUY − SELL) cumulative | −20.2 % | −173.6 % |

Verdict mix: 16 buy / 23 hold / 21 sell. Component evals (`tests/fixtures/golden/`, run with
`-m integration`): retrieval **hit@3 = 1.00 / MRR = 1.00** (12 relevance pairs), fundamentals metrics
match hand-verified 10-K computations for 5 tickers, and re-runs are **byte-identical** (seed + T = 0
+ disk cache).

**Honest read:** over this (bull-market) window the verdicts show **no proven predictive edge** — IC
is indistinguishable from zero at 21 d and negative at 63 d, and the system's value / mean-reversion
lean underperformed momentum (its SELL calls rose the most). n = 60 over a single regime is far too
small for significance either way; the point of the harness is that this is *measured*, reproducible,
and re-runnable on a wider universe. The confidence-calibration layer then makes the reported
confidence honest about that track record.

---

## What it does

Given a ticker and an as-of date, the system:

1. **Gathers point-in-time evidence** — prices, SEC 10-K fundamentals, filing text and news.
2. **Runs four specialised agents**, each an LLM constrained to one narrow, schema-validated judgment:
   - **Fundamentals** — valuation & financial health (P/E, ROE, D/E, margins, FCF, current ratio)
     from the latest 10-K on/before `as_of`, plus RAG-retrieved filing sections.
   - **Technical** — trend & momentum (SMA-50/200 cross, RSI, 3-month trend).
   - **Sentiment** — tone of recent ticker-relevant news (RAG-retrieved).
   - **Risk** — volatility, beta, max drawdown; used as a *confidence gate*, not a direction.
3. **Reconciles** the opinions: a deterministic weighted score is the anchor, blended with an LLM
   portfolio-manager, gated by risk.
4. **Self-critiques** each opinion against its own evidence (lowers confidence when a rationale is
   weak) and **calibrates** confidence by the class's measured historical hit-rate.
5. **Returns a `Verdict`** — direction, score, confidence, narrative, per-agent opinions, disclaimers
   — streamed to the web client token-by-token over SSE.

### Example verdict (shape)

```json
{
  "ticker": "AAPL",
  "verdict": "hold",
  "score": 0.12,
  "confidence": 0.41,
  "narrative": "Analysts are split: technicals and sentiment lean bullish while fundamentals flag …",
  "opinions": [
    {"agent": "fundamentals", "stance": "bearish", "score": -0.30, "confidence": 0.55,
     "rationale": "High P/E versus modest growth; strong margins but rich valuation.",
     "key_facts": ["P/E 44.2x", "Revenue growth +6.4%"],
     "metrics": {"pe": 44.2, "roe": 1.56, "net_margin": 0.25},
     "critique": "The P/E-vs-ROE comparison is not directly comparable without more context."},
    {"agent": "technical", "stance": "bullish", "score": 0.70, "confidence": 0.80,
     "rationale": "Golden cross with RSI in a healthy range.", "key_facts": ["golden cross", "RSI 67"]},
    {"agent": "sentiment", "stance": "neutral", "score": 0.0, "confidence": 0.30, "rationale": "…"},
    {"agent": "risk", "stance": "neutral", "score": 0.62, "confidence": 0.62, "rationale": "…"}
  ],
  "calibration_note": "Confidence tempered ×0.67 by historical sell accuracy",
  "disclaimer": "Automated research aid, not investment advice."
}
```

---

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

**Data sources:** yfinance (prices, quotes, news) with a Stooq cross-check · SEC EDGAR via
`edgartools` (10-K fundamentals + filing text) · Chroma (vectors) · Ollama (LLM + embeddings).

### Design decision: code-driven orchestration, not a free-form agent loop

The orchestrator is deterministic Python; each agent is an LLM call constrained to a narrow,
schema-validated judgment. This was chosen over a LangGraph / tool-calling-loop design deliberately:

| | Code-driven orchestrator (this repo) | Free-form tool-calling loop |
|---|---|---|
| Control flow | fixed, inspectable | emergent, model-dependent |
| Testability | every stage unit-testable with mocks | needs trajectory-level evals |
| Reproducibility | seed + T = 0 + cache → byte-identical | hard even at T = 0 |
| Failure modes | typed, localised | tool loops, silent drift |
| Flexibility | low — new evidence = new code | high — model decides what to fetch |

For a *research* system that must be backtested, reproducibility and testability win. The trade-off
is flexibility: the agents cannot decide to fetch evidence the pipeline didn't anticipate. A
tool-calling variant of the portfolio-manager, evaluated side-by-side in the same backtest, is on the
[roadmap](#roadmap). Design docs live under [`docs/superpowers/`](docs/superpowers/).

---

## Engineering highlights

- **Point-in-time correctness / no look-ahead.** `PriceProvider.history(ticker, as_of)` truncates the
  series at `as_of`; `EdgarProvider.company_facts(ticker, as_of)` selects the latest 10-K with
  `filing_date ≤ as_of`. See [Limitations](#limitations) for the one source where this is partial.
- **Reproducible, structured LLM output.** Grammar-constrained JSON (`format=schema`) + retry +
  `jsonschema` validation; fixed seed, temperature 0; content-addressed disk cache. Same input →
  same output (verified by the reproducibility test and the golden verdict baselines).
- **Grounding guardrail.** Every cited fact is checked against the actual metrics/context; fabricated
  numbers are dropped. Retrieved text is fenced as untrusted data to blunt prompt injection
  (`tests/test_grounding.py`).
- **Self-critique + calibration.** A critic LLM reviews each opinion against its evidence and can only
  *lower* confidence; a calibration layer then tempers confidence by the class's measured hit-rate
  (`min(1, 2·hit)`), never changing the verdict — so confidence reflects the real track record.
- **RAG.** Chroma vector store, MMR retrieval, an optional cross-encoder re-ranker
  (`sentence-transformers`), and a ParentDocument scheme for long 10-K sections (child chunks indexed,
  whole sections returned).
- **Resilient data layer.** yfinance primary with a Stooq cross-check; retries with exponential
  backoff and per-call timeouts; every field degrades to `NaN` rather than crashing; `pandera`
  schemas validate frames at the boundary.
- **Evaluation harness.** Historical backtest (hit-rate by class, mean forward return, Spearman IC,
  long–short curve at 21/63 d) + golden set (hand-verified fundamentals, retrieval relevance pairs,
  end-to-end verdict baselines) + retrieval hit@k / MRR.
- **Streaming API.** FastAPI + Server-Sent Events bridging the synchronous core to the browser.
- **210 tests.** Deterministic unit suite by default; live integration tests behind the
  `integration` marker.

### How it's tested

| Layer | Approach |
|---|---|
| Analytics (indicators, metrics, risk) | pure functions, fixture tests |
| Data providers | fake fetchers injected; live calls only under `-m integration` |
| LLM agents | fake `chat_fn` + content-addressed cache → deterministic; schema validation asserted |
| Grounding / self-critique | adversarial fixtures (fabricated numbers, injected instructions) |
| Retrieval | golden relevance pairs → hit@k / MRR thresholds |
| End-to-end | golden verdict baselines; drift fails the suite |

---

## Operational profile (approximate, dev machine)

| | Value |
|---|---|
| One `analyze` run, cold cache (qwen2.5:7b, all layers on) | ~2–3 min |
| One `analyze` run, warm cache | seconds |
| LLM calls per verdict | up to ~8 (4 agents + 3 critiques + portfolio-manager) |
| Backtest (60 verdicts, cold) | ~15–20 min |

Latency is dominated by local LLM inference; the disk cache makes re-runs near-instant and is what
keeps the backtest reproducible.

---

## Quickstart

Prerequisites: [Ollama](https://ollama.com/) running locally, and [uv](https://docs.astral.sh/uv/).

```bash
# 1. models
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 2. install
uv sync

# 3. configure — set your SEC user agent (name + email) in config.yaml (edgar_user_agent)

# 4. analyze a ticker (uses today's date)
uv run python -m equity_research.cli analyze AAPL

# 5. ingest news + filings into the vector store
uv run python -m equity_research.cli ingest AAPL

# 6. run the backtest (writes backtest_report.md)
uv run python -m equity_research.cli backtest

# 7. serve the API (for the web client — separate repo)
uv run uvicorn api.main:app --port 8000
```

Tests:

```bash
uv run pytest                     # fast deterministic suite (210)
uv run pytest -m integration      # live tests (network + Ollama)
```

---

## Configuration

`config.yaml` drives everything: model & seed, agent weights, risk thresholds, RAG parameters,
network timeouts/retries, the backtest universe/dates, and the SEC `edgar_user_agent`. Optional layers:

| Key | Purpose |
|---|---|
| `judge_model`, `narrative_model` | route judging / narrative to a stronger model |
| `pm_enabled`, `pm_weight` | LLM portfolio-manager blended with the mechanical score |
| `self_critique_enabled` | critic pass that can only lower confidence |
| `calibration_enabled`, `calibration_horizon` | temper confidence by measured hit-rate |
| `weights`, `risk.*`, `rag.*`, `net.*`, `backtest.*` | agent weights, gate, retrieval, retries, eval window |

### LLM backend

The core is **not tied to a specific provider**. `OllamaClient` is constructed with an injected
`chat_fn` and a model name, and `judge_model` / `narrative_model` already let you route different
roles to different models. Wiring a hosted backend (OpenAI, Anthropic, Azure OpenAI) is therefore an
adapter, not a rewrite: supply a `chat_fn` that returns the same message shape and maps the structured
-output call (Ollama's grammar-constrained `format=schema` → the provider's JSON-schema / tool mode).

**Status:** only the local **Ollama** backend is implemented and tested (all results above are on
`qwen2.5:7b`). Hosted providers — and a cross-provider backtest comparison — are on the
[roadmap](#roadmap); they are not claimed to work until that lands.

---

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
tests/            210 tests; tests/fixtures/golden/*, tests/integration/* behind the `integration` marker
docs/             design specs & implementation plans
```

The **web client** (React + TypeScript + Vite) lives in a separate repository.

---

## Limitations

Stated plainly, because they affect how much to trust the numbers above.

- **News is not fully point-in-time.** yfinance returns *current* headlines, not a historical archive.
  Sentiment retrieval filters by publish date ≤ `as_of`, so for historical backtest dates most news
  is filtered out and the Sentiment agent is effectively neutral there. Prices and 10-K fundamentals
  *are* strictly point-in-time. A dated news source is on the roadmap.
- **Calibration is not train/test split.** Class hit-rates are computed on the 2023–2024 backtest and
  applied to live confidence; there is no held-out split yet, so calibration is descriptive, not
  validated out-of-sample.
- **Small sample, single regime.** 60 verdicts over one bull-market window; IC intervals are wide.
  Read the results as "no evidence of edge", not "evidence of no edge".
- **Single model family.** All judgments come from qwen2.5:7b, so agent disagreement is correlated.
- **US large-caps only.** EDGAR coverage and yfinance quality drop off outside large US names, and a
  few tickers expose no diluted-share/debt fields (those metrics degrade to `NaN`).

---

## Roadmap

- [ ] **Pluggable LLM providers** — OpenAI / Anthropic behind the same client interface; backtest across providers.
- [ ] **Tool-calling portfolio-manager** — a ReAct-style PM compared against the code-driven PM in the same backtest.
- [ ] **Dated news source** to make Sentiment fully point-in-time.
- [ ] **Wider, multi-regime backtest** — 100+ tickers × monthly dates across 2019–2024 (incl. the 2022 drawdown), with out-of-sample calibration.
- [ ] **Docker Compose** (API + Ollama + Chroma) and CI with coverage.
- [ ] **Observability** — per-agent tracing (tokens, latency, prompt versions).

---

## License

For research and educational use. Not investment advice.
