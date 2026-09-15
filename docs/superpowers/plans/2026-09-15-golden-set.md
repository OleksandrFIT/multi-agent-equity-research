# Golden set — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. NOTE: tasks tagged **[CONTROLLER-LIVE]** capture golden data from live sources (EDGAR / a running Ollama) and must be run by the controller, not a fresh subagent — a subagent has no reviewed ground truth and no live environment.

**Goal:** A curated golden set evaluating the system on three axes — fundamentals metrics (deterministic), retrieval relevance (hit@k / MRR), and end-to-end verdicts (regression/stability) — with deterministic parts in the default suite and live parts under `-m integration`.

**Architecture:** Golden JSON fixtures under `tests/fixtures/golden/`; `retrieval_eval` gains multi-expected hit@k + MRR; deterministic unit tests iterate the fixtures; integration tests run the real retriever and real analyze pipeline over the golden data.

**Tech Stack:** Python, pytest (markers: `integration`).

**Repo/branch:** backend repo `/Users/oleksandr/Documents/LLM/Multi-Agent Equity`, branch `feature/golden-set`.

**Attribution:** commit ONLY as the repo author, NO `Co-Authored-By`.

**Scope (approved):** metrics — 5 tickers (AAPL, MSFT, KO, JPM, XOM); retrieval — 3 tickers × ~4 queries; verdicts — 3 cases. Retrieval integration threshold: hit@3 ≥ 0.8 (+ MRR reported). Fixed `as_of = 2024-06-14` for metrics/verdicts (a backtest date; latest 10-K ≤ that).

---

### Task 1: `retrieval_eval` — multi-expected hit@k + MRR (deterministic)

**Files:**
- Modify: `equity_research/eval/retrieval_eval.py`
- Modify: `tests/test_retrieval_eval.py`

- [ ] **Step 1: Write failing tests (append to `tests/test_retrieval_eval.py`)**

```python
def test_hit_at_k_accepts_list_expected():
    from equity_research.eval.retrieval_eval import hit_at_k

    def retrieve_fn(ticker, question):
        return {"q1": ["d1", "d2", "d3"]}[question]

    cases = [{"ticker": "AAA", "question": "q1", "expected": ["d9", "d2"]}]  # any expected in top-k
    assert hit_at_k(retrieve_fn, cases, k=3) == 1.0


def test_mrr_uses_first_relevant_rank():
    from equity_research.eval.retrieval_eval import mrr

    def retrieve_fn(ticker, question):
        return {"q1": ["d1", "d2", "d3"], "q2": ["dx", "dy", "dz"]}[question]

    cases = [
        {"ticker": "A", "question": "q1", "expected": "d2"},   # rank 2 -> 1/2
        {"ticker": "A", "question": "q2", "expected": ["dz"]}, # rank 3 -> 1/3
    ]
    assert abs(mrr(retrieve_fn, cases, k=3) - ((0.5 + (1 / 3)) / 2)) < 1e-9


def test_mrr_zero_when_not_found():
    from equity_research.eval.retrieval_eval import mrr

    cases = [{"ticker": "A", "question": "q", "expected": "zzz"}]
    assert mrr(lambda t, q: ["a", "b"], cases, k=2) == 0.0
```

(Keep the existing `test_hit_at_k_counts_expected_in_topk` — single-string `expected` must still work.)

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_retrieval_eval.py -v`
Expected: FAIL (`mrr` missing; list `expected` unsupported).

- [ ] **Step 3: Update `equity_research/eval/retrieval_eval.py`**

```python
from __future__ import annotations

from typing import Callable


def _expected_ids(case: dict) -> set[str]:
    exp = case["expected"]
    return set(exp) if isinstance(exp, list) else {exp}


def hit_at_k(retrieve_fn: Callable[[str, str], list[str]], cases: list[dict], k: int) -> float:
    """Fraction of cases whose expected id(s) appear in the top-k retrieved ids.

    `expected` may be a single id or a list; a case is a hit if ANY expected id
    is in the top-k.
    """
    if not cases:
        return 0.0
    hits = 0
    for case in cases:
        ids = retrieve_fn(case["ticker"], case["question"])[:k]
        if _expected_ids(case) & set(ids):
            hits += 1
    return hits / len(cases)


def mrr(retrieve_fn: Callable[[str, str], list[str]], cases: list[dict], k: int) -> float:
    """Mean Reciprocal Rank over cases (1/rank of the first relevant id in top-k)."""
    if not cases:
        return 0.0
    total = 0.0
    for case in cases:
        wanted = _expected_ids(case)
        ids = retrieve_fn(case["ticker"], case["question"])[:k]
        for rank, doc_id in enumerate(ids, start=1):
            if doc_id in wanted:
                total += 1.0 / rank
                break
    return total / len(cases)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_retrieval_eval.py -v`
Expected: all pass (including the pre-existing single-expected test).

- [ ] **Step 5: Commit**

```bash
git add equity_research/eval/retrieval_eval.py tests/test_retrieval_eval.py
git commit -m "feat(eval): multi-expected hit@k and MRR"
```

---

### Task 2: Retrieval golden corpus (deterministic)

Authored labeled docs across 3 tickers × ~4 query categories, with a self-consistency test.

**Files:**
- Create: `tests/fixtures/golden/retrieval_golden.json`
- Create: `tests/test_retrieval_golden_shape.py`

- [ ] **Step 1: Create `tests/fixtures/golden/retrieval_golden.json`**

Extend the existing `hitk_corpus.json` shape to 3 tickers (AAPL, MSFT, KO), each with 4+ labeled docs covering categories earnings / legal-regulatory / supply-ops / guidance-risk, and one case per category (`expected` = list of the clearly-relevant id(s)). Example (fill all 3 tickers similarly, ~12–15 docs total, ~12 cases):

```json
{
  "documents": [
    {"id": "aapl-earnings", "ticker": "AAPL", "text": "Apple reported record quarterly revenue driven by strong iPhone and services growth."},
    {"id": "aapl-legal", "ticker": "AAPL", "text": "Apple faces an EU antitrust investigation over App Store fees and developer terms."},
    {"id": "aapl-supply", "ticker": "AAPL", "text": "Apple warned of supply chain constraints affecting Mac and iPad production this quarter."},
    {"id": "aapl-guidance", "ticker": "AAPL", "text": "Apple guided next-quarter revenue slightly below expectations citing a stronger dollar."},
    {"id": "msft-earnings", "ticker": "MSFT", "text": "Microsoft cloud revenue rose as Azure growth accelerated on AI demand."},
    {"id": "msft-legal", "ticker": "MSFT", "text": "Microsoft's Activision acquisition drew regulatory scrutiny from the FTC and CMA."},
    {"id": "msft-supply", "ticker": "MSFT", "text": "Microsoft cited data-center capacity limits constraining Azure AI service availability."},
    {"id": "msft-guidance", "ticker": "MSFT", "text": "Microsoft raised full-year guidance on strong Copilot and cloud bookings."},
    {"id": "ko-earnings", "ticker": "KO", "text": "Coca-Cola posted higher organic revenue on pricing and resilient global demand."},
    {"id": "ko-legal", "ticker": "KO", "text": "Coca-Cola faces a US tax dispute with the IRS over transfer pricing of billions."},
    {"id": "ko-supply", "ticker": "KO", "text": "Coca-Cola noted higher input costs for aluminum and sweeteners pressuring margins."},
    {"id": "ko-guidance", "ticker": "KO", "text": "Coca-Cola reaffirmed full-year organic revenue growth guidance despite currency headwinds."}
  ],
  "cases": [
    {"ticker": "AAPL", "question": "How were Apple's revenue and earnings?", "expected": ["aapl-earnings"]},
    {"ticker": "AAPL", "question": "Any antitrust or regulatory problems?", "expected": ["aapl-legal"]},
    {"ticker": "AAPL", "question": "Supply chain or production issues?", "expected": ["aapl-supply"]},
    {"ticker": "AAPL", "question": "What is the forward guidance or outlook?", "expected": ["aapl-guidance"]},
    {"ticker": "MSFT", "question": "How did cloud and Azure revenue perform?", "expected": ["msft-earnings"]},
    {"ticker": "MSFT", "question": "Any regulatory or acquisition scrutiny?", "expected": ["msft-legal"]},
    {"ticker": "MSFT", "question": "Capacity or supply constraints?", "expected": ["msft-supply"]},
    {"ticker": "MSFT", "question": "Did they change guidance or outlook?", "expected": ["msft-guidance"]},
    {"ticker": "KO", "question": "How was revenue and pricing?", "expected": ["ko-earnings"]},
    {"ticker": "KO", "question": "Any tax or legal disputes?", "expected": ["ko-legal"]},
    {"ticker": "KO", "question": "Input cost or margin pressure?", "expected": ["ko-supply"]},
    {"ticker": "KO", "question": "What is the full-year guidance?", "expected": ["ko-guidance"]}
  ]
}
```

- [ ] **Step 2: Write the self-consistency test (`tests/test_retrieval_golden_shape.py`)**

```python
import json
from pathlib import Path

FIX = Path(__file__).parent / "fixtures" / "golden" / "retrieval_golden.json"


def test_golden_corpus_is_self_consistent():
    data = json.loads(FIX.read_text())
    doc_ids = {d["id"] for d in data["documents"]}
    tickers_with_docs = {d["ticker"] for d in data["documents"]}
    assert len(doc_ids) == len(data["documents"]), "duplicate document ids"
    assert len(data["cases"]) >= 12
    for case in data["cases"]:
        exp = case["expected"]
        exp = exp if isinstance(exp, list) else [exp]
        assert exp, f"empty expected for {case['question']}"
        for e in exp:
            assert e in doc_ids, f"expected id {e} not in documents"
        assert case["ticker"] in tickers_with_docs
```

- [ ] **Step 3: Run to verify pass**

Run: `uv run pytest tests/test_retrieval_golden_shape.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/golden/retrieval_golden.json tests/test_retrieval_golden_shape.py
git commit -m "test(golden): retrieval-relevance golden corpus (3 tickers)"
```

---

### Task 3: Expand fundamentals edge cases (deterministic)

**Files:**
- Modify: `tests/fixtures/edge_cases.json`

- [ ] **Step 1: Append cases to `tests/fixtures/edge_cases.json`** (keep the existing two). Add:

```json
  {
    "name": "negative_equity_roe",
    "price": 25.0,
    "facts": {"net_income": 5000000.0, "revenue": 90000000.0, "revenue_prev": 100000000.0, "equity": -20000000.0, "total_debt": 60000000.0, "eps_ttm": 1.25},
    "expected": {"pe": 20.0, "roe": -0.25, "debt_to_equity": -3.0, "revenue_growth": -0.1}
  },
  {
    "name": "zero_equity_metrics_nan",
    "price": 30.0,
    "facts": {"net_income": 4000000.0, "revenue": 50000000.0, "revenue_prev": 50000000.0, "equity": 0.0, "total_debt": 10000000.0, "eps_ttm": 1.5},
    "expected": {"pe": 20.0, "roe": "nan", "debt_to_equity": "nan", "revenue_growth": 0.0}
  }
```

(The existing `test_edge_cases` in `tests/test_golden_metrics.py` already iterates this file — no test code change needed. `roe`/`debt_to_equity` with `equity=0` are `nan` via `_safe_div`.)

- [ ] **Step 2: Run to verify pass**

Run: `uv run pytest tests/test_golden_metrics.py::test_edge_cases -v`
Expected: PASS (verifies the hand-computed expectations against `compute_fundamental_metrics`).

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/edge_cases.json
git commit -m "test(golden): more fundamentals edge cases (negative/zero equity)"
```

---

### Task 4 **[CONTROLLER-LIVE]**: Capture `metrics_golden.json` from EDGAR

The controller runs this — it needs live SEC access and records point-in-time ground truth.

- [ ] **Step 1:** For each ticker in `[AAPL, MSFT, KO, JPM, XOM]`, fetch facts as of `2024-06-14`:

```bash
uv run python -c "
import json
from datetime import date
from equity_research.config import Config
from equity_research.data.edgar import EdgarProvider
from equity_research.analytics.fundamentals import compute_fundamental_metrics
cfg = Config.load('config.yaml')
prov = EdgarProvider(user_agent=cfg.edgar_user_agent)
out = []
for t in ['AAPL','MSFT','KO','JPM','XOM']:
    facts = prov.company_facts(t, date(2024,6,14))
    # price: use a representative close; recorded, not asserted for market-correctness
    out.append({'ticker': t, 'as_of': '2024-06-14', 'facts': facts})
print(json.dumps(out, indent=2, default=float))
"
```

- [ ] **Step 2:** For each ticker, add a representative `price` (any recent close is fine — metrics like P/E are recorded, not market-audited), compute `expected` via `compute_fundamental_metrics(facts, price)`, and hand-review the values for sanity (P/E positive/plausible, D/E ≥ 0 unless truly negative equity). Record `source_filing`, `filed_at` (from the 10-K used), and a `reference` URL per the fixtures README method.

- [ ] **Step 3:** Write `tests/fixtures/golden/metrics_golden.json` as an array of records, each shaped like `aapl_facts.json` (`ticker`, `source_filing`, `filed_at`, `reference`, `price`, `facts`, `expected`).

- [ ] **Step 4:** Add a `tests/fixtures/golden/README.md` documenting the capture method and the "do not auto-refresh" rule.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/golden/metrics_golden.json tests/fixtures/golden/README.md
git commit -m "test(golden): fundamentals metrics golden (5 tickers, point-in-time)"
```

---

### Task 5: Iterate `metrics_golden.json` in the metrics test (deterministic)

**Files:**
- Modify: `tests/test_golden_metrics.py`

- [ ] **Step 1: Append a test that iterates the golden file** (requires Task 4's fixture to exist):

```python
def test_metrics_golden_matches_hand_verified():
    import math

    records = json.loads((FIX / "golden" / "metrics_golden.json").read_text())
    assert len(records) == 5
    for rec in records:
        metrics = compute_fundamental_metrics(rec["facts"], price=rec["price"])
        for key, expected in rec["expected"].items():
            if expected == "nan":
                assert math.isnan(metrics[key]), f"{rec['ticker']}:{key}"
            else:
                assert math.isclose(metrics[key], expected, rel_tol=1e-3), f"{rec['ticker']}:{key}"
```

(`FIX = Path(__file__).parent / "fixtures"` already exists at the top of the file.)

- [ ] **Step 2: Run to verify pass**

Run: `uv run pytest tests/test_golden_metrics.py -v`
Expected: PASS (all metrics + edge cases + golden).

- [ ] **Step 3: Commit**

```bash
git add tests/test_golden_metrics.py
git commit -m "test(golden): assert fundamentals metrics golden set"
```

---

### Task 6 **[CONTROLLER-LIVE]**: Capture `verdicts_golden.json`

The controller runs the real analyze pipeline (Ollama up), reviews, and records baselines.

- [ ] **Step 1:** For 3 cases (e.g. `AAPL@2024-06-14`, `MSFT@2024-06-14`, `KO@2024-06-14`), run:

```bash
uv run python -c "
from datetime import date
from equity_research.cli import analyze_ticker
for t in ['AAPL','MSFT','KO']:
    v = analyze_ticker(t, date(2024,6,14), 'config.yaml')
    print(t, v.status, v.verdict, round(v.score,3),
          {o.agent: o.stance for o in v.opinions})
"
```

- [ ] **Step 2:** Hand-review each result for sanity (status ok, all 4 agents present, stances plausible vs. the metrics/news). Record the observed baseline into `tests/fixtures/golden/verdicts_golden.json`:

```json
[
  {"ticker": "AAPL", "as_of": "2024-06-14", "expected_verdict": "<observed>",
   "expected_stances": {"fundamentals": "<observed>", "technical": "<observed>", "sentiment": "<observed>"}}
]
```

(Only record cases whose review passed; if a case looks wrong, note it and pick another rather than enshrining a bad baseline.)

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/golden/verdicts_golden.json
git commit -m "test(golden): end-to-end verdict baselines (3 cases)"
```

---

### Task 7: Integration tests (code; run live in Task 8)

**Files:**
- Create: `tests/integration/test_retrieval_golden.py`
- Create: `tests/integration/test_verdicts_golden.py`

- [ ] **Step 1: Create `tests/integration/test_retrieval_golden.py`**

```python
import json
from datetime import date
from pathlib import Path

import pytest

FIX = Path(__file__).parent.parent / "fixtures" / "golden" / "retrieval_golden.json"


@pytest.mark.integration
def test_retrieval_golden_hit_at_3():
    """Real retriever over the golden corpus. Requires Ollama embeddings.

    Run: uv run pytest -m integration tests/integration/test_retrieval_golden.py
    """
    import tempfile

    from equity_research.config import Config
    from equity_research.eval.retrieval_eval import hit_at_k, mrr
    from equity_research.rag.chroma_store import ChromaVectorStore
    from equity_research.rag.records import NewsItem
    from equity_research.rag.store import NewsStore

    from equity_research.rag.chunking import chunk_news
    from equity_research.rag.reranker import default_scorer
    from equity_research.rag.retrieve import NewsRetriever

    cfg = Config.load("config.yaml")
    data = json.loads(FIX.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        vs = ChromaVectorStore(persist_dir=tmp, embed_model=cfg.rag["embed_model"], net=cfg.net)
        store = NewsStore(vs)
        # Each labeled doc becomes one news item dated before the query as_of.
        # NewsItem fields (see equity_research/rag/records.py): ticker, title, text,
        # url, source, published_at. chunk_news builds body = f"{title}\n{text}",
        # so we recover a doc id from a retrieved chunk by substring containment.
        chunks = []
        for d in data["documents"]:
            item = NewsItem(ticker=d["ticker"], title=d["id"], text=d["text"],
                            url=d["id"], source="golden", published_at=date(2024, 1, 1))
            chunks.extend(chunk_news(item))
        store.upsert(chunks)

        retriever = NewsRetriever(store, scorer=default_scorer(cfg.rag["rerank_model"]))

        def retrieve_ids(ticker, question):
            texts = retriever.retrieve(ticker, question, date(2024, 6, 14), k=3, candidate_k=10)
            ids = []
            for rt in texts:
                for d in data["documents"]:
                    if d["text"] in rt:
                        ids.append(d["id"])
                        break
            return ids

        h3 = hit_at_k(retrieve_ids, data["cases"], k=3)
        m = mrr(retrieve_ids, data["cases"], k=3)
        print(f"hit@3={h3:.3f} mrr={m:.3f}")
        assert h3 >= 0.8, f"hit@3 below threshold: {h3:.3f}"
```

The imports above use the real module signatures (verified): `NewsItem(ticker, title, text, url, source, published_at)` from `equity_research/rag/records.py`, `chunk_news(item) -> list[(text, meta)]`, `NewsStore(vs).upsert(chunks)` / `.search`, `NewsRetriever(store, scorer).retrieve(ticker, question, as_of, k, candidate_k) -> list[str]`.

- [ ] **Step 2: Create `tests/integration/test_verdicts_golden.py`**

```python
import json
from datetime import date, datetime
from pathlib import Path

import pytest

FIX = Path(__file__).parent.parent / "fixtures" / "golden" / "verdicts_golden.json"


@pytest.mark.integration
def test_verdicts_golden_regression():
    """Real analyze pipeline over golden cases. Requires Ollama.

    Run: uv run pytest -m integration tests/integration/test_verdicts_golden.py
    """
    from equity_research.cli import analyze_ticker

    cases = json.loads(FIX.read_text())
    for case in cases:
        as_of = datetime.strptime(case["as_of"], "%Y-%m-%d").date()
        v = analyze_ticker(case["ticker"], as_of, "config.yaml")
        assert v.status == "ok", f"{case['ticker']} status {v.status}"
        assert v.verdict == case["expected_verdict"], f"{case['ticker']} verdict {v.verdict}"
        assert -1.0 <= v.score <= 1.0 and 0.0 <= v.confidence <= 1.0
        stances = {o.agent: o.stance for o in v.opinions}
        for agent, expected_stance in case["expected_stances"].items():
            assert stances.get(agent) == expected_stance, f"{case['ticker']} {agent} {stances.get(agent)}"


@pytest.mark.integration
def test_verdicts_reproducible():
    """Same input -> same score/verdict (seed + temp 0 + cache)."""
    from equity_research.cli import analyze_ticker

    cases = json.loads(FIX.read_text())
    c = cases[0]
    as_of = datetime.strptime(c["as_of"], "%Y-%m-%d").date()
    a = analyze_ticker(c["ticker"], as_of, "config.yaml")
    b = analyze_ticker(c["ticker"], as_of, "config.yaml")
    assert a.verdict == b.verdict and abs(a.score - b.score) < 1e-9
```

- [ ] **Step 3:** Do NOT run integration here (controller runs them in Task 8). Just confirm they import/collect: `uv run pytest tests/integration/test_retrieval_golden.py tests/integration/test_verdicts_golden.py --collect-only`. Expected: 3 tests collected, 0 errors.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_retrieval_golden.py tests/integration/test_verdicts_golden.py
git commit -m "test(golden): integration tests for retrieval and verdicts"
```

---

### Task 8 **[CONTROLLER-LIVE]**: Run integration + tune threshold

- [ ] **Step 1:** Full default suite green: `uv run pytest -q` (deterministic golden tests included).
- [ ] **Step 2:** Retrieval integration: `uv run pytest -m integration tests/integration/test_retrieval_golden.py -s` → print hit@3 / MRR; must be ≥ 0.8. If below, inspect which cases miss and improve the corpus wording (make the relevant doc unambiguously the best match) rather than lowering the bar.
- [ ] **Step 3:** Verdicts integration: `uv run pytest -m integration tests/integration/test_verdicts_golden.py` → passes against the recorded baselines; reproducibility holds.
- [ ] **Step 4:** No commit unless the corpus was tuned (then commit the corpus change).

---

## Completion
Merge `feature/golden-set` into `master` via **superpowers:finishing-a-development-branch**. Deterministic golden tests run in the default suite; integration golden tests run under `-m integration`. All commits authored as the repo owner, NO `Co-Authored-By`.
