# Phase 2.5 — Filing-text RAG (ParentDocument) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commit attribution:** commit ONLY as the repo author (OleksandrFIT). Do NOT add any `Co-Authored-By` trailer.

**Goal:** Index 10-K Item 1A (Risk Factors) + Item 7 (MD&A) into RAG using a ParentDocument scheme (small child chunks for matching, whole sections returned to the LLM) so the Fundamentals agent complements its numbers with point-in-time qualitative filing text.

**Architecture:** Child chunks live in the existing Chroma collection (filtered by `doc_type`); whole-section parents live in a file-backed `ParentStore`. `FilingStore` ties them together (upsert + MMR search → dedup parents → parent texts); a `FilingRetriever` reranks and trims. `FundamentalsAgent` gains optional filing ingest/retrieve. Extraction is a thin edgartools wrapper verified live.

**Tech Stack:** existing (edgartools, langchain-text-splitters, chromadb, pydantic, pytest). No new deps.

**Scope:** Plan 2.5 only. Spec: `docs/superpowers/specs/2026-09-15-phase-2-5-filing-text-design.md`.

## File Structure

- `equity_research/rag/parent_store.py`, `filing_records.py`, `filing_store.py`, `filing_retrieve.py`, `filings.py`.
- `equity_research/rag/store.py` — NewsStore `doc_type=news` filter.
- `equity_research/agents/fundamentals.py` — optional filing ingest/retrieve.
- `equity_research/config.py` + `config.yaml` — filing rag params.
- `equity_research/cli.py` — ingest filings + wire fundamentals filing retriever.
- Tests mirror each; a live integration test.

Naming contract: `ParentStore(dir)` (`put`/`get`/`has`), `FilingSection`, `filing_parent_id`, `filing_metadata`, `chunk_filing(section) -> (parent_id, parent_text, child_chunks)`, `FilingStore(vector_store, parent_store)` (`upsert`/`search`), `FilingRetriever(store, scorer)` (`retrieve`), `fetch_filing_sections(ticker, as_of, user_agent)`.

---

## Task 1: ParentStore

**Files:** Create `equity_research/rag/parent_store.py`; Test `tests/test_parent_store.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_parent_store.py`:
```python
from equity_research.rag.parent_store import ParentStore


def test_put_get_roundtrip(tmp_path):
    ps = ParentStore(tmp_path)
    ps.put("pid1", "long section text")
    assert ps.get("pid1") == "long section text"
    assert ps.has("pid1")
    assert not ps.has("missing")


def test_persists_across_instances(tmp_path):
    ParentStore(tmp_path).put("pid2", "body")
    assert ParentStore(tmp_path).get("pid2") == "body"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parent_store.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/parent_store.py`:
```python
from __future__ import annotations

from pathlib import Path


class ParentStore:
    """File-backed store mapping a parent id to its full section text."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, parent_id: str) -> Path:
        return self.dir / f"{parent_id}.txt"

    def put(self, parent_id: str, text: str) -> None:
        self._path(parent_id).write_text(text)

    def get(self, parent_id: str) -> str:
        return self._path(parent_id).read_text()

    def has(self, parent_id: str) -> bool:
        return self._path(parent_id).exists()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parent_store.py -v`
Expected: PASS (2 tests). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/parent_store.py tests/test_parent_store.py
git commit -m "feat(rag): ParentStore (file-backed parent-document store)"
```

---

## Task 2: Filing records + chunking

**Files:** Create `equity_research/rag/filing_records.py`; Test `tests/test_filing_records.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_filing_records.py`:
```python
from datetime import date

from equity_research.rag.filing_records import (
    FilingSection,
    chunk_filing,
    filing_metadata,
    filing_parent_id,
)


def _section(text, section="risk_factors"):
    return FilingSection(ticker="AAPL", section=section, filed_at=date(2023, 11, 3), form="10-K", text=text)


def test_parent_id_stable_and_section_sensitive():
    a = filing_parent_id(_section("x"))
    b = filing_parent_id(_section("y"))  # same identity fields, different text
    c = filing_parent_id(_section("x", section="mda"))
    assert a == b            # id depends on identity, not body
    assert a != c            # different section -> different id


def test_metadata_shape():
    m = filing_metadata(_section("x"))
    assert m["doc_type"] == "filing_section"
    assert m["ticker"] == "AAPL"
    assert m["date_int"] == 20231103
    assert m["section"] == "risk_factors"


def test_chunk_short_is_single_child_with_parent_id():
    pid, parent_text, children = chunk_filing(_section("Short risk section."))
    assert parent_text == "Short risk section."
    assert len(children) == 1
    _, meta = children[0]
    assert meta["parent_id"] == pid
    assert meta["doc_type"] == "filing_section"


def test_chunk_long_splits_into_multiple_children():
    long_text = " ".join(f"sentence{i}." for i in range(600))
    pid, parent_text, children = chunk_filing(_section(long_text))
    assert len(children) > 1
    assert all(m["parent_id"] == pid for _, m in children)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_filing_records.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/filing_records.py`:
```python
from __future__ import annotations

import hashlib
from datetime import date

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

# ~500 tokens ≈ ~2000 chars per child chunk for precise matching.
_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)


class FilingSection(BaseModel):
    ticker: str
    section: str  # "risk_factors" | "mda"
    filed_at: date
    form: str
    text: str


def filing_parent_id(section: FilingSection) -> str:
    raw = f"{section.ticker}|{section.form}|{section.section}|{section.filed_at.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def filing_metadata(section: FilingSection) -> dict:
    return {
        "doc_type": "filing_section",
        "ticker": section.ticker,
        "date_int": int(section.filed_at.strftime("%Y%m%d")),
        "form": section.form,
        "section": section.section,
        "content_hash": filing_parent_id(section),
    }


def chunk_filing(section: FilingSection) -> tuple[str, str, list[tuple[str, dict]]]:
    pid = filing_parent_id(section)
    base = filing_metadata(section)
    children = _splitter.split_text(section.text) or [section.text]
    child_chunks = [(c, {**base, "parent_id": pid}) for c in children]
    return pid, section.text, child_chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_filing_records.py -v`
Expected: PASS (4 tests). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/filing_records.py tests/test_filing_records.py
git commit -m "feat(rag): filing records + ParentDocument chunking"
```

---

## Task 3: FilingStore

**Files:** Create `equity_research/rag/filing_store.py`; Test `tests/test_filing_store.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_filing_store.py`:
```python
from datetime import date

from equity_research.rag.filing_records import FilingSection
from equity_research.rag.filing_store import FilingStore
from equity_research.rag.parent_store import ParentStore


class FakeVectorStore:
    def __init__(self):
        self.docs = {}
        self.last_where = None

    def existing_ids(self, ids):
        return {i for i in ids if i in self.docs}

    def add(self, ids, texts, metadatas):
        for i, t, m in zip(ids, texts, metadatas):
            self.docs[i] = (t, m)

    def mmr_search(self, query, where, k):
        self.last_where = where
        return [(t, m) for (t, m) in self.docs.values()][:k]


def _section(text, section="risk_factors"):
    return FilingSection(ticker="AAPL", section=section, filed_at=date(2023, 11, 3), form="10-K", text=text)


def test_upsert_stores_parent_and_children(tmp_path):
    vs = FakeVectorStore()
    ps = ParentStore(tmp_path)
    store = FilingStore(vs, ps)
    store.upsert([_section("A risk factors section with some length.")])
    assert len(vs.docs) >= 1                 # at least one child chunk indexed
    # the parent text is retrievable from the parent store
    pid = next(iter(vs.docs.values()))[1]["parent_id"]
    assert ps.get(pid).startswith("A risk factors")


def test_search_filters_doc_type_and_returns_parent_texts(tmp_path):
    vs = FakeVectorStore()
    ps = ParentStore(tmp_path)
    store = FilingStore(vs, ps)
    store.upsert([_section("Risk factors body text here.")])
    hits = store.search("risks", ticker="AAPL", as_of=date(2024, 6, 1), candidate_k=8)
    assert vs.last_where == {"$and": [
        {"doc_type": {"$eq": "filing_section"}},
        {"ticker": {"$eq": "AAPL"}},
        {"date_int": {"$lte": 20240601}},
    ]}
    assert len(hits) == 1                      # one parent, deduped
    assert hits[0][0] == "Risk factors body text here."  # parent text returned
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_filing_store.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/filing_store.py`:
```python
from __future__ import annotations

from datetime import date

from equity_research.rag.filing_records import chunk_filing


class FilingStore:
    def __init__(self, vector_store, parent_store):
        self.vs = vector_store
        self.ps = parent_store

    def upsert(self, sections) -> None:
        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict] = []
        for section in sections:
            pid, parent_text, children = chunk_filing(section)
            self.ps.put(pid, parent_text)
            for i, (ctext, cmeta) in enumerate(children):
                ids.append(f"{pid}:{i}")
                texts.append(ctext)
                metas.append(cmeta)
        if not ids:
            return
        existing = self.vs.existing_ids(ids)
        keep = [(i, t, m) for i, t, m in zip(ids, texts, metas) if i not in existing]
        if keep:
            self.vs.add([i for i, _, _ in keep], [t for _, t, _ in keep], [m for _, _, m in keep])

    def search(self, query: str, ticker: str, as_of: date, candidate_k: int) -> list[tuple[str, dict]]:
        where = {"$and": [
            {"doc_type": {"$eq": "filing_section"}},
            {"ticker": {"$eq": ticker}},
            {"date_int": {"$lte": int(as_of.strftime("%Y%m%d"))}},
        ]}
        hits = self.vs.mmr_search(query, where, candidate_k)
        out: list[tuple[str, dict]] = []
        seen: set[str] = set()
        for _text, meta in hits:
            pid = meta.get("parent_id")
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            out.append((self.ps.get(pid), meta))
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_filing_store.py -v`
Expected: PASS (2 tests). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/filing_store.py tests/test_filing_store.py
git commit -m "feat(rag): FilingStore (ParentDocument upsert + parent retrieval)"
```

---

## Task 4: NewsStore doc_type filter

**Files:** Modify `equity_research/rag/store.py`; Modify `tests/test_rag_store.py`.

- [ ] **Step 1: Update the failing test**

In `tests/test_rag_store.py`, the `test_search_builds_ticker_and_date_filter` asserts the `where`. Update its expected `where` to include the doc_type clause first:
```python
    assert where == {"$and": [
        {"doc_type": {"$eq": "news"}},
        {"ticker": {"$eq": "AAPL"}},
        {"date_int": {"$lte": 20260915}},
    ]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_store.py::test_search_builds_ticker_and_date_filter -v`
Expected: FAIL — current `where` has no doc_type clause.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/rag/store.py`, in `NewsStore.search`, change the `where` to include doc_type:
```python
        where = {"$and": [
            {"doc_type": {"$eq": "news"}},
            {"ticker": {"$eq": ticker}},
            {"date_int": {"$lte": int(as_of.strftime("%Y%m%d"))}},
        ]}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_store.py -v`
Expected: PASS. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/store.py tests/test_rag_store.py
git commit -m "feat(rag): NewsStore filters doc_type=news (shared collection)"
```

---

## Task 5: FilingRetriever

**Files:** Create `equity_research/rag/filing_retrieve.py`; Test `tests/test_filing_retrieve.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_filing_retrieve.py`:
```python
from datetime import date

from equity_research.rag.filing_retrieve import FilingRetriever


class FakeFilingStore:
    def __init__(self, results):
        self._results = results
        self.searched = None

    def search(self, query, ticker, as_of, candidate_k):
        self.searched = (query, ticker, as_of, candidate_k)
        return self._results


def test_retrieve_reranks_and_trims(tmp_path):
    cands = [("risk A", {}), ("risk B", {}), ("risk C", {})]
    store = FakeFilingStore(cands)
    scores = {"risk A": 0.1, "risk B": 0.9, "risk C": 0.5}
    r = FilingRetriever(store, scorer=lambda q, ts: [scores[t] for t in ts])
    out = r.retrieve("AAPL", "risks", as_of=date(2024, 6, 1), k=2, candidate_k=3)
    assert out == ["risk B", "risk C"]
    assert store.searched[3] == 3


def test_retrieve_without_scorer_keeps_order():
    store = FakeFilingStore([("a", {}), ("b", {})])
    r = FilingRetriever(store, scorer=None)
    assert r.retrieve("AAPL", "q", as_of=date(2024, 6, 1), k=5, candidate_k=8) == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_filing_retrieve.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/filing_retrieve.py`:
```python
from __future__ import annotations

from datetime import date

from equity_research.rag.reranker import Scorer, rerank


class FilingRetriever:
    def __init__(self, store, scorer: Scorer | None):
        self.store = store
        self.scorer = scorer

    def retrieve(self, ticker: str, query: str, as_of: date, k: int, candidate_k: int) -> list[str]:
        candidates = self.store.search(query, ticker, as_of, candidate_k)
        reranked = rerank(query, candidates, self.scorer)
        return [text for text, _ in reranked[:k]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_filing_retrieve.py -v`
Expected: PASS (2 tests). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/filing_retrieve.py tests/test_filing_retrieve.py
git commit -m "feat(rag): FilingRetriever (search -> rerank -> top-k section texts)"
```

---

## Task 6: FundamentalsAgent filing context

**Files:** Modify `equity_research/agents/fundamentals.py`; Test `tests/test_agents.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agents.py`:
```python
def test_fundamentals_gather_adds_filing_context(tmp_path):
    closes = [float(x) for x in range(100, 360)]
    provider = PriceProvider(fetch_yfinance=lambda t: _prices(closes), fetch_stooq=lambda t: _prices(closes))
    facts = {"net_income": 100.0, "revenue": 1000.0, "revenue_prev": 900.0,
             "equity": 500.0, "total_debt": 250.0, "eps_ttm": 5.0}

    class FakeFilingRetriever:
        def retrieve(self, ticker, query, as_of, k, candidate_k):
            return ["Risk factors: supply concentration."]

    ingested = {}
    agent = FundamentalsAgent(
        facts_source=type("F", (), {"company_facts": staticmethod(lambda t, as_of: facts)})(),
        prices=provider, client=_client(tmp_path),
        filing_retriever=FakeFilingRetriever(), filing_ingest_fn=lambda t: ingested.setdefault(t, True),
    )
    ev = agent.gather("AAPL", as_of=date(2025, 9, 1))
    assert "pe" in ev.metrics
    assert any("supply concentration" in c for c in ev.context)
    assert ingested["AAPL"] is True
```
(`_prices`, `_client` helpers already exist in `tests/test_agents.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agents.py::test_fundamentals_gather_adds_filing_context -v`
Expected: FAIL — `FundamentalsAgent` has no `filing_retriever`/`filing_ingest_fn`.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/agents/fundamentals.py`, extend `FundamentalsAgent`:
```python
_FILING_QUERY = "risk factors, financial health, business outlook and liquidity"


class FundamentalsAgent:
    name = "fundamentals"

    def __init__(self, facts_source, prices, client,
                 filing_retriever=None, filing_ingest_fn=None, k: int = 3, candidate_k: int = 12):
        self.facts_source = facts_source
        self.prices = prices
        self.client = client
        self.filing_retriever = filing_retriever
        self.filing_ingest_fn = filing_ingest_fn
        self.k = k
        self.candidate_k = candidate_k

    def gather(self, ticker: str, as_of: date) -> Evidence:
        hist, note = self.prices.history(ticker, as_of)
        price = float(hist["Close"].iloc[-1])
        facts = self.facts_source.company_facts(ticker, as_of)
        metrics = compute_fundamental_metrics(facts, price=price)
        context: list[str] = []
        if self.filing_ingest_fn is not None:
            self.filing_ingest_fn(ticker)
        if self.filing_retriever is not None:
            context = self.filing_retriever.retrieve(ticker, _FILING_QUERY, as_of, self.k, self.candidate_k)
        notes = [note] if note else []
        return Evidence(ticker=ticker, as_of=as_of, metrics=metrics, context=context, notes=notes)

    def judge(self, evidence: Evidence) -> AgentOpinion:
        return judge_evidence(self.name, evidence, self.client)
```
(Keep the existing imports; the type hints for the injected params are intentionally loose.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agents.py tests/test_fundamentals.py -v`
Expected: PASS (new + existing; the existing fundamentals test constructs the agent without filing params, which now default to None → context empty → still green). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/agents/fundamentals.py tests/test_agents.py
git commit -m "feat(agents): Fundamentals gathers point-in-time filing context"
```

---

## Task 7: Extraction + config + CLI wiring

**Files:** Create `equity_research/rag/filings.py`; Modify `equity_research/config.py`, `config.yaml`, `equity_research/cli.py`; Test `tests/test_cli.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:
```python
def test_analyze_ticker_fundamentals_has_filing_retriever(monkeypatch):
    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator):
            captured["agents"] = agents

        def run(self, ticker, as_of):
            from equity_research.orchestration.aggregator import Verdict
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    cli_module.analyze_ticker("AAPL", date(2026, 9, 15), "config.yaml")
    fund = next(a for a in captured["agents"] if a.name == "fundamentals")
    assert fund.filing_retriever is not None
    assert fund.filing_ingest_fn is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_analyze_ticker_fundamentals_has_filing_retriever -v`
Expected: FAIL — fundamentals built without filing wiring.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/filings.py` (thin extraction; verified live in Task 8):
```python
from __future__ import annotations

from datetime import date

from equity_research.data.edgar import select_filing_asof
from equity_research.rag.filing_records import FilingSection


def fetch_filing_sections(ticker: str, as_of: date, user_agent: str) -> list[FilingSection]:
    """Extract Item 1A (Risk Factors) + Item 7 (MD&A) from the latest 10-K <= as_of.

    edgartools section attribute names are version-dependent — verify live and
    adjust only this function if they differ.
    """
    from edgar import Company, set_identity

    set_identity(user_agent)
    filings = Company(ticker).get_filings(form=["10-K"])
    filing = select_filing_asof(list(filings), as_of)
    tenk = filing.obj()
    filed = filing.filing_date
    out: list[FilingSection] = []
    for attr, name in [("risk_factors", "risk_factors"), ("management_discussion", "mda")]:
        text = getattr(tenk, attr, None)
        if text:
            out.append(FilingSection(ticker=ticker, section=name, filed_at=filed, form="10-K", text=str(text)))
    return out
```

In `equity_research/config.py`, extend the `rag` default dict with filing params (add these keys inside the existing `rag` Field default):
```python
        "filing_retrieve_k": 3,
        "filing_candidate_k": 12,
        "parent_dir": ".parents",
```
And add the same three keys under `rag:` in `config.yaml`.

In `equity_research/cli.py`, add a helper that builds the filing pieces and wire Fundamentals in BOTH `analyze_ticker` and `build_backtest_verdict`. Add imports:
```python
from equity_research.rag.filing_retrieve import FilingRetriever
from equity_research.rag.filing_store import FilingStore
from equity_research.rag.filings import fetch_filing_sections
from equity_research.rag.parent_store import ParentStore
```
Add a builder:
```python
def _build_filing_pieces(cfg, news_store_vs):
    parent_store = ParentStore(cfg.rag["parent_dir"])
    filing_store = FilingStore(news_store_vs, parent_store)
    retriever = FilingRetriever(filing_store, scorer=default_scorer(cfg.rag["rerank_model"]))
    ingest_fn = lambda t: filing_store.upsert(
        resilient(fetch_filing_sections, cfg.net)(t, date.today(), cfg.edgar_user_agent))
    return retriever, ingest_fn
```
Note `news_store_vs` is the `ChromaVectorStore` instance (the same collection). In `analyze_ticker`, the code already builds `ChromaVectorStore(...)` inside `NewsStore(...)`. Refactor so the `ChromaVectorStore` is a named local (e.g. `vs = ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"], embed_model=cfg.rag["embed_model"])`, then `news_store = NewsStore(vs)`), and pass `vs` to `_build_filing_pieces`. Then build fundamentals with the filing pieces:
```python
    filing_retriever, filing_ingest = _build_filing_pieces(cfg, vs)
    ...
        FundamentalsAgent(facts_source=edgar, prices=prices, client=client,
                          filing_retriever=filing_retriever, filing_ingest_fn=filing_ingest,
                          k=cfg.rag["filing_retrieve_k"], candidate_k=cfg.rag["filing_candidate_k"]),
```
Do the same wiring in `build_backtest_verdict` (build a `ChromaVectorStore` there too and pass filing pieces to its `FundamentalsAgent`). For backtest, `fetch_filing_sections` must use `as_of` (not today) — use a per-call closure: in the backtest fundamentals, the ingest happens inside `gather` which passes the run's `as_of` to the retriever's search (point-in-time), but the INGEST fetches sections as-of. Simplest correct approach: in `build_backtest_verdict`, the filing ingest closure ignores date (it upserts whatever the latest ≤ today 10-K is), while `FilingStore.search` filters `filed_at ≤ as_of` — so an old as_of only sees filings whose `date_int ≤ as_of`. Since the store accumulates filings across runs, and search is PIT-filtered, correctness holds as long as the as-of filing was ingested. To guarantee the as-of filing is present, the backtest ingest should fetch as-of: pass the run date. Implement the backtest ingest as `lambda t: filing_store.upsert(fetch_filing_sections(t, <as_of>, cfg.edgar_user_agent))` — but `filing_ingest_fn(ticker)` has no as_of. Since `FundamentalsAgent.gather(ticker, as_of)` calls `filing_ingest_fn(ticker)` without as_of, the ingest cannot see as_of. Accept the live-analyze default (fetch latest ≤ today) for ingest, and rely on `search`'s `filed_at ≤ as_of` filter for point-in-time selection; for backtest dates within the last ~1-2 years the latest 10-K's filed_at may be AFTER the as_of and thus filtered out (no filing context for that as_of) — that is acceptable and honest (older as_of simply has no ingested older filing). Note this limitation in the report/comment; a fuller as-of filing ingest is a later enhancement.

Also update the standalone `ingest` command to ingest filings too:
```python
@app.command()
def ingest(ticker: str, config: str = "config.yaml"):
    cfg = Config.load(config)
    vs = ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"], embed_model=cfg.rag["embed_model"])
    n = ingest_news(resilient(fetch_news, cfg.net), NewsStore(vs), ticker.upper())
    sections = resilient(fetch_filing_sections, cfg.net)(ticker.upper(), date.today(), cfg.edgar_user_agent)
    FilingStore(vs, ParentStore(cfg.rag["parent_dir"])).upsert(sections)
    typer.echo(f"Ingested {n} news items and {len(sections)} filing sections for {ticker.upper()}")
```
Add `.parents/` to `.gitignore`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS. Confirm `uv run python -c "import equity_research.cli, equity_research.rag.filings"` works offline. Full suite `uv run pytest -m "not integration" -p no:warnings -q` green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/filings.py equity_research/config.py config.yaml equity_research/cli.py .gitignore tests/test_cli.py
git commit -m "feat: wire filing extraction + Fundamentals filing retriever + ingest"
```

---

## Task 8: Live verification

**Files:** Create `tests/integration/test_filings_live.py`.

- [ ] **Step 1: Write the (marked) integration test**

`tests/integration/test_filings_live.py`:
```python
from datetime import date

import pytest


@pytest.mark.integration
def test_fetch_filing_sections_live():
    """Requires network (SEC). Run: uv run pytest -m integration."""
    from equity_research.rag.filings import fetch_filing_sections

    sections = fetch_filing_sections("AAPL", date.today(), "Equity Research test@example.com")
    assert len(sections) >= 1
    assert all(len(s.text) > 200 for s in sections)  # real section bodies, not stubs
    assert {s.section for s in sections} <= {"risk_factors", "mda"}
```

- [ ] **Step 2: Confirm deselected by default**

Run: `uv run pytest -m "not integration" -q`
Expected: this test is NOT run. Do not run `-m integration` until Step 3.

- [ ] **Step 3: Live verification**

Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run pytest tests/integration/test_filings_live.py -m integration -q`
Expected: PASS. If edgartools exposes the sections under different attribute names (not `risk_factors` / `management_discussion`), fix ONLY `fetch_filing_sections` in `equity_research/rag/filings.py` (inspect the `TenK` object's attributes live), then re-run. The pure `chunk_filing` / `FilingStore` logic must not change.

Then confirm the full analyze surfaces filing context (needs Ollama + `nomic-embed-text`):
Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run python -m equity_research.cli ingest AAPL`
Then: `DISABLE_PANDERA_IMPORT_WARNING=True uv run python -m equity_research.cli analyze AAPL`
Expected: ingest reports N filing sections; analyze still renders a fundamentals opinion (now informed by filing text — its rationale/key_facts may reference risk factors or MD&A). If a wrapper issue appears, fix only `filings.py`.

- [ ] **Step 4: Commit any fix**

```bash
git add -A
git commit -m "test(rag): live filing-section extraction"
```
(Include any `filings.py` fix here.)

---

## Self-Review (completed during authoring)

- **Spec coverage (2.5):** ParentStore (Task 1); filing records + ParentDocument chunking (Task 2); FilingStore upsert/search with doc_type filter + parent retrieval (Task 3); NewsStore doc_type=news filter for the shared collection (Task 4); FilingRetriever rerank/trim (Task 5); Fundamentals optional filing ingest/retrieve into context, point-in-time (Task 6); extraction + config + CLI/backtest wiring + ingest-filings (Task 7); live verification (Task 8). Point-in-time: `FilingStore.search` filters `date_int ≤ as_of`; a documented limitation on backtest ingest-as-of is noted in Task 7.
- **Placeholder scan:** none. `filings.py` carries real code + an explicit "verify live, fix only this file" note (EdgarProvider pattern).
- **Type consistency:** `FilingSection`, `filing_parent_id`, `filing_metadata`, `chunk_filing -> (parent_id, parent_text, child_chunks)`, `ParentStore.put/get/has`, `FilingStore(vector_store, parent_store).upsert/search`, `FilingRetriever(store, scorer).retrieve(ticker, query, as_of, k, candidate_k)`, `fetch_filing_sections(ticker, as_of, user_agent)`, and the `FundamentalsAgent(..., filing_retriever, filing_ingest_fn, k, candidate_k)` constructor are consistent across tasks and reuse existing pieces (`select_filing_asof`, `rerank`, `ChromaVectorStore`, `default_scorer`, `resilient`).
