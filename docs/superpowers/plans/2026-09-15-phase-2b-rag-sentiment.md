# Phase 2B — RAG infra + Sentiment agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent news-RAG layer (Chroma + local embeddings + MMR retrieval + optional cross-encoder rerank) and a live-only Sentiment agent that judges retrieved news, plus a grounding-check guardrail that drops unsupported `key_facts`.

**Architecture:** All retrieval logic sits behind small protocols so it is unit-testable with in-memory fakes; the real Chroma / embeddings / reranker / news-fetch are thin wrappers verified live (same pattern as EdgarProvider in Phase 1). The Sentiment agent gathers news into `Evidence.context` and reuses the existing `judge_evidence` LLM step (with the untrusted-content delimiting already in place). Grounding-check runs after every agent's LLM parse.

**Tech Stack:** existing + `langchain-chroma`, `langchain-ollama`, `langchain-text-splitters`, `chromadb`, `feedparser` (core); `sentence-transformers` (optional `rerank` extra — pulls torch). Local embedding model `nomic-embed-text` via Ollama.

**Scope:** Plan 2B only. Filing-text RAG is Phase 2.5; backtest is Phase 3. Spec: `docs/superpowers/specs/2026-09-15-phase-2-risk-rag-sentiment-design.md`.

## File Structure

- `equity_research/rag/__init__.py`
- `equity_research/rag/records.py` — `NewsItem`, `content_hash`, `news_metadata` (pure).
- `equity_research/rag/chunking.py` — `chunk_news(item)` (pure, uses text splitter).
- `equity_research/rag/store.py` — `VectorStore` protocol + `NewsStore` (pure logic: dedup, where-filter, mmr).
- `equity_research/rag/reranker.py` — `rerank(query, items, scorer)` + `default_scorer()` (optional torch).
- `equity_research/rag/retrieve.py` — `NewsRetriever` (search → rerank → texts).
- `equity_research/rag/ingest.py` — `ingest_news(fetch, store, splitter, ticker)` (fetch → chunk → upsert).
- `equity_research/rag/chroma_store.py` — `ChromaVectorStore` (thin Chroma wrapper, integration).
- `equity_research/rag/adapters.py` — `fetch_news(ticker)` (yfinance/RSS, thin, integration).
- `equity_research/agents/grounding.py` — `ground(opinion, evidence)` (pure).
- `equity_research/agents/sentiment.py` — `SentimentAgent`.
- Modify `equity_research/agents/judge.py` — apply grounding after parse.
- Modify `equity_research/agents/prompts.py` — add `sentiment` role.
- Modify `equity_research/config.py` + `config.yaml` — `rag` block.
- Modify `equity_research/cli.py` — `ingest` command + wire Sentiment + auto-ingest.
- `pyproject.toml` — deps.
- Tests mirror each module.

Naming contract (do not rename): `NewsItem`, `content_hash(item)`, `news_metadata(item)`, `chunk_news(item) -> list[tuple[str, dict]]`, `VectorStore` (`add`, `existing_ids`, `mmr_search`), `NewsStore(store)` (`.upsert(chunks)`, `.search(query, ticker, as_of, k)`), `rerank(query, items, scorer)`, `NewsRetriever(store, reranker)` (`.retrieve(ticker, query, as_of, k)`), `ingest_news(fetch, news_store, ticker)`, `ground(opinion, evidence)`, `SentimentAgent(retriever, ingest_fn, client)`.

---

## Task 1: Dependencies + config rag block

**Files:**
- Modify: `pyproject.toml`
- Modify: `equity_research/config.py`
- Modify: `config.yaml`
- Test: `tests/test_config.py` (add assertion)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:
```python
def test_config_has_rag_defaults():
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".cache",
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    assert cfg.rag["chroma_dir"] == ".chroma"
    assert cfg.rag["embed_model"] == "nomic-embed-text"
    assert cfg.rag["retrieve_k"] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_config_has_rag_defaults -v`
Expected: FAIL — `Config` has no `rag`.

- [ ] **Step 3: Write minimal implementation**

In `pyproject.toml`, add to `dependencies`:
```toml
    "langchain-chroma>=0.1",
    "langchain-ollama>=0.1",
    "langchain-text-splitters>=0.2",
    "chromadb>=0.5",
    "feedparser>=6.0",
```
And add an optional extra (keeps torch out of the base install):
```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
rerank = ["sentence-transformers>=3.0"]
```
Then `uv sync --extra dev`.

In `equity_research/config.py`, add a `rag` field to `Config` (after `risk`):
```python
    rag: dict = Field(default_factory=lambda: {
        "chroma_dir": ".chroma",
        "embed_model": "nomic-embed-text",
        "retrieve_k": 6,
        "candidate_k": 20,
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    })
```
(Change the `risk` field annotation `dict[str, float]` is fine; `rag` uses `dict` since values are mixed str/int.)

In `config.yaml`, add:
```yaml
rag:
  chroma_dir: .chroma
  embed_model: nomic-embed-text
  retrieve_k: 6
  candidate_k: 20
  rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

Add `.chroma/` to `.gitignore`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (existing config tests + new one).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock equity_research/config.py config.yaml .gitignore tests/test_config.py
git commit -m "chore(rag): add RAG deps (rerank optional) + config block"
```

---

## Task 2: News records + content hash + metadata

**Files:**
- Create: `equity_research/rag/__init__.py` (empty)
- Create: `equity_research/rag/records.py`
- Test: `tests/test_rag_records.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rag_records.py`:
```python
from datetime import date

from equity_research.rag.records import NewsItem, content_hash, news_metadata


def _item(**kw):
    base = dict(ticker="AAPL", title="Apple beats", text="Apple beat estimates.",
                url="http://x/1", source="yahoo", published_at=date(2026, 9, 10))
    base.update(kw)
    return NewsItem(**base)


def test_content_hash_stable_and_url_sensitive():
    a = content_hash(_item())
    b = content_hash(_item())
    c = content_hash(_item(url="http://x/2"))
    assert a == b
    assert a != c


def test_metadata_has_date_int_and_ticker():
    m = news_metadata(_item())
    assert m["doc_type"] == "news"
    assert m["ticker"] == "AAPL"
    assert m["date_int"] == 20260910  # YYYYMMDD int for range filtering
    assert m["content_hash"] == content_hash(_item())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_records.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/__init__.py`: empty.

`equity_research/rag/records.py`:
```python
from __future__ import annotations

import hashlib
from datetime import date

from pydantic import BaseModel


class NewsItem(BaseModel):
    ticker: str
    title: str
    text: str
    url: str
    source: str
    published_at: date


def content_hash(item: NewsItem) -> str:
    raw = f"{item.ticker}|{item.url}|{item.title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def news_metadata(item: NewsItem) -> dict:
    return {
        "doc_type": "news",
        "ticker": item.ticker,
        "date_int": int(item.published_at.strftime("%Y%m%d")),
        "content_hash": content_hash(item),
        "source": item.source,
        "url": item.url,
        "title": item.title,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_records.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/__init__.py equity_research/rag/records.py tests/test_rag_records.py
git commit -m "feat(rag): NewsItem, content_hash, metadata (date_int for range filter)"
```

---

## Task 3: News chunking

**Files:**
- Create: `equity_research/rag/chunking.py`
- Test: `tests/test_rag_chunking.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rag_chunking.py`:
```python
from datetime import date

from equity_research.rag.chunking import chunk_news
from equity_research.rag.records import NewsItem


def _item(text):
    return NewsItem(ticker="AAPL", title="Apple beats", text=text,
                    url="http://x/1", source="yahoo", published_at=date(2026, 9, 10))


def test_short_news_is_single_chunk_with_title_prefix():
    chunks = chunk_news(_item("Short body."))
    assert len(chunks) == 1
    text, meta = chunks[0]
    assert text.startswith("Apple beats")  # title prefixed for embedding context
    assert "Short body." in text
    assert meta["ticker"] == "AAPL"


def test_long_news_splits_into_multiple_chunks():
    long_text = " ".join(f"sentence{i}." for i in range(400))  # well over ~1000 tokens
    chunks = chunk_news(_item(long_text))
    assert len(chunks) > 1
    # every chunk carries the same metadata content_hash
    assert len({m["content_hash"] for _, m in chunks}) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_chunking.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/chunking.py`:
```python
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from equity_research.rag.records import NewsItem, news_metadata

# ~1000 tokens ≈ ~4000 chars; below this a news item stays a single chunk.
_MAX_SINGLE_CHARS = 4000
_splitter = RecursiveCharacterTextSplitter(chunk_size=3200, chunk_overlap=480)


def chunk_news(item: NewsItem) -> list[tuple[str, dict]]:
    meta = news_metadata(item)
    body = f"{item.title}\n{item.text}"
    if len(body) <= _MAX_SINGLE_CHARS:
        return [(body, meta)]
    return [(chunk, meta) for chunk in _splitter.split_text(body)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_chunking.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/chunking.py tests/test_rag_chunking.py
git commit -m "feat(rag): per-type news chunking (single short, recursive long)"
```

---

## Task 4: NewsStore (dedup + metadata filter + MMR) over a protocol

**Files:**
- Create: `equity_research/rag/store.py`
- Test: `tests/test_rag_store.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rag_store.py`:
```python
from datetime import date

from equity_research.rag.store import NewsStore


class FakeVectorStore:
    def __init__(self):
        self.docs = {}  # id -> (text, meta)
        self.last_query = None

    def existing_ids(self, ids):
        return {i for i in ids if i in self.docs}

    def add(self, ids, texts, metadatas):
        for i, t, m in zip(ids, texts, metadatas):
            self.docs[i] = (t, m)

    def mmr_search(self, query, where, k):
        self.last_query = (query, where, k)
        return [(t, m) for (t, m) in self.docs.values()][:k]


def _chunk(hash_, ticker="AAPL", date_int=20260910):
    return ("Apple beat estimates.", {"doc_type": "news", "ticker": ticker,
            "date_int": date_int, "content_hash": hash_})


def test_upsert_dedups_by_content_hash():
    fake = FakeVectorStore()
    store = NewsStore(fake)
    store.upsert([_chunk("h1"), _chunk("h1"), _chunk("h2")])  # h1 twice
    assert len(fake.docs) == 2  # only h1 and h2


def test_search_builds_ticker_and_date_filter():
    fake = FakeVectorStore()
    store = NewsStore(fake)
    store.upsert([_chunk("h1")])
    store.search("news about apple", ticker="AAPL", as_of=date(2026, 9, 15), k=6)
    query, where, k = fake.last_query
    assert where == {"$and": [{"ticker": {"$eq": "AAPL"}}, {"date_int": {"$lte": 20260915}}]}
    assert k == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_store.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/store.py`:
```python
from __future__ import annotations

from datetime import date
from typing import Protocol


class VectorStore(Protocol):
    def existing_ids(self, ids: list[str]) -> set[str]: ...
    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None: ...
    def mmr_search(self, query: str, where: dict, k: int) -> list[tuple[str, dict]]: ...


class NewsStore:
    def __init__(self, store: VectorStore):
        self.store = store

    def upsert(self, chunks: list[tuple[str, dict]]) -> None:
        ids, texts, metas = [], [], []
        seen: set[str] = set()
        # id = content_hash + chunk index within this batch, so multi-chunk docs stay distinct
        counts: dict[str, int] = {}
        for text, meta in chunks:
            h = meta["content_hash"]
            idx = counts.get(h, 0)
            counts[h] = idx + 1
            cid = f"{h}:{idx}"
            if cid in seen:
                continue
            seen.add(cid)
            ids.append(cid)
            texts.append(text)
            metas.append(meta)
        fresh = [i for i in ids if i not in self.store.existing_ids(ids)]
        if not fresh:
            return
        keep = [(i, t, m) for i, t, m in zip(ids, texts, metas) if i in set(fresh)]
        self.store.add([i for i, _, _ in keep], [t for _, t, _ in keep], [m for _, _, m in keep])

    def search(self, query: str, ticker: str, as_of: date, k: int) -> list[tuple[str, dict]]:
        where = {"$and": [
            {"ticker": {"$eq": ticker}},
            {"date_int": {"$lte": int(as_of.strftime("%Y%m%d"))}},
        ]}
        return self.store.mmr_search(query, where, k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_store.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/store.py tests/test_rag_store.py
git commit -m "feat(rag): NewsStore dedup + code-built ticker/date filter + MMR"
```

---

## Task 5: Reranker (optional, injectable scorer)

**Files:**
- Create: `equity_research/rag/reranker.py`
- Test: `tests/test_rag_reranker.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rag_reranker.py`:
```python
from equity_research.rag.reranker import rerank


def test_rerank_orders_by_scorer_desc():
    items = [("a", {}), ("b", {}), ("c", {})]
    # scorer gives b highest, then c, then a
    scores = {"a": 0.1, "b": 0.9, "c": 0.5}

    def scorer(query, texts):
        return [scores[t] for t in texts]

    out = rerank("q", items, scorer)
    assert [t for t, _ in out] == ["b", "c", "a"]


def test_rerank_none_scorer_is_identity():
    items = [("a", {}), ("b", {})]
    assert rerank("q", items, None) == items
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_reranker.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/reranker.py`:
```python
from __future__ import annotations

from typing import Callable

Scorer = Callable[[str, list[str]], list[float]]


def rerank(query: str, items: list[tuple[str, dict]], scorer: Scorer | None) -> list[tuple[str, dict]]:
    if scorer is None or not items:
        return items
    texts = [t for t, _ in items]
    scores = scorer(query, texts)
    order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    return [items[i] for i in order]


def default_scorer(model_name: str) -> Scorer | None:
    """Cross-encoder scorer if sentence-transformers is installed, else None (fallback to MMR order).

    Verify model download/behavior in the integration task; requires the `rerank` extra.
    """
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    encoder = CrossEncoder(model_name)

    def score(query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in encoder.predict([(query, t) for t in texts])]

    return score
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_reranker.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/reranker.py tests/test_rag_reranker.py
git commit -m "feat(rag): rerank with injectable scorer + optional cross-encoder"
```

---

## Task 6: NewsRetriever (search → rerank → texts)

**Files:**
- Create: `equity_research/rag/retrieve.py`
- Test: `tests/test_rag_retrieve.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rag_retrieve.py`:
```python
from datetime import date

from equity_research.rag.retrieve import NewsRetriever


class FakeNewsStore:
    def __init__(self, results):
        self._results = results
        self.searched = None

    def search(self, query, ticker, as_of, k):
        self.searched = (query, ticker, as_of, k)
        return self._results


def test_retrieve_reranks_and_trims_to_top_k():
    candidates = [("a", {}), ("b", {}), ("c", {})]
    store = FakeNewsStore(candidates)
    scores = {"a": 0.1, "b": 0.9, "c": 0.5}
    retriever = NewsRetriever(store, scorer=lambda q, ts: [scores[t] for t in ts])
    texts = retriever.retrieve("AAPL", "apple news", as_of=date(2026, 9, 15), k=2, candidate_k=3)
    assert texts == ["b", "c"]  # reranked desc, trimmed to k=2
    assert store.searched[3] == 3  # searched with candidate_k


def test_retrieve_without_scorer_keeps_store_order():
    candidates = [("x", {}), ("y", {})]
    retriever = NewsRetriever(FakeNewsStore(candidates), scorer=None)
    texts = retriever.retrieve("AAPL", "q", as_of=date(2026, 9, 15), k=5, candidate_k=10)
    assert texts == ["x", "y"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_retrieve.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/retrieve.py`:
```python
from __future__ import annotations

from datetime import date

from equity_research.rag.reranker import Scorer, rerank


class NewsRetriever:
    def __init__(self, store, scorer: Scorer | None):
        self.store = store
        self.scorer = scorer

    def retrieve(self, ticker: str, query: str, as_of: date, k: int, candidate_k: int) -> list[str]:
        candidates = self.store.search(query, ticker, as_of, candidate_k)
        reranked = rerank(query, candidates, self.scorer)
        return [text for text, _ in reranked[:k]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_retrieve.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/retrieve.py tests/test_rag_retrieve.py
git commit -m "feat(rag): NewsRetriever (MMR candidates -> rerank -> top-k texts)"
```

---

## Task 7: Ingest pipeline

**Files:**
- Create: `equity_research/rag/ingest.py`
- Test: `tests/test_rag_ingest.py`

- [ ] **Step 1: Write the failing test**

`tests/test_rag_ingest.py`:
```python
from datetime import date

from equity_research.rag.ingest import ingest_news
from equity_research.rag.records import NewsItem


class FakeNewsStore:
    def __init__(self):
        self.upserted = []

    def upsert(self, chunks):
        self.upserted.extend(chunks)


def _item(url):
    return NewsItem(ticker="AAPL", title="t", text="Body text here.",
                    url=url, source="yahoo", published_at=date(2026, 9, 10))


def test_ingest_chunks_and_upserts():
    store = FakeNewsStore()
    n = ingest_news(lambda t: [_item("u1"), _item("u2")], store, "AAPL")
    assert n == 2  # two items ingested
    assert len(store.upserted) == 2  # each short item -> one chunk


def test_ingest_empty_is_noop():
    store = FakeNewsStore()
    assert ingest_news(lambda t: [], store, "AAPL") == 0
    assert store.upserted == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag_ingest.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/ingest.py`:
```python
from __future__ import annotations

from typing import Callable

from equity_research.rag.chunking import chunk_news
from equity_research.rag.records import NewsItem

NewsFetcher = Callable[[str], list[NewsItem]]


def ingest_news(fetch: NewsFetcher, news_store, ticker: str) -> int:
    items = fetch(ticker)
    chunks = []
    for item in items:
        chunks.extend(chunk_news(item))
    if chunks:
        news_store.upsert(chunks)
    return len(items)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rag_ingest.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/ingest.py tests/test_rag_ingest.py
git commit -m "feat(rag): ingest_news (fetch -> chunk -> upsert)"
```

---

## Task 8: Grounding-check guardrail

**Files:**
- Create: `equity_research/agents/grounding.py`
- Modify: `equity_research/agents/judge.py`
- Test: `tests/test_grounding.py`

- [ ] **Step 1: Write the failing test**

`tests/test_grounding.py`:
```python
from datetime import date

from equity_research.agents.base import AgentOpinion
from equity_research.agents.grounding import ground
from equity_research.data.models import Evidence


def _ev(metrics=None, context=None):
    return Evidence(ticker="AAPL", as_of=date(2026, 9, 15),
                    metrics=metrics or {}, context=context or [])


def test_drops_fact_with_unseen_number():
    op = AgentOpinion(agent="fundamentals", stance="bullish", score=0.5, confidence=0.8,
                      rationale="r", key_facts=["P/E is 44.6", "Made-up revenue of 999999"])
    ev = _ev(metrics={"pe": 44.6})
    grounded = ground(op, ev)
    assert "P/E is 44.6" in grounded.key_facts       # 44.6 appears in metrics
    assert "Made-up revenue of 999999" not in grounded.key_facts  # 999999 unsupported
    assert any("dropped" in n.lower() for n in grounded.dropped_facts)


def test_grounds_fact_by_context_substring():
    op = AgentOpinion(agent="sentiment", stance="bearish", score=-0.3, confidence=0.6,
                      rationale="r", key_facts=["Analysts cite supply constraints"])
    ev = _ev(context=["Reuters: Apple faces supply constraints in Q4."])
    grounded = ground(op, ev)
    assert "Analysts cite supply constraints" in grounded.key_facts


def test_fact_without_numbers_or_evidence_is_dropped():
    op = AgentOpinion(agent="technical", stance="bullish", score=0.4, confidence=0.7,
                      rationale="r", key_facts=["The stock will definitely moon"])
    grounded = ground(op, _ev(metrics={"rsi14": 55.0}))
    assert grounded.key_facts == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grounding.py -v`
Expected: FAIL — import error; `AgentOpinion` has no `dropped_facts`.

- [ ] **Step 3: Write minimal implementation**

First add a field to `AgentOpinion` in `equity_research/agents/base.py` (after `key_facts`):
```python
    dropped_facts: list[str] = Field(default_factory=list)
```

`equity_research/agents/grounding.py`:
```python
from __future__ import annotations

import re

from equity_research.agents.base import AgentOpinion
from equity_research.data.models import Evidence

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _norm(tok: str) -> str:
    try:
        f = float(tok)
    except ValueError:
        return tok
    return ("%f" % f).rstrip("0").rstrip(".")  # 44.600000 -> 44.6 ; 100.000000 -> 100


def _numbers(s: str) -> set[str]:
    return {_norm(n) for n in _NUM.findall(s)}


def ground(opinion: AgentOpinion, evidence: Evidence) -> AgentOpinion:
    metric_numbers: set[str] = set()
    for v in evidence.metrics.values():
        metric_numbers |= _numbers(str(v))
    context_blob = "\n".join(evidence.context).lower()

    kept: list[str] = []
    dropped: list[str] = []
    for fact in opinion.key_facts:
        fact_numbers = _numbers(fact)
        number_supported = bool(fact_numbers & metric_numbers)
        # a lenient text overlap: any 5+ char word from the fact present in context
        words = [w for w in re.findall(r"[a-zA-Z]{5,}", fact.lower())]
        text_supported = any(w in context_blob for w in words) if context_blob else False
        if number_supported or text_supported:
            kept.append(fact)
        else:
            dropped.append(f"dropped unsupported: {fact}")
    return opinion.model_copy(update={"key_facts": kept, "dropped_facts": dropped})
```

Then wire it into `equity_research/agents/judge.py` — after building the opinion, ground it:
```python
from equity_research.agents.grounding import ground
```
Change the success branch of `judge_evidence`:
```python
        raw = client.generate_json(prompt, OPINION_SCHEMA)
        return ground(AgentOpinion(agent=agent, **raw), evidence)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grounding.py tests/test_judge.py -v`
Expected: PASS (grounding tests + existing judge tests — the existing judge tests use empty metrics/context so their facts are `[]` and unaffected).

- [ ] **Step 5: Commit**

```bash
git add equity_research/agents/grounding.py equity_research/agents/base.py equity_research/agents/judge.py tests/test_grounding.py
git commit -m "feat(guardrail): grounding-check drops unsupported key_facts"
```

---

## Task 9: Sentiment agent

**Files:**
- Create: `equity_research/agents/sentiment.py`
- Modify: `equity_research/agents/prompts.py` (add `sentiment` role)
- Test: `tests/test_sentiment_agent.py`

- [ ] **Step 1: Write the failing test**

`tests/test_sentiment_agent.py`:
```python
from datetime import date

from equity_research.agents.sentiment import SentimentAgent
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient


class FakeRetriever:
    def __init__(self, texts):
        self._texts = texts

    def retrieve(self, ticker, query, as_of, k, candidate_k):
        return self._texts


def _client(tmp_path, content):
    return OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                        chat_fn=lambda **k: {"message": {"content": content}})


def test_gather_puts_news_into_context(tmp_path):
    retr = FakeRetriever(["Apple beats estimates.", "Supply chain concerns."])
    ingested = {}
    agent = SentimentAgent(retriever=retr, ingest_fn=lambda t: ingested.setdefault(t, True),
                           client=_client(tmp_path, "x"))
    ev = agent.gather("AAPL", as_of=date(2026, 9, 15))
    assert "Apple beats estimates." in ev.context
    assert ingested["AAPL"] is True  # fresh news ingested before retrieval


def test_judge_returns_sentiment_opinion(tmp_path):
    retr = FakeRetriever(["Apple beats estimates."])
    content = '{"stance":"bullish","score":0.6,"confidence":0.7,"rationale":"good news","key_facts":["Apple beats estimates"]}'
    agent = SentimentAgent(retriever=retr, ingest_fn=lambda t: None, client=_client(tmp_path, content))
    op = agent.judge(agent.gather("AAPL", as_of=date(2026, 9, 15)))
    assert op.agent == "sentiment"
    assert op.stance == "bullish"
    assert "Apple beats estimates" in op.key_facts  # grounded against context


def test_no_news_degrades(tmp_path):
    agent = SentimentAgent(retriever=FakeRetriever([]), ingest_fn=lambda t: None,
                           client=_client(tmp_path, "not json"))
    op = agent.judge(agent.gather("AAPL", as_of=date(2026, 9, 15)))
    assert op.stance == "neutral"
    assert op.confidence == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sentiment_agent.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

Add a `sentiment` role to `_ROLES` in `equity_research/agents/prompts.py`:
```python
    "sentiment": "a market-sentiment analyst judging the tone of recent news",
```

`equity_research/agents/sentiment.py`:
```python
from __future__ import annotations

from datetime import date
from typing import Callable

from equity_research.agents.base import AgentOpinion
from equity_research.agents.judge import judge_evidence
from equity_research.data.models import Evidence
from equity_research.llm.ollama_client import OllamaClient

_QUERY = "recent news sentiment, outlook, risks and catalysts"


class SentimentAgent:
    name = "sentiment"

    def __init__(self, retriever, ingest_fn: Callable[[str], object], client: OllamaClient,
                 k: int = 6, candidate_k: int = 20):
        self.retriever = retriever
        self.ingest_fn = ingest_fn
        self.client = client
        self.k = k
        self.candidate_k = candidate_k

    def gather(self, ticker: str, as_of: date) -> Evidence:
        self.ingest_fn(ticker)  # pull fresh news into the store first (live)
        texts = self.retriever.retrieve(ticker, _QUERY, as_of, self.k, self.candidate_k)
        notes = [] if texts else ["no news retrieved"]
        return Evidence(ticker=ticker, as_of=as_of, metrics={}, context=texts, notes=notes)

    def judge(self, evidence: Evidence) -> AgentOpinion:
        return judge_evidence(self.name, evidence, self.client)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sentiment_agent.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/agents/sentiment.py equity_research/agents/prompts.py tests/test_sentiment_agent.py
git commit -m "feat(agents): SentimentAgent (ingest + retrieve news -> judge)"
```

---

## Task 10: Chroma store + embeddings + news adapter (thin, integration)

**Files:**
- Create: `equity_research/rag/chroma_store.py`
- Create: `equity_research/rag/adapters.py`
- Test: `tests/integration/test_rag_live.py`

These are thin I/O wrappers. Field/API details depend on installed library versions — the unit-tested logic above does not touch them. Verify live in Task 11.

- [ ] **Step 1: Write the (marked) integration test**

`tests/integration/test_rag_live.py`:
```python
from datetime import date

import pytest


@pytest.mark.integration
def test_chroma_roundtrip_and_news_fetch(tmp_path):
    """Requires Ollama (nomic-embed-text pulled) + network. Run: uv run pytest -m integration."""
    from equity_research.rag.adapters import fetch_news
    from equity_research.rag.chroma_store import ChromaVectorStore
    from equity_research.rag.ingest import ingest_news
    from equity_research.rag.store import NewsStore

    vs = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"),
                           embed_model="nomic-embed-text", collection="news_test")
    store = NewsStore(vs)
    n = ingest_news(fetch_news, store, "AAPL")
    assert n >= 0  # network-dependent; ingest must not raise
    hits = store.search("apple earnings and outlook", ticker="AAPL", as_of=date.today(), k=3)
    assert isinstance(hits, list)
```

- [ ] **Step 2: Run to confirm it is collected but deselected by default**

Run: `uv run pytest -m "not integration" -q`
Expected: this test is NOT run (deselected). Do not run `-m integration` yet (wrappers not written).

- [ ] **Step 3: Write minimal implementation**

`equity_research/rag/chroma_store.py`:
```python
from __future__ import annotations


class ChromaVectorStore:
    """Thin Chroma wrapper implementing the VectorStore protocol.

    API specifics (langchain-chroma / chromadb versions) are verified live in
    Task 11; adjust only this file if they differ.
    """

    def __init__(self, persist_dir: str, embed_model: str, collection: str = "news"):
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        self._db = Chroma(
            collection_name=collection,
            persist_directory=persist_dir,
            embedding_function=OllamaEmbeddings(model=embed_model),
        )

    def existing_ids(self, ids: list[str]) -> set[str]:
        got = self._db.get(ids=ids)
        return set(got.get("ids", []))

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        self._db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def mmr_search(self, query: str, where: dict, k: int) -> list[tuple[str, dict]]:
        docs = self._db.max_marginal_relevance_search(query, k=k, filter=where)
        return [(d.page_content, d.metadata) for d in docs]
```

`equity_research/rag/adapters.py`:
```python
from __future__ import annotations

from datetime import date, datetime

from equity_research.rag.records import NewsItem


def fetch_news(ticker: str) -> list[NewsItem]:
    """Fetch recent news for a ticker via yfinance. Thin; verified live in Task 11."""
    import yfinance as yf

    raw = getattr(yf.Ticker(ticker), "news", []) or []
    items: list[NewsItem] = []
    for entry in raw:
        content = entry.get("content", entry)  # yfinance schema varies by version
        title = content.get("title") or ""
        url = (content.get("canonicalUrl", {}) or {}).get("url") or content.get("link") or ""
        summary = content.get("summary") or content.get("description") or title
        pub = content.get("pubDate") or content.get("providerPublishTime")
        try:
            published = (datetime.fromisoformat(pub.replace("Z", "+00:00")).date()
                         if isinstance(pub, str) else
                         datetime.utcfromtimestamp(int(pub)).date() if pub else date.today())
        except Exception:
            published = date.today()
        if not title or not url:
            continue
        items.append(NewsItem(ticker=ticker, title=title, text=summary, url=url,
                              source="yfinance", published_at=published))
    return items
```

- [ ] **Step 4: Verify unit suite still green (integration deselected)**

Run: `uv run pytest -m "not integration" -p no:warnings -q`
Expected: PASS; the live RAG test is deselected. `uv run python -c "import equity_research.rag.chroma_store, equity_research.rag.adapters"` must succeed offline (imports are lazy inside functions/__init__).

- [ ] **Step 5: Commit**

```bash
git add equity_research/rag/chroma_store.py equity_research/rag/adapters.py tests/integration/test_rag_live.py
git commit -m "feat(rag): thin Chroma store + yfinance news adapter (integration)"
```

---

## Task 11: Wire into CLI (ingest command + sentiment in analyze) + live verify

**Files:**
- Modify: `equity_research/cli.py`
- Test: `tests/test_cli.py` (add assertions)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:
```python
def test_analyze_ticker_includes_sentiment_agent(monkeypatch):
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
    names = [a.name for a in captured["agents"]]
    assert "sentiment" in names
    assert {"fundamentals", "technical", "risk"} <= set(names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_analyze_ticker_includes_sentiment_agent -v`
Expected: FAIL — no sentiment agent.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/cli.py`, add imports:
```python
from equity_research.agents.sentiment import SentimentAgent
from equity_research.rag.adapters import fetch_news
from equity_research.rag.chroma_store import ChromaVectorStore
from equity_research.rag.ingest import ingest_news
from equity_research.rag.reranker import default_scorer
from equity_research.rag.retrieve import NewsRetriever
from equity_research.rag.store import NewsStore
```

In `analyze_ticker`, after building `edgar`, construct the RAG pieces and add the sentiment agent:
```python
    news_store = NewsStore(ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"],
                                             embed_model=cfg.rag["embed_model"]))
    retriever = NewsRetriever(news_store, scorer=default_scorer(cfg.rag["rerank_model"]))
    sentiment = SentimentAgent(
        retriever=retriever,
        ingest_fn=lambda t: ingest_news(fetch_news, news_store, t),
        client=client, k=cfg.rag["retrieve_k"], candidate_k=cfg.rag["candidate_k"],
    )
```
And add `sentiment` to the `agents` list (after `TechnicalAgent`, before or after `RiskAgent`):
```python
    agents = [
        FundamentalsAgent(facts_source=edgar, prices=prices, client=client),
        TechnicalAgent(prices=prices, client=client),
        sentiment,
        RiskAgent(prices=prices, client=client, benchmark=cfg.benchmark, risk_cfg=cfg.risk),
    ]
```

Add a standalone `ingest` command:
```python
@app.command()
def ingest(ticker: str, config: str = "config.yaml"):
    cfg = Config.load(config)
    store = NewsStore(ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"], embed_model=cfg.rag["embed_model"]))
    n = ingest_news(fetch_news, store, ticker.upper())
    typer.echo(f"Ingested {n} news items for {ticker.upper()}")
```
Note: adding a second command means typer now requires the subcommand name — `analyze AAPL` and `ingest AAPL` both work; update the integration smoke invocation accordingly.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS. Note: the existing `test_analyze_prints_markdown` invokes `app` with `["AAPL"]`; with two commands this must become `["analyze", "AAPL"]` — update that one invocation. Fix it now and keep the assertion.

- [ ] **Step 5: Commit**

```bash
git add equity_research/cli.py tests/test_cli.py
git commit -m "feat(cli): ingest command + wire SentimentAgent into analyze"
```

---

## Task 12: Full suite + live RAG verification

**Files:** none (verification), unless a wrapper needs fixing.

- [ ] **Step 1: Full unit suite**

Run: `uv run pytest -m "not integration" -p no:warnings -q`
Expected: all PASS (Phase 0/1/2A + all 2B unit tests).

- [ ] **Step 2: Pull the embedding model**

Run: `ollama pull nomic-embed-text`
Expected: model downloaded (~275 MB).

- [ ] **Step 3: Live RAG + full analyze**

Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run pytest -m integration -q`
Then: `DISABLE_PANDERA_IMPORT_WARNING=True uv run python -m equity_research.cli ingest AAPL`
Then: `DISABLE_PANDERA_IMPORT_WARNING=True uv run python -m equity_research.cli analyze AAPL`
Expected: ingest reports a news count; analyze shows a `sentiment` opinion in the trace (or `sentiment` in skipped if no news / embeddings unavailable). If Chroma/langchain/yfinance-news APIs differ from the wrappers, fix ONLY `chroma_store.py` / `adapters.py`, then re-run — the unit-tested logic must not change.

- [ ] **Step 4: (Optional) enable reranker**

Run: `uv sync --extra rerank` then re-run analyze; confirm it still works (reranker now active). If disk is tight, skip — the MMR fallback is the default.

- [ ] **Step 5: Commit any wrapper fixes**

```bash
git add -A && git commit -m "fix(rag): live-integration adjustments"
```
(Skip if nothing changed.)

---

## Self-Review (completed during authoring)

- **Spec coverage (2B):** `rag/` package — store/records/chunking/retrievers/ingest (Tasks 2-7, 10); Chroma + nomic-embed (Tasks 10, config Task 1); MMR + code metadata filter with `date ≤ as_of` (Task 4); cross-encoder rerank, made optional with graceful MMR fallback (Task 5, deviation noted below); `ingest` CLI + analyze auto-ingest (Tasks 9, 11); SentimentAgent live-only (Task 9); grounding-check for all agents (Task 8); tests incl. retrieval logic and live verify (throughout, Task 12). Deferred (documented): filing-text (2.5), backtest (3).
- **Deviation from spec (intentional, flagged):** the cross-encoder reranker is an OPTIONAL dependency (`rerank` extra) with a graceful MMR fallback, rather than a hard v1 requirement — because `sentence-transformers` pulls `torch` (multi-GB) onto a disk-constrained host. The retrieval interface is identical; enabling it is one `uv sync --extra rerank`. Retrieval-relevance golden (hit@k) from the spec is NOT built in this plan (needs a curated live corpus); it is deferred to a follow-up once real news is flowing — noted so it isn't silently dropped.
- **Placeholder scan:** none — every code step is complete. Thin wrappers (`chroma_store.py`, `adapters.py`) carry real code plus an explicit "verify live, adjust only this file" note, matching the Phase 1 EdgarProvider pattern.
- **Type consistency:** `NewsItem`/`content_hash`/`news_metadata`, `chunk_news -> list[tuple[str,dict]]`, `VectorStore`(`existing_ids`/`add`/`mmr_search`), `NewsStore`(`upsert`/`search`), `rerank(query, items, scorer)` + `Scorer`, `NewsRetriever.retrieve(ticker, query, as_of, k, candidate_k)`, `ingest_news(fetch, news_store, ticker)`, `ground(opinion, evidence)` + `AgentOpinion.dropped_facts`, `SentimentAgent(retriever, ingest_fn, client, k, candidate_k)` are consistent across tasks and align with existing signatures (`Evidence`, `AgentOpinion`, `judge_evidence`, `OllamaClient`).
