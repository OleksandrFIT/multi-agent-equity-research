# Multi-Agent Equity Research — Phase 2.5 design (filing-text RAG)

Дата: 2026-09-15
Статус: затверджено (brainstorming), готово до планування (writing-plans)
Попередні етапи: Phase 0/1 + 2A + 2B + 2.6 + 3A/3B + 4A/4B у `master`.
Базовий дизайн: `docs/superpowers/specs/2026-09-15-multi-agent-equity-research-design.md`.

## 1. Мета і межі

Додати якісний текст 10-K (**Item 1A Risk Factors** + **Item 7 MD&A**) у RAG, щоб
**Fundamentals-агент** доповнював точні числа якісним контекстом. Point-in-time коректно
(працює і в backtest, на відміну від news/sentiment).

**Рішення (brainstorming):**
- Чанкування: **ParentDocument** — child ~500 токенів для матчу, parent = ціла секція для LLM.
- Сховище: **реюз того самого Chroma-стору** з `doc_type`-фільтром (одна колекція).
- Секції: **10-K** (той самий annual filing, що для as-of метрик у 3A), Item 1A + Item 7.

**Поза межами:** 10-Q секції; earnings-call транскрипти; verdict_memory.

## 2. Компоненти

### 2.1 Екстракція (`rag/filings.py`, тонка, integration-verified)
`fetch_filing_sections(ticker, as_of) -> list[FilingSection]` — через edgartools бере найсвіжіший
10-K з `filing_date ≤ as_of` (реюз `select_filing_asof` із 3A) і витягує текст Item 1A + Item 7.
Кожна `FilingSection`: `ticker, section ("risk_factors"|"mda"), filed_at, form="10-K", text`.
Точний edgartools-API (напр. `TenK.risk_factors` / `.management_discussion`) уточнюємо наживо.

### 2.2 Parent-docstore (`rag/parent_store.py`, чистий, файловий)
`ParentStore(dir)`: `put(parent_id, text)` пише `<dir>/<parent_id>.txt`; `get(parent_id) -> str`.
Персистентний, простий, тестований на tmp_path.

### 2.3 Records + chunking (`rag/filing_records.py`, чистий)
- `filing_parent_id(section) -> str` — `sha256(ticker|form|section|filed_at)` (стабільний id секції).
- `filing_metadata(section) -> dict` — `{doc_type:"filing_section", ticker, date_int (filed_at),
  form, section, content_hash}`.
- `chunk_filing(section) -> (parent_id, parent_text, list[(child_text, child_meta)])` —
  parent_text = уся секція; children = recursive-спліт (~500 токенів), кожен child_meta =
  filing_metadata + `parent_id`. (Recursive у межах секції — межі Item не перетинаються, бо кожна
  секція обробляється окремо.)

### 2.4 FilingStore (`rag/filing_store.py`, чистий логіка на VectorStore+ParentStore)
- `upsert(sections)`: для кожної секції — `put(parent_id, parent_text)` у ParentStore, і
  `vector_store.add(child_ids, child_texts, child_metas)` у Chroma (дедуп по child id =
  `parent_id:idx`).
- `search(query, ticker, as_of, k) -> list[str]`: `vector_store.mmr_search(query, where, k*4)` з
  **where = doc_type=filing_section ∧ ticker ∧ date_int≤as_of**; збирає `parent_id` у ранговому
  порядку (дедуп), тягне повні тексти з ParentStore, повертає top-k унікальних секцій.

### 2.5 NewsStore doc_type-фільтр
Оскільки одна колекція тепер містить `news` і `filing_section`, `NewsStore.search` додає
`{"doc_type": {"$eq": "news"}}` у свій where — інакше news-ретрів забирав би filing-чанки.

### 2.6 Fundamentals-агент
`FundamentalsAgent` отримує `filing_retriever` + `filing_ingest_fn` (ін'єкція). `gather`:
1. рахує метрики (код, як зараз);
2. `filing_ingest_fn(ticker)` — інжест filing-секцій (live дотягує/оновлює);
3. `filing_retriever.retrieve(ticker, query, as_of, k)` з query «risk factors, financial health,
   business outlook» → кладе тексти в `Evidence.context`.
LLM у judge бачить і числа (metrics), і якісний текст (`<untrusted_content>` — делімітація вже є).
Backtest: filing-ретрів PIT через `filed_at ≤ as_of` — фундаменталка отримує якісний контекст і в
історії (без look-ahead).

### 2.7 CLI + config
- `ingest TICKER` додатково інжестить filing-секції (news + filings).
- `analyze` і backtest-wiring підключають `FundamentalsAgent` з filing-ретрівером.
- Config `rag`: `filing_retrieve_k` (напр. 3), `filing_candidate_k` (напр. 12), `parent_dir`
  (напр. `.parents`).

## 3. Тестування
- `parent_store`: put/get roundtrip, персистентність.
- `filing_records`/`chunk_filing`: parent_id стабільний; children мають parent_id; коротка секція →
  1 child; довга → кілька.
- `FilingStore`: upsert кладе parent у ParentStore і children у vector store; search збирає
  parent_ids і повертає унікальні parent-тексти; where містить doc_type+ticker+date.
- `NewsStore`: where тепер містить doc_type=news.
- `FundamentalsAgent`: gather кладе filing-текст у context (мокнутий ретрівер+ingest) поряд з
  метриками.
- Інтеграція: реальна екстракція 10-K секцій (Item 1A/Item 7 непорожні); analyze показує
  fundamentals із filing-контекстом; backtest не падає.

## 4. Файлова структура (доповнення)
```
equity_research/rag/
  filings.py          # екстракція 10-K секцій (thin, integration)
  parent_store.py     # файловий parent-docstore
  filing_records.py   # id/metadata/chunk_filing (чисті)
  filing_store.py     # FilingStore (ParentDocument логіка)
  filing_retrieve.py  # FilingRetriever (search + rerank -> top-k parent texts)
  store.py            # NewsStore: + doc_type=news фільтр
equity_research/agents/fundamentals.py  # + filing ingest/retrieve у gather
equity_research/cli.py                  # ingest filings; wire fundamentals filing retriever
config.yaml + config.py                 # filing rag params
```

## 5. Обробка помилок / краєві випадки
- Немає 10-K ≤ as_of, або секція порожня → filing-ingest дає 0 секцій; fundamentals працює лише на
  метриках (context порожній) — не падає.
- Ембединги/Chroma недоступні → filing-ретрів кидає; fundamentals-агент gracefully skipped
  (наявний orchestrator try/except) — метрики теж не дійдуть, але це той самий шлях, що й зараз.
- Дедуп по content_hash/child id уникає повторного індексування тієї самої секції.

## 6. Ризики / відкриті питання
- **edgartools section API** (як саме дістати Item 1A/Item 7 текст) — уточнюємо в інтеграції;
  ризик знижено тонкою обгорткою.
- **Розмір parent-секцій** (Risk Factors у 10-K буває дуже великий) — parent-текст у промпті може
  бути завеликим; за потреби обрізати parent до ~перших N символів у ретрівері (уточнити на живих
  даних).
- **Backtest повільніший** (fundamentals тепер робить ретрів+ембединги на кожну as_of-точку) —
  прийнятно за пріоритетом коректності; кеш ембедингів у межах прогону допомагає.
