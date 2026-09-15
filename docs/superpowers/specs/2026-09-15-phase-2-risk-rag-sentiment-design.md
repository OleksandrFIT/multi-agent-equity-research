# Multi-Agent Equity Research — Phase 2 design (Risk + RAG + Sentiment)

Дата: 2026-09-15
Статус: затверджено (brainstorming), готово до планування (writing-plans)
Попередній етап: Phase 0+1 змерджено в `master` (працює наживо на `qwen2.5:7b`).
Базовий дизайн: `docs/superpowers/specs/2026-09-15-multi-agent-equity-research-design.md`.

## 1. Мета і межі

Додати два агенти й RAG-шар до вже робочого ядра:
- **Risk-агент** — детерміновані ризик-метрики з цін; діє як **гейт на confidence**, не як directional-голос.
- **RAG-інфраструктура** — Chroma + локальні ембединги + ingest + ретрівер з реранкером.
- **Sentiment-агент** — LLM-оцінка тональності свіжих новин поверх RAG. **Live-only.**
- **Guardrail: grounding-check** — відкладений з Фази 1, вмикається тут.

**Декомпозиція (два плани, кожен дає робочий результат):**
- **План 2A — Risk-агент** (лише ціни, без RAG). Спочатку.
- **План 2B — RAG-інфра + Sentiment-агент** (Chroma, ембединги, ingest, ретрівер+реранкер, потім Sentiment).

**Поза межами (пізніше):** filing-текст у RAG (Фаза 2.5), backtest-harness (Фаза 3), web UI, verdict_memory.

**Незмінні принципи (з базового дизайну):** код-оркестрація; числа детерміновані, LLM лише інтерпретує; строгий JSON-вивід; RAG доповнює якісний контекст; коректність > латенсі.

## 2. План 2A — Risk-агент

### 2A.1 Метрики (детерміновано з цін)
`RiskAgent.gather(ticker, as_of)` тягне ціни через наявний `PriceProvider` (і для тікера, і для ринкового бенчмарку SPY) і рахує в чистому коді:
- **annualized volatility** = σ(денних лог-ретернів) × √252;
- **max drawdown** = найбільше падіння від піку на серії;
- **beta vs SPY** = cov(ret_ticker, ret_spy) / var(ret_spy) на спільному вікні дат.

Усе — чисті функції в `equity_research/analytics/risk.py`, повністю юніт-тестовані на заморожених рядах. SPY тягнеться тим самим `PriceProvider.history` (як вторинне джерело — недоступність не фатальна: beta → NaN, гейт це толерує).

### 2A.2 Роль в агрегації — confidence-гейт (НЕ directional)
Ризик не голосує «buy/sell». Пропонований контракт:
- **Deterministic gate value.** Чиста функція `risk_level(metrics, config) -> float ∈ [0,1]` нормалізує vol/beta/drawdown відносно конфігурованих референс-смуг (напр. vol 15%→low, 60%→high). Число детерміноване, не від LLM.
- **`RiskAgent.judge`** повертає звичайний `AgentOpinion(agent="risk", stance="neutral", score=risk_level, confidence=risk_level, rationale=<LLM-наратив>, key_facts=[метрики])`. Для risk-агента поле `score` = **risk level ∈ [0,1]** (задокументоване перевизначення семантики; НЕ directional). LLM викликається лише для короткого наративу (`generate_text`), числа не чіпає.

### 2A.3 Зміни в агрегаторі
- `DIRECTIONAL_AGENTS = {"fundamentals", "technical", "sentiment"}`.
- Directional `score`/`base_confidence` рахуються тільки з directional-опіній (ваги перенормовуються по наявних directional-агентах через `Config.normalized_weights`). Risk у directional-сумі **не бере участі**.
- Якщо є risk-опінія: `rl = risk_opinion.score`; `final_confidence = base_confidence × (1 − risk_gate_strength × rl)` (клемп у [0,1]); якщо `rl ≥ risk_caution_threshold` — до `Verdict` додається caution-note.
- Якщо risk-агента нема (напр. backtest без нього) — гейт не застосовується, поведінка як зараз.
- `Verdict` отримує нове поле `caution: str | None` (риск-застереження), яке `reporting` показує окремим рядком.

### 2A.4 Конфіг (`config.yaml`)
- `weights.risk` **лишаємо в конфізі, але агрегатор його ігнорує** для напряму (directional беруться лише fund/tech/sentiment). Ключ не видаляємо, щоб не ламати наявні тести Config/normalized_weights.
- Новий блок:
  ```yaml
  risk:
    gate_strength: 0.5        # наскільки сильно ризик гасить confidence
    caution_threshold: 0.6    # rl >= цього → caution-note
    vol_low: 0.15             # референс-смуги для нормалізації risk_level
    vol_high: 0.60
    beta_high: 1.5
    drawdown_high: 0.40
  ```
- `benchmark: SPY` (бенчмарк для beta).

### 2A.5 Тестування 2A
- Golden-тести метрик: vol/drawdown/beta на заморожених рядах з відомими значеннями.
- `risk_level` — межові кейси (low/high vol, high beta, велика просадка).
- Агрегатор: гейт знижує confidence; risk не змінює directional score; caution-note при високому rl; відсутність risk → без гейту.
- `RiskAgent.judge` на мокнутому клієнті (наратив не ламає числа).

## 3. План 2B — RAG-інфра + Sentiment

### 3.1 Пакет `rag/`
- `store.py` — Chroma (persistent, шлях із конфігу) + `OllamaEmbeddings("nomic-embed-text")` (потрібен `ollama pull nomic-embed-text`, ~275 MB).
- `records.py` — тип запису `news` + метадата-модель (`doc_type`, `ticker`, `date`, `content_hash`, `source`, `url`, `title`).
- `chunking.py` — по-типове: `news` не різати, якщо < ~1000 токенів; інакше recursive 800/120. Title у метадаті + префіксом до тексту чанку.
- `retrievers.py` — фабрика: `news` → **MMR** (k~6) з **кодовим** метадата-фільтром (`ticker`, `date ≤ as_of`); поверх — `CrossEncoderReranker` (`sentence-transformers`, напр. `BAAI/bge-reranker-base`, ~90 MB). Ніякого `SelfQueryRetriever`.
- `ingest.py` — `fetch (RSS/yfinance) → chunk → embed → upsert`, дедуп по `content_hash`.

### 3.2 CLI
- Нова команда `ingest TICKER` (наповнити стор новинами).
- `analyze` перед sentiment-кроком сам дотягує свіжі новини (live ingest).
- Примітка: після додавання другої команди typer вимагатиме назву субкоманди — тобто `analyze TICKER` і `ingest TICKER` запрацюють природно (зникне поточний single-command-колапс).

### 3.3 Sentiment-агент
- `gather(ticker, as_of)` — ingest свіжих новин → retrieve top-k релевантних (MMR + rerank) → кладе тексти в `Evidence.context`; метрик нема (або лічильник статей).
- `judge` — LLM читає делімітований `<untrusted_content>` (механізм уже є) і повертає `AgentOpinion(agent="sentiment", …)` зі stance/score/confidence/rationale/key_facts (key_facts = конкретні заголовки/факти з новин).
- **Live-only:** у backtest пропускається (нема історичних безкоштовних новин) — оркестратор і directional-нормалізація вже це толерують.

### 3.4 Guardrail: grounding-check
- Чиста функція `ground(opinion, evidence) -> opinion`, що фільтрує `key_facts`: лишає лише ті, що підтверджені в evidence — тобто мають спільне число з `metrics` **або** значущий підрядок із `context`. Непідтверджені відкидаються, і додається note про відкинуті факти (для прозорості).
- Застосовується в `judge_evidence` для **всіх** агентів після парсингу опінії (тож і fundamentals/technical теж перевіряються проти своїх метрик).
- Навмисно лояльний (відкидає лише явно невідповідне), щоб не гризти валідні перефразування.

### 3.5 Тестування 2B
- `chunking`: короткі новини не ріжуться; довгі — ріжуться з overlap; title у метадаті.
- `retrievers`: метадата-фільтр по ticker/date застосовується (на мокнутому/in-memory сторі); реранкер міняє порядок (на детермінованому фейку скорера).
- `ingest`: дедуп по content_hash (той самий запис двічі → один документ).
- retrieval-relevance golden: маленький набір `(ticker+питання → очікуваний документ)`, метрика hit@k.
- `SentimentAgent.judge`: парсинг/деградація на мокнутому клієнті.
- `ground`: вигаданий факт відкинуто; підтверджений — лишається.

## 4. Файлова структура (доповнення)
```
equity_research/
  analytics/risk.py            # 2A: volatility, max_drawdown, beta, risk_level
  agents/risk.py               # 2A: RiskAgent
  agents/sentiment.py          # 2B: SentimentAgent
  agents/grounding.py          # 2B: ground(opinion, evidence)
  rag/
    store.py records.py chunking.py retrievers.py ingest.py   # 2B
  orchestration/aggregator.py  # 2A: directional set + risk gate + caution
  reporting/report.py          # 2A: show caution line
  cli.py                       # 2B: add `ingest`; analyze auto-ingest
config.yaml                    # 2A: risk block + benchmark
tests/                         # відповідні тести обох планів
```

## 5. Обробка помилок і краєві випадки
- SPY недоступний → beta=NaN, `risk_level` рахується без beta (нормалізація по наявних складниках).
- Новин нема / стор порожній → sentiment повертає `neutral, confidence=0` з поміткою (наявний degrade-механізм).
- Ембединг/реранкер-модель не завантажена → ingest/retrieve кидає зрозумілу помилку з підказкою `ollama pull nomic-embed-text` / встановити sentence-transformers; sentiment-агент при цьому просто skipped (directional нормалізація толерує).
- Chroma-стор на диску (враховуємо дискові обмеження хоста).

## 6. Ризики / відкриті питання
- **RSS-релевантність:** заголовки короткі; реранкер має допомогти, але якість sentiment на 7B перевіримо на реальних прикладах.
- **grounding-check надто агресивний** міг би відкидати валідні перефразування — тримаємо лояльним, калібруємо на прикладах.
- **Дисковий бюджет:** nomic-embed (~275 MB) + reranker (~90 MB) + Chroma-стор. Наразі ~14 GB вільно — ок.
- **beta-вікно:** узгодити довжину вікна (напр. 1 рік денних ретернів) у конфігу.
