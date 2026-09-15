# Golden set — design

Дата: 2026-09-15
Статус: чернетка на затвердження

## Мета
Курований «золотий» набір еталонів для оцінки якості та стабільності системи у трьох вимірах:
1. **Fundamentals-метрики** — детерміновані, hand-verified розрахунки.
2. **Retrieval-relevance** — детермінована оцінка ретрівера (hit@k, MRR) на мічених парах.
3. **End-to-end вердикти** — регресія/стабільність суджень усієї системи.

## Що вже є (розширюємо, не дублюємо)
- `tests/fixtures/aapl_facts.json`, `tests/fixtures/edge_cases.json` — fundamentals-метрики (лише AAPL + 2 краєві).
- `tests/fixtures/hitk_corpus.json` — retrieval (лише AAPL, 3 кейси).
- `equity_research/eval/retrieval_eval.py` — `hit_at_k` (single expected id).

## Ключовий принцип «expected»
- Метрики й retrieval — **об'єктивна** істина (ручний розрахунок / очевидна релевантність).
- Вердикти — **НЕ** претензія на ринкову правоту. «expected» = baseline stance/verdict, зафіксований
  після ручного огляду; тест — це **guard стабільності** (seed+temp0 → відтворюваність) і **санності**
  (структура повна, діапазони притомні, немає фейкового вердикту), а не оракул ринку.

## Дизайн за вимірами

### 1. Fundamentals-метрики (детерміновано, unit)
- Новий `tests/fixtures/golden/metrics_golden.json`: масив записів для тікерів backtest-universe
  (AAPL, MSFT, KO, JPM, XOM) — кожен: `ticker`, `source_filing`, `filed_at`, `reference` (URL),
  `price`, `facts`, `expected` (pe, roe, debt_to_equity, revenue_growth). **Facts беремо live з EDGAR
  на фіксовану дату** під час реалізації (point-in-time), `expected` рахуємо і фіксуємо.
- Розширити `edge_cases.json` (додати off-cycle fiscal, missing filing, negative equity).
- `tests/test_golden_metrics.py` — ітерує `metrics_golden.json` (як зараз aapl). Детерміновано.

### 2. Retrieval-relevance (детермінована метрика + live-прогін)
- Розширити `tests/fixtures/hitk_corpus.json` → `retrieval_golden.json`: кілька тікерів, більше
  мічених `documents` (news + короткі filing-снипети), `cases` з `expected` як **список** id
  (кілька релевантних). Категорії запитів: earnings, legal/regulatory, supply/ops, guidance, risk.
- `equity_research/eval/retrieval_eval.py`: `hit_at_k` приймає `expected` як list (хіт, якщо будь-який
  очікуваний у топ-k); додати `mrr(retrieve_fn, cases, k)` (Mean Reciprocal Rank).
- `tests/test_retrieval_eval.py` — unit на hit@k (multi-expected) і MRR (детерміновано, фейковий
  retrieve_fn).
- `tests/integration/test_retrieval_golden.py` — **реальний** ретрівер (Ollama-embed) над golden-
  корпусом: інжест → retrieve → `hit_at_k ≥ поріг` (напр. 0.8) і `mrr ≥ поріг`. Marked `integration`.

### 3. End-to-end вердикти (регресія/стабільність, live)
- Новий `tests/fixtures/golden/verdicts_golden.json`: масив `{ticker, as_of, expected_verdict,
  expected_stances: {fundamentals, technical, sentiment}}` для кількох (ticker × as_of) з
  backtest-набору. Значення **захоплюємо live** з реального прогону після ручного огляду.
- `tests/integration/test_verdicts_golden.py` (marked `integration`):
  - для кожного кейсу `analyze_ticker(ticker, as_of)` → перевірка: `verdict.status == "ok"`,
    `verdict.verdict == expected_verdict`, стенси price-агентів збігаються, усі агенти присутні
    (немає несподіваних skip), score/confidence у [-1,1]/[0,1].
  - **Відтворюваність:** запустити двічі → однакові score/verdict (seed+temp0 + DiskCache).

## Файли
- Дані: `tests/fixtures/golden/metrics_golden.json`, `retrieval_golden.json`, `verdicts_golden.json`;
  оновлений `tests/fixtures/edge_cases.json`; README у `tests/fixtures/golden/`.
- Код: `equity_research/eval/retrieval_eval.py` (multi-expected + MRR).
- Тести: `tests/test_golden_metrics.py`, `tests/test_retrieval_eval.py` (unit);
  `tests/integration/test_retrieval_golden.py`, `tests/integration/test_verdicts_golden.py` (integration).

## Тестування / прийомка
- Unit (детерміновані) — у звичайному `uv run pytest`.
- Integration (Ollama/мережа) — `uv run pytest -m integration`; не блокують звичайний прогін.
- Live-capture даних (facts, verdicts) робимо під час реалізації і фіксуємо у фікстурах з
  посиланнями/датами; README фіксує метод («не автооновлювати; нові дати — нові файли»).

## Відкриті рішення (підтвердити при рев'ю)
1. Скільки кейсів на вимір для старту? Пропозиція: метрики — 5 тікерів; retrieval — 3 тікери×~4
   запити; вердикти — 3 (ticker×as_of). Розширюємо потім.
2. Поріг для retrieval integration (hit@1 vs hit@3, MRR)? Пропозиція: hit@3 ≥ 0.8.

## Поза межами
Автооновлення фікстур; повний leaderboard; вердикт як оракул ринкової правоти.
