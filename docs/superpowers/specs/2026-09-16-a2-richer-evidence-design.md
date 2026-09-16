# A2 — Richer point-in-time evidence — design

Дата: 2026-09-16
Статус: чернетка на затвердження
Частина: A2 (напрями 2+3). Після Q1✔/Q3✔/A3✔. Мета — дати шанс на **сигнал** через кращі докази,
залишаючись **point-in-time** (щоб A3 міг це виміряти).

## Мета
1. **Багатший історичний фундаментал** з EDGAR: маржі (operating/net/FCF) + current ratio, поверх
   наявних P/E, ROE, D/E, revenue growth. Усе з 10-K на as_of → вимірюється.
2. **Стійкість `company_facts`**: жодних крашів на відсутньому полі (nan, не `TypeError`) — закриває
   KO/XOM/GOOGL.
3. **Додаткова filing-секція**: `business` (Item 1) у RAG для fundamentals — якісна глибина.
4. Пере-вимір через A3-backtest: чи зрушив IC.

## Проба (наживо, підтверджено)
- `get_financial_metrics()` дає: `operating_income, free_cash_flow, current_ratio, current_assets,
  current_liabilities, total_assets, operating_cash_flow, capital_expenditures, net_income, revenue,
  stockholders_equity, total_liabilities, shares_outstanding_diluted/basic`.
- Доступні filing-секції: `risk_factors`, `management_discussion`, **`business`** (решта — None).
- Краш KO — `float(None)` на якомусь полі; фікс = дефенсивна екстракція всіх полів.

## Backend

### `company_facts` (`data/edgar.py`) — стійка + багатша
- Ввести `_num(m, key) -> float` (повертає `nan` на None/відсутнє/нечислове) і брати ВСІ поля через
  нього (без прямих `float(...)`, що падають).
- Повертати, крім наявних (net_income, revenue, revenue_prev, equity, total_debt, eps_ttm), ще:
  `operating_income, free_cash_flow, current_ratio`.
- eps_ttm: `net_income/shares` лише коли обидва скінченні, інакше nan.

### `compute_fundamental_metrics` (`analytics/fundamentals.py`) — нові метрики
- Додати: `operating_margin = op_income/revenue`, `net_margin = net_income/revenue`,
  `fcf_margin = free_cash_flow/revenue`, `current_ratio = current_ratio` (passthrough).
- Усе через `_safe_div` (nan-safe). Наявні pe/roe/debt_to_equity/revenue_growth без змін.

### `_METRIC_SPECS` (`agents/prompts.py`) — конвенції
- Додати ярлики+підказки: operating/net/fcf margin (`% доходу; вище = прибутковіше`),
  current_ratio (`>1 = поточні активи покривають поточні зобов'язання; ліквідність`).
- Гейджі на фронті автоматично підхоплять нові метрики (MetricGauges SPECS — додати їх у клієнт).

### Filing-секції (`rag/filings.py`)
- У `fetch_filing_sections` додати `business` (Item 1) поряд із risk_factors/mda (той самий
  guard на довжину). Fundamentals-агент отримає ширший контекст.

## Frontend (мінімально)
- `MetricGauges` SPECS: додати operating_margin/net_margin/fcf_margin (0..0.5, %),
  current_ratio (0..3, x). Щоб нові метрики мали гейджі.

## Файли
- `equity_research/data/edgar.py`, `equity_research/analytics/fundamentals.py`,
  `equity_research/agents/prompts.py`, `equity_research/rag/filings.py`.
- Frontend: `src/components/MetricGauges.tsx`.
- Тести: `test_edgar_asof`/новий (дефенсивний _num), `test_golden_metrics`/`test_fundamentals`
  (нові метрики), `test_prompts` (конвенції), filings — integration.

## Тестування
- Детерміновано: `_num` (None→nan); `compute_fundamental_metrics` рахує нові метрики на фейкових
  facts (+ nan-краї); prompt містить нові конвенції; MetricGauges Vitest.
- Integration: `fetch_filing_sections` повертає business (жива, `-m integration`); company_facts на
  KO/XOM **не падає** (жива).
- **Пере-вимір:** повторити A3-backtest (10×6) — порівняти IC до/після A2. Golden re-capture.

## Чесна засторога
Це point-in-time, тож вимірне — але фундаментал історично слабо корелює з 21/63-денними рухами;
шанс на сигнал є, гарантії немає. Головна перевага гарантована: **стійкість** (нема крашів) і
багатший контекст.

## Відкриті рішення (підтвердити)
1. Набір метрик: operating_margin, net_margin, fcf_margin, current_ratio (реком.) — ок?
2. Додаємо секцію `business` (реком.) — ок?

## Поза межами
Форвардні аналітики (невимірно), транскрипти (сорсинг), Q2/A1.
