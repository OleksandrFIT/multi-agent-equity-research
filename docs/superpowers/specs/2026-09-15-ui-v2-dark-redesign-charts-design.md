# Web UI v2 — dark fintech redesign + charts (design)

Дата: 2026-09-15
Статус: затверджено (brainstorming), готово до планування (writing-plans)
Стосується: FastAPI backend (`Multi-Agent Equity/api/`) + React client (`Multi-Agent Equity Client/`).

## 1. Мета і межі

Редизайн веб-клієнта в **dark fintech-термінал** + повноцінні **графіки**: ціна з таймфреймами
та SMA-оверлеями, RSI-сабчарт, метрики-гейджі, backtest-графіки. Логіку/агентів/eval не чіпаємо —
лише presentational шар + один новий backend-ендпоінт для цінових серій.

**Рішення (brainstorming):** тема = dark fintech; графіки = усі (price+SMA+timeframes, RSI,
gauges, backtest); чарт-тех = **lightweight-charts** (TradingView).

**Декомпозиція (два плани):**
- **План А — backend `/api/prices`** (розширення FastAPI). Спочатку.
- **План Б — frontend редизайн + графіки** (web client repo). Залежить від А.

**Поза межами:** нові агенти; свічки з OHLC (yfinance-adapter дає лише Close → лінія/area); realtime
оновлення цін; авторизація.

## 2. План А — backend `/api/prices`

### 2.1 Чиста серія (`equity_research/analytics/series.py`)
`build_price_series(close: pd.Series, period_days: int) -> dict` — з full-history Close рахує:
- `candles`: `[{"time": "YYYY-MM-DD", "value": close}]` за останні `period_days` торгових днів;
- `sma50`, `sma200`: `[{"time","value"}]` (rolling із наявних; лишаємо лише визначені точки в межах
  вікна — NaN відкидаємо);
- `rsi`: `[{"time","value"}]` (reuse `analytics.indicators.rsi`, у межах вікна).
SMA/RSI рахуються на повній історії, потім зрізаються до вікна (щоб SMA200 була визначена).
Чиста, тестована на заморожених рядах.

### 2.2 Endpoint (`api/core.py` + `api/main.py`)
- `core.prices(ticker, period) -> dict`: `period ∈ {1M,3M,6M,1Y}` → `period_days ∈ {21,63,126,252}`;
  тягне довгу історію (resilient `fetch_yfinance_long`), tz-normalize, `build_price_series`.
- `GET /api/prices?ticker=AAPL&period=6M` → JSON `{ticker, period, candles, sma50, sma200, rsi}`.
  Невалідний period → 400/дефолт 6M. Тести: `build_price_series` (чистий) + endpoint TestClient
  (monkeypatch `core.prices`).

## 3. План Б — frontend редизайн + графіки

### 3.1 Дизайн-система (`src/ui/`)
- Токени (Tailwind): фон deep-slate, поверхні slate-900/800, бордери slate-700, текст slate-100/400;
  акцент cyan; семантика emerald(bull/buy)/amber(neutral/hold)/rose(bear/sell). `tabular-nums` +
  mono для чисел.
- Примітиви: `Card`, `Pill`, `ScoreBar` (−1…+1, нульова вісь), `Meter` (confidence), `StatTile`
  (KPI), `Chip`, `Gauge` (радіальний/бар SVG), `TimeframeTabs`.

### 3.2 App-shell
Темний фон на всю сторінку, top-bar: бренд + таби (active-стан) + **health-чіп** (`/api/health`).
Адаптив під вузькі екрани; side-gutter ≥16px.

### 3.3 Analyze
- **Progress-rail** 4 агентів (pending→running(pulse)→done) під час стріму.
- **PriceChart** (lightweight-charts): area/line ціни + оверлеї **SMA50/200** + `TimeframeTabs`
  (1M/3M/6M/1Y) → фетч `/api/prices`.
- **RSI-сабчарт** (окрема панель lightweight-charts, лінії рівнів 30/70).
- **AgentCard** (як у мокапі): акцент-смуга по стансу, `Pill`, `ScoreBar`, confidence, key-facts
  `Chip`-и; skip → приглушено.
- **Метрики-гейджі:** fundamentals (P/E, ROE, D/E) + risk (vol/beta/drawdown) як `Gauge`.
- **VerdictCard:** великий кольоровий бейдж + score + confidence-`Meter` + наратив-панель +
  risk-caution callout.
- Стани: скелетони/shimmer під час рахунку, fade-in карток, error.

### 3.4 Backtest
- **StatTile**-и (hit rate / mean return / IC) по горизонтах.
- **Equity-крива** long-short + **forward-return по горизонтах** (lightweight-charts).
- Progress-список під час прогону.

### 3.5 Ingest / health
Чиста форма + результат-бейдж; health-банер у стилі теми.

### 3.6 Залежності / реалізація
`npm install lightweight-charts`. Presentational-only: типи/`client.ts`/SSE не чіпаємо (додаємо лише
`getPrices` у client). Компоненти-графіки — тонкі обгортки lightweight-charts (ефект-хук: create
chart, set data, cleanup). Тести Vitest: примітиви (`ScoreBar`/`Gauge`/`StatTile`) + рендер карток
на фікстурах; графік-компоненти верифікуємо наживо (canvas у jsdom не рендериться — тестуємо, що
монтується без помилки з мокнутою lib).

## 4. Тестування
- А: `build_price_series` (кількість точок = period; SMA/RSI лише визначені; time-формат) +
  `/api/prices` (форма JSON, невалідний period).
- Б: Vitest на примітиви й картки (значення/кольори/SVG-точки); графік-компоненти — mount-тест із
  мокнутою lightweight-charts; жива перевірка в браузері (ціна+SMA+RSI+гейджі рендеряться,
  таймфрейми перемикаються, стрімінг-поліш працює).

## 5. Обробка помилок / краєві випадки
- `/api/prices` тікер без даних → порожні серії; чарт показує «no data».
- lightweight-charts недоступна/помилка → чарт-панель показує fallback-повідомлення, решта UI живе.
- Короткі періоди / молоді тікери → SMA200 може бути порожньою → просто не малюємо цю лінію.
- Health down → банер + чарт/analyze показують зрозумілу помилку.

## 6. Ризики / відкриті питання
- **lightweight-charts у React** (ефект-lifecycle, resize, theme) — стандартний патерн, але потребує
  акуратного cleanup; canvas не тестується в jsdom (тому mount-тест + жива перевірка).
- **Розмір бандла** (+lightweight-charts ~45KB gzip) — прийнятно.
- **Точність відображення** (SMA warmup) — рахуємо на повній історії, зрізаємо вікно.
