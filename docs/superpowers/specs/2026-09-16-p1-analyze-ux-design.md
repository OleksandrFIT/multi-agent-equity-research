# P1 — Analyze UX (ticker-tape + LLM ticker resolution) — design

Дата: 2026-09-16
Статус: чернетка на затвердження
Частина декомпозиції: P1 (з P1→P2→P3).

## Мета
Зробити вибір акції в Analyze наочним і стійким до помилок:
1. **Ticker-tape** зверху — 50 великих акцій з ціною та %-зміною, що їдуть справа наліво по колу; клік → аналіз.
2. **LLM-виправлення тікера** — некоректний ввід (`APPL`) автоматично зводиться до валідного (`AAPL`) з перевіркою, що така ціна існує; якщо нічого схожого — чесне «такої акції немає».

## Backend

### `GET /api/quotes?tickers=AAPL,MSFT,...`
- Повертає `[{ticker, price, change_pct}]` — остання ціна та денна %-зміна.
- Реалізація: batch через yfinance (короткий період, останні 2 закриття), tz-normalize; часткові збої (тікер без даних) просто пропускаємо.
- **Кеш на день** (in-process, ключ = дата) — тап не б'є мережу на кожен рендер.
- `core.quotes(tickers: list[str]) -> list[dict]`; невалідний/порожній список → `[]`.

### `GET /api/resolve?query=<raw>`
- Повертає `{input, resolved, corrected}`.
- Логіка (`core.resolve(query)`):
  1. `q = query.strip().upper()`; якщо `q` має цінову історію → `{input, resolved: q, corrected: false}`.
  2. Інакше **LLM** пропонує найімовірніший тікер (промпт: «Reply with ONLY the most likely valid US stock ticker for '<query>', or NONE»); нормалізуємо до верхнього регістру.
  3. Кандидат валідуємо **перевіркою цінової історії** (щоб відсікти галюцинації). Якщо валідний і ≠ q → `{input, resolved: <cand>, corrected: true}`.
  4. Інакше → `{input, resolved: null, corrected: false}`.
- Чиста функція `resolve_ticker(query, llm_guess, price_ok) -> dict` (тестована на фейках), плюс тонкий seam у `core`.

## Frontend (`Multi-Agent Equity Client`)

### `TickerTape`
- Тягне `/api/quotes` для сталого списку **TOP50** (курований масив тікерів у `src/ui/top50.ts`).
- Рендерить marquee (CSS-анімація, справа наліво, безкінечний цикл; дубльований контент для безшовності).
- Кожен елемент: `TICKER  $price  ▲/▼ change%` (emerald/rose), клікабельний → `onSelect(ticker)`.
- `prefers-reduced-motion` → без анімації (статичний скрол-ряд).

### Analyze — новий флоу
- **Розкладка:** `TickerTape` зверху (над пошуком). Пошук нижче. Результати (chart+agents+verdict) — під пошуком.
- **Сабміт/клік:** спершу `getResolve(query)`:
  - `resolved && corrected` → показати нотатку «Показую **AAPL** (ви ввели APPL)» і аналізувати `resolved`.
  - `resolved && !corrected` → аналізувати одразу.
  - `resolved == null` → повідомлення «Такої акції не знайшлося — можливо, некоректна назва», без запуску аналізу.
- `client.ts`: `getQuotes(tickers)`, `getResolve(query)`; типи `Quote`, `Resolve`.

## Тестування
- Backend: `resolve_ticker` (валідний одразу; корекція APPL→AAPL через фейк-LLM+price_ok; null коли LLM=NONE або кандидат невалідний); `core.quotes` (мапінг+кеш, часткові збої); ендпоінти через TestClient (monkeypatch core).
- Frontend: `TickerTape` (рендер елементів, клік → onSelect; мок getQuotes); Analyze-флоу (resolve→analyze; нотатка корекції; «не знайдено») з моком client.
- Жива: тап їде з цінами; `APPL`→показує AAPL і аналізує; `ZZZZ`→«не знайдено»; клік по тапу запускає аналіз.

## Помилки / краєві
- `/api/quotes` увесь батч впав → `[]`; тап показує заглушку «quotes unavailable», решта UI жива.
- `/api/resolve` LLM недоступний → трактуємо як `resolved:null` (не валимо аналіз); повідомлення «не вдалося перевірити тікер».
- Дуже короткий/сміттєвий ввід → `resolved:null`.

## Відкриті рішення (підтвердити при рев'ю)
1. Склад TOP50 — пропоную стандартний список мегакапів США (AAPL, MSFT, NVDA, GOOGL, AMZN, META, ... + кілька секторних). Ок?
2. Тап оновлюється раз на день (кеш). Достатньо, чи хочеш «живіше» (напр., щогодини)?

## Поза межами (окремі P2/P3)
Вкладка News/tool-інжест (P2); backtest-деталізація та калібрація (P3); реалтайм-котирування.
