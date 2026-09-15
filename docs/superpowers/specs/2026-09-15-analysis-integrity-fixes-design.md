# Analysis integrity fixes (A–E) — design

Дата: 2026-09-15
Статус: чернетка на затвердження
Мотивація: під час тестування невалідний тікер `APPL` (друкарська помилка) дав впевнений
«HOLD 80%» на основі лише sentiment-агента із загальними ринковими новинами, а Fundamentals
падав на транзієнтній помилці Ollama `/tokenize`. Система не має брехати, коли даних немає.

## Проблеми і рішення (за пріоритетом)

### A. Невідомий/невалідний тікер → чесний стан «немає даних»
**Зараз:** для `APPL` цінові агенти (fundamentals/technical/risk) падають із
`PriceValidationError: no price history`, але sentiment відпрацьовує на загальних новинах, і
агрегатор усе одно видає вердикт.
**Рішення (pre-flight):** у `analyze_ticker` перед `orch.run` робимо один
`prices.history(ticker, as_of)`. Якщо `PriceValidationError` → **коротке замикання**: агенти НЕ
запускаються, повертаємо `Verdict` зі `status="unknown_ticker"` (без opinions), фронт показує
панель «Невідомий тікер — немає цінових даних», без бейджа buy/hold/sell. Так ми не показуємо
оманливі загальні новини для неіснуючого тікера.

### B. Недостатньо даних (валідний тікер, але цінові агенти впали)
**Зараз:** якщо fundamentals+technical впали (напр. транзієнтна помилка), а sentiment відпрацював,
агрегатор рахує впевнений HOLD із самого sentiment.
**Рішення:** в `Aggregator` вводимо поняття **price-based directional = {fundamentals, technical}**.
Якщо жоден із них не дав opinion → `status="insufficient_data"`, `confidence=0`, без нормального
вердикту. Фронт показує «Недостатньо даних для оцінки {ticker}» + список skip-причин замість
бейджа. (Sentiment сам по собі більше не «керує» вердиктом.)

**Контракт (спільний для A+B):** додаємо в `Verdict` поле
`status: Literal["ok","unknown_ticker","insufficient_data"] = "ok"`.
Фронт: якщо `status != "ok"` — рендерить інформаційну панель замість звичайної шапки вердикту.
Ціновий графік уже коректно показує «no data».

### C. Релевантність новин sentiment (валідні тікери)
**Зараз:** для валідного тікера sentiment може підтягнути загальні ринкові новини, не про компанію.
**Рішення (легке, ітеративне):** best-effort дістаємо коротку назву компанії
(`yfinance Ticker(t).info` short/long name, кешовано, необов'язково), і у `SentimentAgent.gather`
фільтруємо отримані тексти: лишаємо ті, що згадують тікер АБО назву компанії; якщо назва недоступна
— фільтр не застосовуємо (щоб не відкидати зайве). Якщо після фільтра порожньо → sentiment =
«no news» (neutral, conf 0), а не загальний шум. **Це найменш певний пункт** — можливо, зажадає
подальшого доопрацювання; робимо останнім.

### D. Стійкість до транзієнтних помилок Ollama-embed
**Зараз:** `/tokenize connection reset by peer` під час embed валить агента (Fundamentals skipped).
**Рішення:** обгортаємо embed-операції у `ChromaVectorStore` (`add`, `mmr_search`) у
`resilient_call` (attempts/base_delay з `cfg.net`). `ChromaVectorStore` приймає необов'язковий
`net` dict; коли він є — ретраї активні. Покриває і новини, і файлінги (спільний vector store).

### E. Лоадер при зміні таймфрейму (фронт)
**Зараз:** при перемиканні періоду старий графік «висить» без індикатора до приходу даних.
**Рішення:** у `PricePanel` тримаємо `loading`; під час перефетчу показуємо тонкий індикатор і
приглушуємо старий графік (не прибираємо його — плавніше).

## Архітектура / файли
- **Backend**
  - `equity_research/orchestration/aggregator.py` — `Verdict.status`; правило price-based directional (B); шлях insufficient_data.
  - `equity_research/cli.py` (`analyze_ticker`) — pre-flight price check → `status="unknown_ticker"` (A).
  - `equity_research/agents/sentiment.py` (+ невеликий helper для назви компанії) — фільтр релевантності (C).
  - `equity_research/rag/chroma_store.py` — `net`-ретраї навколо embed (D).
- **Frontend** (`Multi-Agent Equity Client`)
  - `src/api/types.ts` — `Verdict.status`.
  - `src/components/VerdictCard.tsx` — гілки для `unknown_ticker` / `insufficient_data` (A+B).
  - `src/components/PricePanel.tsx` — лоадер на зміні періоду (E).

## Тестування
- A: unit — `analyze_ticker` з фейковим `prices`, що кидає `PriceValidationError` → verdict
  `status="unknown_ticker"`, агенти не викликані.
- B: unit `Aggregator` — лише sentiment opinion (fundamentals/technical skipped) →
  `status="insufficient_data"`, confidence 0.
- C: unit `SentimentAgent.gather` — фільтр лишає лише релевантні тексти; без назви компанії фільтр
  не застосовується.
- D: unit `ChromaVectorStore` — embed кидає раз, потім успіх → `add`/`mmr_search` повертають
  результат (ретрай спрацював). (Мокнутий Chroma.)
- E: фронт — mount-тест PricePanel показує індикатор під час завантаження; жива перевірка.
- Жива перевірка: `APPL` → «невідомий тікер»; `AAPL` норм; зміна таймфрейму показує лоадер.

## Відкриті рішення (прошу підтвердити при рев'ю)
1. **A:** коротке замикання без новин (реком.) чи все ж показувати новини з ярликом «тікер невідомий»?
2. **C:** фільтр за назвою компанії (реком., але потребує yfinance info) чи відкласти C окремо?

## Поза межами
Нові агенти; зміна вагів; кешування цін між агентами (окремий рефактор).
