# P2 — News tab (replaces Ingest) — design

Дата: 2026-09-16
Статус: чернетка на затвердження
Частина декомпозиції: P2 (з P1✔→P2→P3).

## Мета
Замінити службову вкладку **Ingest** на **News**: користувач вводить назву або тікер → LLM визначає
компанію → система тягне свіжі новини, **показує їх** і **дописує у векторну БД ті, яких там ще немає**
(з дедупом). Так sentiment-агент має свіжий, релевантний матеріал, а користувач бачить самі новини.

## Підхід до «LLM + tool»
Робимо **код-оркестрацію з LLM-резолвом** (рекомендовано), а не крихке tool-calling на локальній 7B:
1. LLM (через уже наявний `resolve`) зводить назву/тікер до валідного символу (`Apple`→`AAPL`,
   `APPL`→`AAPL`), з перевіркою ціною.
2. Код викликає інструмент отримання новин (`fetch_news`) — це і є «tool».
3. Інжест у векторну БД лише нових (дедуп за `content_hash` уже є в `NewsStore`).

Поведінка для користувача ідентична tool-calling, але надійна й переюзує P1. (Справжнє Ollama
tool-calling — як альтернатива, якщо наполягаєш.)

## Backend

### `NewsStore.upsert` → повертає кількість нових чанків
Зараз повертає `None`. Змінюємо на `int` (к-сть щойно доданих), щоб звітувати «додано N нових».
Наявні виклики не ламаються (ігнорують результат).

### `core.news(query) -> dict`
1. `r = resolve(query)` (переюз `core.resolve`). Якщо `r.resolved is None` →
   `{query, resolved: None, items: [], fetched: 0, added: 0}`.
2. `ticker = r.resolved`; `items = resilient(fetch_news)(ticker)`;
   `chunks = chunk_news(...)`; `added = store.upsert(chunks)`.
3. Повертає `{query, resolved: ticker, corrected: r.corrected,
   items: [{title, url, source, published_at}], fetched: len(items), added}`.

### `POST /api/news` body `{query}` → `core.news(query)`
- `/api/ingest` лишаємо на backend (CLI-парність), просто веб ним більше не користується.

## Frontend (`Multi-Agent Equity Client`)

### Заміна вкладки
- Прибрати `src/pages/Ingest.tsx` і роут `/ingest`; у навбарі (`App.tsx`) замість Ingest — **News** (`/news`).

### `src/pages/News.tsx`
- Інпут (назва або тікер) → `postJSON('/api/news', {query})`.
- Якщо `resolved` є: заголовок «News for AAPL» (+ «(you typed APPL)» якщо corrected) + рядок
  «fetched N · stored M new»; список новин: **заголовок-посилання** (target=_blank, rel=noopener),
  джерело + дата дрібним.
- Якщо `resolved == null`: «No such stock found — the name may be incorrect.»
- Стани: спінер під час запиту, помилка.
- `client.ts`: `getNews(query)` (POST); тип `NewsResult`, `NewsArticle`.

## Тестування
- Backend: `NewsStore.upsert` повертає к-сть нових (unit на фейк-vector-store);
  `core.news` (monkeypatch resolve+fetch_news+store: resolved→items+added; null→порожньо);
  `POST /api/news` через TestClient (monkeypatch `core.news`).
- Frontend: `News.tsx` (рендер списку з фікстури; «not found»; мок getNews); навбар більше не має Ingest,
  має News.
- Жива: `Apple` → News for AAPL + список посилань + «stored M new»; повтор → «stored 0 new»;
  `ZZZZ` → not found.

## Помилки / краєві
- `fetch_news` порожньо/впав → `items: []`, `added: 0`, повідомлення «no recent news».
- LLM/резолв недоступний → `resolved: null` (не валимо), повідомлення про неможливість перевірити.
- Посилання новин: `rel="noopener noreferrer"`, `target="_blank"`.

## Відкриті рішення (підтвердити при рев'ю)
1. Підхід «LLM+tool»: **код-оркестрація з resolve** (реком.) чи справжнє Ollama tool-calling?
2. Показувати самі новини списком (реком.) чи достатньо лічильника «додано N»?

## Поза межами
P3 (backtest-деталізація + калібрація); реалтайм-стрічка новин; фільтри/пошук усередині News.
