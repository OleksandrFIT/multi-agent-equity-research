# Multi-Agent Equity Research — Phase 4 design (web UI)

Дата: 2026-09-15
Статус: затверджено (brainstorming), готово до планування (writing-plans)
Попередні етапи: Phase 0/1 + 2A + 2B + 2.6 + 3A + 3B у `master` (працює наживо на `qwen2.5:7b`).
Базовий дизайн: `docs/superpowers/specs/2026-09-15-multi-agent-equity-research-design.md`.

## 1. Мета і межі

Веб-інтерфейс над наявним Python-ядром: аналіз тікера зі стрімінгом прогресу по-агентно,
backtest, і ingest новин. Ядро (оркестратор/агенти/RAG/backtest) майже не змінюється — лише
додається опційний streaming-хук.

**Стек (зафіксовано):** React + TypeScript + Vite + Tailwind (frontend); FastAPI (backend);
SSE для стрімінгу; inline-SVG для графіка (нуль залежностей).

**Структура (окремі проєкти/репо):**
```
Multi-Agent Equity/            # Python-репо: equity_research/ (ядро) + api/ (FastAPI, план 4A)
Multi-Agent Equity Client/     # окремий git-репо: React+TS+Vite (план 4B)
```
Фронтенд не імпортує Python — єдиний зв'язок HTTP API. Тому 4B — власний репозиторій;
`api/` живе в Python-репо, бо імпортує `equity_research`.

**Декомпозиція (два плани):** 4A (FastAPI backend) → 4B (React frontend).

**Поза межами:** авторизація/мультиюзер; деплой у хмару; realtime co-editing; графіки понад
long-short криву.

**Обсяг екранів v1:** Analyze, Backtest, Ingest.

## 2. План 4A — FastAPI backend

### 2.1 Streaming-хук в оркестраторі
`Orchestrator.run(ticker, as_of, on_event=None)` — опційний колбек, викликається після кожного
агента з подією: `{"agent": name, "opinion": AgentOpinion}` або `{"agent": name, "skipped": True,
"reason": str}`. Backward-compatible (default `None` → поведінка не змінюється). Це єдина зміна
ядра; уся HTTP-логіка — в `api/`.

### 2.2 Ендпоінти (`api/main.py`, тонкі обгортки)
- `GET /api/health` → `{ok, model, ollama_reachable}` — пінг Ollama (для дружніх помилок в UI).
- `GET /api/analyze?ticker=AAPL` → **SSE-стрім** подій:
  - `event: agent` — по кожному агенту (opinion або skip+reason) у міру готовності;
  - `event: verdict` — фінальний `Verdict` (JSON) після агрегації;
  - `event: error` — при збої (напр. Ollama недоступна).
- `POST /api/ingest {ticker}` → `{ticker, ingested: n}` (швидко, без стріму).
- `GET /api/backtest` → **SSE-стрім**: `event: progress` по кожному `(тікер×дата)`, наприкінці
  `event: report` (метрики по горизонтах + long-short серія, JSON).
- `GET /api/backtest/config` → `{universe, dates, horizons}` (для UI).

### 2.3 CORS + wiring
- `CORSMiddleware` дозволяє dev-origin фронтенда (за замовчуванням `http://localhost:5173`),
  конфігуровно. У dev Vite-proxy усуває CORS; middleware — запобіжник для не-проксі-сценаріїв.
- `analyze` будує **повний** оркестратор (fundamentals+technical+sentiment+risk) і викликає
  `run(ticker, as_of, on_event=…)`, щоб стрімити події; `backtest` перевикористовує наявний
  `build_backtest_verdict` (без sentiment). Обидва спираються на наявне wiring ядра.
- Запуск: `uvicorn api.main:app` (uvicorn/fastapi — нові dev-залежності Python-репо, опційний
  extra `web`).

### 2.4 Тестування 4A
- `Orchestrator.on_event` — чистий тест: колбек викликається по кожному агенту (успіх і skip).
- FastAPI `TestClient`: `/health` (мокнутий ping), `/ingest` (мокнутий core), `/analyze` SSE —
  ін'єктуємо фейкове ядро/оркестратор, перевіряємо послідовність SSE-подій (agent…, verdict) і
  форму JSON; `/backtest` SSE (фейковий engine) — progress…, report; `/backtest/config`.
- Ніякої реальної мережі/LLM у юнітах — усе через ін'єкцію/мок.

## 3. План 4B — React+TS frontend (окремий репо)

### 3.1 Каркас
- Vite + React + TypeScript + Tailwind; `react-router` (3 роути: /analyze, /backtest, /ingest).
- Dev-proxy у `vite.config.ts`: `/api → http://localhost:8000`.
- `src/api/types.ts` — TS-типи, що дзеркалять JSON API (`Verdict`, `AgentOpinion`, `BacktestReport`),
  синхронізуються вручну (їх мало).
- `src/api/client.ts` — `EventSource`-хелпери для SSE + `fetch` для POST/GET.

### 3.2 Екрани/компоненти
- **Analyze** (`/analyze`): `TickerInput` → відкриває `EventSource('/api/analyze?ticker=…')` →
  `AgentCard` з'являється по кожній `agent`-події (stance/score/confidence/rationale/key_facts,
  або skip+reason) → `VerdictCard` по `verdict`-події (buy/hold/sell headline, score, confidence,
  наратив, risk caution). Стан: loading/streaming/done/error.
- **Backtest** (`/backtest`): показує `universe/dates/horizons` з `/api/backtest/config`; кнопка
  Run → `EventSource('/api/backtest')` → `ProgressList` (тікер×дата) → `BacktestReport` (метрики
  по горизонтах + `LongShortCurve` inline-SVG).
- **Ingest** (`/ingest`): `TickerInput` → POST `/api/ingest` → показує лічильник новин.
- Спільне: `HealthBanner` (з `/api/health`) — попереджає, якщо Ollama недоступна.

### 3.3 Тестування 4B
- Vitest + React Testing Library: рендер `VerdictCard`, `AgentCard`, `BacktestReport`,
  `LongShortCurve` на фікстурах (перевірка тексту/значень/SVG-точок).
- Базовий тест SSE-обробника (мок `EventSource`): події → стан оновлюється.
- (Vitest globals:false → ручний RTL cleanup, якщо знадобиться.)

## 4. Обробка помилок
- `/api/health` фронт показує банером; аналіз при недоступній Ollama → `error`-подія → повідомлення.
- SSE-помилки (мережа/збій агента) → `error`-подія, UI показує зрозуміле, не сирий стек.
- Ingest/backtest — HTTP-помилки → дружні повідомлення.

## 5. Ризики / відкриті питання
- **SSE + повільний backtest** (хвилини) — тримати з'єднання живим; heartbeat-подія за потреби.
- **Дублювання типів** (Python Pydantic ↔ TS) — вручну; малий контракт, ризик розсинхрону низький;
  за потреби пізніше генерувати з OpenAPI.
- **Два процеси в dev** (uvicorn + vite) — задокументувати запуск (README у Client-репо).
- **Node/npm-залежності** у фронтенді — окремий toolchain від Python; свій git тримає їх ізольовано.
