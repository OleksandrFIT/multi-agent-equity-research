# Q1 — Better judge prompts (CoT + few-shot) + model-per-role — design

Дата: 2026-09-16
Статус: чернетка на затвердження
Частина: Q1 (перший крок напрямів 2+3: якість суджень).

## Мета
Підвищити якість суджень кожного агента (fundamentals/technical/sentiment) без зміни архітектури:
1. **Chain-of-thought** — модель спершу міркує, потім видає структурований вердикт (в одному
   JSON-виклику, детерміновано).
2. **Few-shot + per-role rubric** — кілька різнопланових прикладів і чіткіші правила під роль.
3. **Модель-по-ролі** — конфіг дозволяє окрему (сильнішу) модель для судження/наративу, за
   замовчуванням = поточна єдина (без зміни поведінки, поки не задано).

## Поточний стан
- `agents/prompts.py`: `build_judge_prompt(agent, evidence)` + `OPINION_SCHEMA` (stance/score/
  confidence/rationale/key_facts), один `_EXAMPLE`, спільний `_RUBRIC`, метрики з конвенціями.
- `agents/judge.py`: `judge_evidence` → `client.generate_json(prompt, OPINION_SCHEMA)` →
  `AgentOpinion(agent, **raw)` → grounding + reconcile.
- `llm/ollama_client.py`: `generate_json` (grammar `format=schema` + retry + jsonschema),
  cache-ключ включає `model`.
- `config.py`: єдине поле `model`.

## Дизайн

### 1. CoT через поле `reasoning` (в `OPINION_SCHEMA`)
- Додаємо **першою** властивість `reasoning` (string, required). Граматика генерує властивості за
  порядком схеми, тож модель спершу пише міркування, потім stance/score — це «reason-then-answer»
  в одному детермінованому виклику.
- `judge_evidence` **відкидає** `reasoning` (це чернетка) перед `AgentOpinion(agent, **rest)` —
  модель AgentOpinion не змінюється. Промпт інструктує: «think step by step in `reasoning`
  (2-4 sentences) grounded ONLY in the data, then decide».

### 2. Few-shot + per-role rubric (`agents/prompts.py`)
- Замінити один `_EXAMPLE` на **3 різнопланові** (bullish/bearish/neutral), кожен із коротким
  `reasoning` (демонструє формат CoT).
- Додати `_ROLE_RUBRIC[agent]` — короткі підказки під роль:
  - fundamentals: зважай P/E vs зростання, ROE/леверидж; висока оцінка при слабкому зростанні → ведмеже.
  - technical: тренд і положення SMA50/200 (golden/death cross), RSI як momentum, не напрям сам по собі.
  - sentiment: тон новин; за відсутності релевантних новин → neutral, низька confidence.
- Загальний `_RUBRIC` лишаємо; додаємо per-role блок у промпт.

### 3. Модель-по-ролі
- `config.py`: додати `judge_model: str | None = None`, `narrative_model: str | None = None`
  (за замовчуванням None → падають назад на `model`).
- `cli.analyze_ticker` і `build_backtest_verdict`: будувати **judge-клієнт** з
  `cfg.judge_model or cfg.model` і передавати його агентам; **narrative-клієнт** з
  `cfg.narrative_model or cfg.model` — Aggregator. Спільний DiskCache. Коли моделі однакові —
  ідентична поведінка (той самий кеш-ключ).
- Використати більшу модель — це `ollama pull <model>` користувачем + запис у config; код готовий.

## Файли
- `equity_research/agents/prompts.py` — `reasoning` у схемі, per-role rubric, 3 few-shot.
- `equity_research/agents/judge.py` — відкидати `reasoning`.
- `equity_research/config.py` — `judge_model`, `narrative_model`.
- `equity_research/cli.py` — окремі judge/narrative клієнти.

## Тестування
- `OPINION_SCHEMA` має `reasoning` першим і required; `build_judge_prompt` містить CoT-інструкцію,
  per-role rubric і ≥3 приклади.
- `judge_evidence` відкидає `reasoning` і будує валідний `AgentOpinion` (мок generate_json, що
  повертає reasoning+поля).
- `Config` дефолти: `judge_model`/`narrative_model` = None; `judge_model or model` працює.
- Жива: analyze AAPL — вердикт формується, у сирому JSON видно `reasoning`; **re-capture
  verdicts_golden** (промпти змінились → baseline оновлюємо як новий знімок).

## Відкриті рішення (підтвердити)
1. CoT — поле `reasoning` в одному виклику (реком.) чи окремий текстовий крок?
2. Модель-по-ролі — вводимо конфіг зараз із дефолтом = поточна модель (реком.), чи відкладемо?

## Поза межами
Q2 (self-critique), Q3 (LLM-PM), A2/A3; зміна архітектури агрегації; реальний tool-calling.
