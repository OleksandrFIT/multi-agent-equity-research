# Multi-Agent Equity Research — Phase 3 design (eval / backtest)

Дата: 2026-09-15
Статус: затверджено (brainstorming), готово до планування (writing-plans)
Попередні етапи: Phase 0/1 + 2A + 2B + 2.6 у `master` (працює наживо на `qwen2.5:7b`).
Базовий дизайн: `docs/superpowers/specs/2026-09-15-multi-agent-equity-research-design.md`.

## 1. Мета і межі

Побудувати чесний eval — історичний backtest вердиктів із point-in-time коректністю, плюс
відтворювану retrieval-relevance метрику. Reproducibility-тест уже існує з Phase 0/1.

**Декомпозиція (два плани):**
- **План 3A — as-of EDGAR** (point-in-time fundamentals). Самодостатній; чистить і live-фундаменталку.
- **План 3B — backtest engine + метрики + CLI + retrieval hit@k**.

**Поза межами:** зміна LLM-логіки агентів; web UI (Фаза 4); торгівля.

**Принципи:** числа детерміновані; **жодного look-ahead** у backtest; коректність > латенсі;
LLM-кеш заради відтворюваності.

## 2. План 3A — as-of EDGAR (point-in-time fundamentals)

Наразі `EdgarProvider.company_facts(ticker)` бере **останній** filing — у backtest це look-ahead.

- **`FactsSource.company_facts(ticker, as_of)`** — протокол розширюється параметром `as_of`.
- **`EdgarProvider.company_facts(ticker, as_of)`** — через edgartools вибирає 10-K/10-Q з
  `filing_date ≤ as_of`, бере найсвіжіший, з нього тягне метрики (net_income, revenue, equity,
  total_liabilities, diluted_shares) + попередній період для revenue_growth. Тонка обгортка —
  **верифікуємо наживо** (як EdgarProvider у Phase 1); точний edgartools-API уточнюємо в інтеграції.
- **`FundamentalsAgent.gather(ticker, as_of)`** передає `as_of` у `facts_source.company_facts`.
- **Live поведінка не змінюється:** `as_of = today` → найсвіжіший filing ≤ сьогодні = поточний.
- Fallback: якщо немає filing ≤ as_of (тікер молодший за дату) → підняти помилку → агент
  gracefully skipped (наявний механізм оркестратора).

### Тестування 3A
- Юніт: чиста функція вибору filing (мокнутий список `(form, filing_date)` → обрано найсвіжіший
  ≤ as_of; порожньо → помилка).
- Інтеграція: реальний edgar для двох as_of-дат дає різні (історично коректні) факти.

## 3. План 3B — backtest engine + метрики + hit@k

### 3B.1 Backtest engine (`eval/backtest.py`)
- Вхід: `universe: list[str]`, `dates: list[date]` (as_of-точки), `horizons: [21, 63]` (торгові дні).
- Для кожного `(ticker × as_of)`: будує оркестратор **без sentiment** (live-only, у backtest
  виключений) — тобто **fundamentals(as-of) + technical + risk** — і бере `Verdict`
  (`verdict`, `score`).
- **Обсяг:** мінімальний дефолт (~5 тікерів × 4 дати = 20 прогонів), universe/дати/горизонти в
  конфізі (`backtest`-блок). Перший прогін повільний (LLM), далі кеш.
- Sentiment виключено конструктивно (не додається у список агентів backtest-оркестратора);
  directional-нормалізація вже це толерує.

### 3B.2 Forward-дохідність
- Окремий **full-history** fetch цін (НЕ as-of-обрізаний) для розрахунку майбутнього ретерну.
- Для кожного вердикту: `fwd_return_h = price[as_of + h торгових днів] / price[as_of] − 1` для
  `h ∈ {21, 63}`. Якщо майбутніх цін бракує (as_of близько до «сьогодні») → рядок пропускається
  з поміткою.
- Агенти при цьому бачать лише минулі ціни (PIT через `PriceProvider.history(ticker, as_of)`);
  forward-ретерн береться з майбутніх — коректно розділено.

### 3B.3 Метрики (чисті функції, `eval/metrics.py`)
На парах `(verdict, score, fwd_return)` по кожному горизонту:
- **Hit rate по класах:** частка buy з fwd_return>0 та sell з fwd_return<0.
- **Середній forward-ретерн по класах:** mean(fwd_return) для buy / hold / sell окремо.
- **Information coefficient:** Spearman-кореляція `score ↔ fwd_return` (чи число предиктивне).
- **Naive long-short крива:** покрокова серія та кумулятив (+fwd_return для buy, −fwd_return для
  sell, 0 для hold), згруповано по as_of-датах.

### 3B.4 Вивід
- `eval/report.py`: Markdown + JSON — метрики по обох горизонтах + серія equity-кривої + перелік
  пропущених рядків (нема майбутніх цін / агент skipped). Без важких графічних залежностей —
  крива як дані + компактний текстовий підсумок.
- **CLI:** команда `backtest` (читає `backtest`-конфіг, проганяє, пише звіт у файл).

### 3B.5 Retrieval hit@k (`eval/retrieval_eval.py`)
- **Заморожений міні-корпус** лейбльованих новин (фікстура): набір статей + `(ticker, question,
  expected_content_hash)` пар.
- Інжестимо корпус в ефемерний Chroma → для кожного питання ретрівимо top-k → **hit@k** =
  частка питань, де очікуваний документ у top-k.
- Відтворювано (не залежить від сьогоднішніх новин), але наживо проти ollama-ембедингів+Chroma —
  тому integration-тест.

### 3B.6 Конфіг (`config.yaml`)
```yaml
backtest:
  universe: [AAPL, MSFT, KO, JPM, XOM]
  dates: [2024-03-15, 2024-06-14, 2024-09-13, 2024-12-13]
  horizons: [21, 63]
  report_path: backtest_report.md
```

### Тестування 3B
- Чисті юніти на кожну метрику (hit rate / mean / IC / long-short) на синтетичних
  `(verdict, score, fwd_return)`.
- Forward-return розрахунок на заморожених цінових рядах (правильний зсув на N торгових днів,
  пропуск при нестачі даних).
- Harness на **фейкнутому оркестраторі** (детермінований Verdict) → перевірка агрегації метрик
  без LLM/мережі.
- hit@k на замороженому корпусі (integration, ollama+Chroma).

## 4. Файлова структура (доповнення)
```
equity_research/
  data/edgar.py                 # 3A: FactsSource + EdgarProvider.company_facts(ticker, as_of) + filing-as-of вибір
  agents/fundamentals.py        # 3A: gather передає as_of
  eval/
    backtest.py                 # 3B: engine (universe×dates → verdicts)
    forward.py                  # 3B: forward-return розрахунок
    metrics.py                  # 3B: hit rate / mean / IC / long-short (чисті)
    report.py                   # 3B: MD+JSON звіт
    retrieval_eval.py           # 3B: hit@k на замороженому корпусі
  cli.py                        # 3B: команда backtest
config.yaml                     # 3B: backtest-блок
tests/                          # відповідні тести + eval/fixtures для hit@k
```

## 5. Обробка помилок і краєві випадки
- Немає filing ≤ as_of → fundamentals-агент skipped (як зараз); backtest-рядок усе одно рахується
  на technical+risk.
- Бракує майбутніх цін для горизонту → рядок пропускається з причиною у звіті.
- Порожній клас (напр. жодного sell) → метрики цього класу = n/a, не діляться на нуль.
- Backtest детермінований: seed + temp 0 + кеш → повторний прогін дає ті самі метрики.

## 6. Ризики / відкриті питання
- **edgartools as-of API** — точний спосіб фільтрувати filings по даті уточнюємо в інтеграції 3A;
  ризик знижено тонкою обгорткою.
- **Час першого backtest-прогону** на 7b (навіть 20 прогонів × ~4 LLM-виклики) — десятки хвилин;
  прийнятно за пріоритетом коректності, кеш робить повтори швидкими.
- **Мала вибірка (20)** — статистична потужність низька; звіт чесно показує n по класах.
- **hit@k на замороженому корпусі** тестує ретрівер+ембединги, не якість живих новин — свідоме
  обмеження (repeatable > realistic для цієї метрики).
