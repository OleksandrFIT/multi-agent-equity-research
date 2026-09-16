# Q3 — LLM Portfolio-Manager aggregator — design

Дата: 2026-09-16
Статус: чернетка на затвердження
Частина: Q3 (напрями 2+3). Після Q1✔.

## Мета
Замість суто фіксованої зваженої суми — **мета-агент (LLM-PM)**, що читає всі думки агентів і
видає **обґрунтований** вердикт. Механічний зважений score лишається **якорем** (детермінований,
його використовує backtest IC), а PM накладається зверху — але так, щоб локальна 7B не «збожеволіла».

## Ключовий принцип: PM re-scores, з якорем через blend
- PM бачить усі think-думки (agent, stance, score, confidence, rationale, key_facts) + механічний
  score, і повертає **власний score в [-1,1] + confidence + narrative + reasoning**.
- Фінальний score = `pm_weight*mech + (1-pm_weight)*pm_score` (blend). Вердикт виводимо з фінального
  score → **вердикт і score завжди узгоджені**, а `pm_weight` (дефолт 0.6 на механіку) обмежує вплив
  PM. narrative беремо від PM (замінює окремий narrative-виклик, коли PM увімкнено).
- Risk-гейт confidence лишається як є.

## Увімкнення (безпечний дефолт)
- Код: `pm_enabled: bool = False` — існуюча поведінка й усі агрегатор-тести незмінні (baseline).
- Активуємо в `config.yaml` (`pm_enabled: true`) — щоб додаток одразу працював із PM; golden
  перезнімаємо під PM-on.
- `pm_weight: float = 0.6`. `pm_enabled: false` → точно поточна механіка.

## Backend

### `orchestration/portfolio_manager.py`
- `PM_SCHEMA` = `{reasoning, score[-1..1], confidence[0..1], narrative}` (reasoning першим — CoT).
- `build_pm_prompt(ticker, verdict_ctx, opinions)` — роль «portfolio manager», список думок агентів
  (stance/score/conf/rationale/key_facts), механічний score як орієнтир; інструкція: зважити
  згоду/незгоду, силу доказів і ризик, і дати підсумковий score+narrative. Думки — це вже
  дистиляція доказів; сирий Evidence не потрібен.
- `run_pm(client, ...) -> dict` через `generate_json(PM_SCHEMA)` (детерміновано, кеш).

### `orchestration/aggregator.py`
- `Aggregator.__init__` бере `pm_enabled`/`pm_weight` з `config` (дефолти з Config).
- У `aggregate`, коли `pm_enabled` і є price-directional:
  1. рахуємо механічні `mech_score`, `base_conf`, risk-гейт як зараз;
  2. `pm = run_pm(self.client, ticker, mech_score, opinions)`;
  3. `final = pm_weight*mech_score + (1-pm_weight)*pm["score"]`;
  4. verdict з `final` (ті самі пороги); `confidence` = risk-гейтований `pm["confidence"]`;
     `narrative = pm["narrative"]`;
  5. на будь-якому збої PM (невалідний JSON) — **фолбек на механічний шлях** (поточна поведінка).
- Коли `pm_enabled=False` — гілка не міняється взагалі.

## Файли
- `equity_research/orchestration/portfolio_manager.py` (новий).
- `equity_research/orchestration/aggregator.py` — PM-гілка + фолбек.
- `equity_research/config.py` — `pm_enabled`, `pm_weight`.
- `config.yaml` — `pm_enabled: true`.

## Тестування
- Існуючі агрегатор-тести (pm off) — без змін.
- `build_pm_prompt` містить думки, механічний орієнтир, роль PM; `PM_SCHEMA` має reasoning першим.
- Aggregator PM-on (мок-клієнт повертає PM JSON): фінальний score = очікуваний blend; вердикт із
  blended; narrative = PM; risk-гейт застосовано.
- Aggregator PM-on із невалідним PM JSON → фолбек = механічний score/narrative (як pm off).
- Жива: analyze AAPL з `pm_enabled: true` — вердикт формується, narrative від PM; **re-capture
  verdicts_golden** під PM-on.

## Чесна засторога
При IC≈−0.16 ми не знаємо, чи PM покращує **передбачення** — Q3 дає *обґрунтованіший і читабельніший*
вердикт із запобіжником-якорем, а чи це «краще за монетку», об'єктивно покаже лише **A3 (backtest)**.
Тому PM опційний і з вагою-якорем.

## Відкриті рішення (підтвердити)
1. PM re-scores + blend з механічним якорем (реком.) чи PM обирає лейбл вердикту напряму з guardrail?
2. Дефолт: код OFF + увімкнути в `config.yaml` (реком.), `pm_weight=0.6` на механіку — ок?

## Поза межами
Q2 (self-critique), A2 (докази), A3 (backtest), A1 (tool-агент).
