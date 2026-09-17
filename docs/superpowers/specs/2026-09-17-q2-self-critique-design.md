# Q2 — Self-critique layer — design

Дата: 2026-09-17
Статус: чернетка на затвердження
Частина: Q2 (напрями 2+3). Мета — **довіра/стійкість**: ловити необґрунтовані судження, а не edge
(вимір показав, що інкременти якості edge не дають).

## Мета
Після того як агент видав думку, **критик-LLM** звіряє її з доказами і:
- може **тільки знизити** confidence (ніколи не підняти) — консервативно, не погіршує;
- ставить коротку **примітку** (`critique`), коли стенс/раціонал слабо підкріплені доказами.
Це розширює наявний `ground()` (він ріже необґрунтовані **факти**) на рівень **міркування**.

## Дизайн

### `AgentOpinion.critique: str | None = None`
Нове необов'язкове поле — примітка критика (нема проблем → None).

### `agents/critic.py`
- `CRITIC_SCHEMA` = `{reasoning, supported (bool), confidence [0..1], issue (str)}` (reasoning першим).
- `build_critic_prompt(agent, evidence, opinion)` — дає метрики+контекст (як untrusted data) і думку
  (stance/score/rationale/key_facts); питає: чи виправдовує доказова база стенс; яка confidence
  доречна; якщо раціонал спирається на непідкріплене — прапорець.
- `critique(client, agent, evidence, opinion) -> AgentOpinion`:
  `new_conf = min(opinion.confidence, critic.confidence)`;
  `note = None if supported else (issue or "unsupported by evidence")`;
  повертає `opinion.model_copy(confidence=new_conf, critique=note)`. Будь-який збій критика → думка
  без змін.
- `make_critic(client)` → функція `(agent, evidence, opinion)`, що **пропускає `risk`** (щоб не
  чіпати risk-gate) і критикує решту.

### Хук — в оркестраторі (там є і evidence, і opinion)
- `Orchestrator.__init__(agents, aggregator, critic=None)`.
- У `run`: після `opinion = agent.judge(evidence)` → `if self.critic: opinion = self.critic(agent.name, evidence, opinion)` → далі як зараз (metrics, append, on_event).

### Увімкнення (безпечний дефолт)
- `config.py`: `self_critique_enabled: bool = False`.
- `cli.analyze_ticker` / `build_backtest_verdict`: коли увімкнено — `critic = make_critic(judge_client)`,
  передати в `Orchestrator`.
- `config.yaml`: `self_critique_enabled: true`.

### Frontend (мінімально)
- `types.ts`: `critique?: string` в `AgentOpinion`.
- `AgentCard`: якщо `critique` — показати дрібну amber-примітку «⚠ critique: …». Це і є payoff довіри —
  видно, чому confidence знижено.

## Файли
- `equity_research/agents/base.py` (`critique` поле), `equity_research/agents/critic.py` (новий),
  `equity_research/orchestration/orchestrator.py` (хук), `equity_research/config.py`,
  `equity_research/cli.py`, `config.yaml`.
- Frontend: `src/api/types.ts`, `src/components/AgentCard.tsx`.

## Тестування
- `CRITIC_SCHEMA` reasoning-first; `build_critic_prompt` містить стенс/раціонал/метрики.
- `critique` **тільки знижує** confidence (критик >orig → лишається orig); ставить note коли
  `supported=false`; збій → без змін.
- `make_critic` пропускає `risk`.
- Orchestrator із critic застосовує його (мок).
- `AgentOpinion.critique` дефолт None; серіалізація ок.
- Frontend: AgentCard показує critique-примітку; без неї — нічого.
- Жива: analyze AAPL з `self_critique_enabled` — у деяких думок нижча confidence + примітка; вердикт/
  стенси **не міняються** (критик чіпає лише confidence), тож golden лишається валідним (перевірити
  integration).

## Чесна засторога
Це про **довіру й стійкість**, не про сигнал. Додає **ще один LLM-виклик на агента** (повільніше).
Тому опційно й дефолт off.

## Відкриті рішення (підтвердити)
1. Критик лише знижує confidence + примітка (реком.) — ок?
2. Пропускати `risk` (реком.) — ок?

## Поза межами
A1 (tool-агент), сигнальні/режимні зміни.
