# Q3 — LLM Portfolio-Manager aggregator — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`). Tasks tagged **[CONTROLLER-LIVE]** need a running Ollama — the controller runs them.

**Goal:** An optional LLM portfolio-manager that re-scores the verdict from all agent opinions, blended with the mechanical weighted score (the anchor). Off by default in code; enabled in config.yaml.

**Architecture:** New `portfolio_manager.py` (prompt + schema + call). `Aggregator` gains a `_decide` step: PM blend when enabled, mechanical otherwise, with graceful fallback on PM failure. Determinism preserved (seeded JSON call, cached).

**Repo/branch:** backend `/Users/oleksandr/Documents/LLM/Multi-Agent Equity`, branch `feature/q3-portfolio-manager`.

**Attribution:** commit ONLY as the repo author, NO `Co-Authored-By`.

---

### Task 1: `pm_enabled` / `pm_weight` config

**Files:**
- Modify: `equity_research/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Append a test to `tests/test_config.py`**

```python
def test_pm_config_defaults_off():
    from equity_research.config import Config

    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".c",
                 edgar_user_agent="x", weights={"fundamentals": 1})
    assert cfg.pm_enabled is False
    assert cfg.pm_weight == 0.6
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_config.py -v -k pm_config` (FAIL: fields missing).

- [ ] **Step 3: Add fields to `equity_research/config.py`** (after `narrative_model`):

```python
    pm_enabled: bool = False
    pm_weight: float = 0.6
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_config.py -v`.

- [ ] **Step 5: Commit**

```bash
git add equity_research/config.py tests/test_config.py
git commit -m "feat(config): optional portfolio-manager (pm_enabled, pm_weight)"
```

---

### Task 2: `portfolio_manager` prompt + schema + call

**Files:**
- Create: `equity_research/orchestration/portfolio_manager.py`
- Create: `tests/test_portfolio_manager.py`

- [ ] **Step 1: Write failing tests (`tests/test_portfolio_manager.py`)**

```python
from pathlib import Path

from equity_research.agents.base import AgentOpinion
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient
from equity_research.orchestration.portfolio_manager import PM_SCHEMA, build_pm_prompt, run_pm


def _ops():
    return [
        AgentOpinion(agent="fundamentals", stance="bearish", score=-0.4, confidence=0.6,
                     rationale="pricey", key_facts=["P/E 40x"]),
        AgentOpinion(agent="technical", stance="bullish", score=0.5, confidence=0.7, rationale="uptrend"),
    ]


def test_pm_schema_has_reasoning_first():
    assert list(PM_SCHEMA["properties"])[0] == "reasoning"
    assert set(PM_SCHEMA["required"]) == {"reasoning", "score", "confidence", "narrative"}


def test_pm_prompt_lists_opinions_and_reference():
    prompt = build_pm_prompt("AAPL", 0.1, _ops())
    assert "AAPL" in prompt and "portfolio manager" in prompt.lower()
    assert "fundamentals" in prompt and "technical" in prompt
    assert "P/E 40x" in prompt          # key facts surfaced
    assert "+0.10" in prompt            # mechanical reference score shown


def test_run_pm_parses_json(tmp_path: Path):
    payload = ('{"reasoning":"analysts split; trend wins","score":0.3,'
               '"confidence":0.65,"narrative":"Cautiously constructive."}')
    client = OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                          chat_fn=lambda **k: {"message": {"content": payload}})
    out = run_pm(client, "AAPL", 0.1, _ops())
    assert out["score"] == 0.3 and out["narrative"] == "Cautiously constructive."
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_portfolio_manager.py -v` (module not found).

- [ ] **Step 3: Implement `equity_research/orchestration/portfolio_manager.py`**

```python
from __future__ import annotations

import json

from equity_research.agents.base import AgentOpinion
from equity_research.llm.ollama_client import OllamaClient

PM_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": -1.0, "maximum": 1.0},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "narrative": {"type": "string"},
    },
    "required": ["reasoning", "score", "confidence", "narrative"],
}


def build_pm_prompt(ticker: str, mech_score: float, opinions: list[AgentOpinion]) -> str:
    lines = "\n".join(
        f"- {o.agent}: {o.stance} (score {o.score:+.2f}, confidence {o.confidence:.0%}) — {o.rationale}"
        + (f" [facts: {', '.join(o.key_facts)}]" if o.key_facts else "")
        for o in opinions
    )
    return (
        f"You are the portfolio manager deciding on {ticker}. Your analysts' opinions:\n"
        f"{lines}\n"
        f"A mechanical weighted blend of their scores is {mech_score:+.2f} (a reference, not a rule).\n"
        "First reason step by step in `reasoning`: weigh agreement versus disagreement, the strength "
        "of the evidence, and risk. Then output a final score in [-1,1] (sign = buy/sell direction, "
        "magnitude = conviction), a confidence in [0,1], and a 3-4 sentence narrative explaining the "
        "decision and noting any disagreement. Do not use price targets or guarantees.\n"
        f"JSON schema: {json.dumps(PM_SCHEMA)}"
    )


def run_pm(client: OllamaClient, ticker: str, mech_score: float,
           opinions: list[AgentOpinion]) -> dict:
    return client.generate_json(build_pm_prompt(ticker, mech_score, opinions), PM_SCHEMA)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_portfolio_manager.py -v`.

- [ ] **Step 5: Commit**

```bash
git add equity_research/orchestration/portfolio_manager.py tests/test_portfolio_manager.py
git commit -m "feat(orchestration): portfolio-manager prompt, schema, call"
```

---

### Task 3: Aggregator PM branch + fallback

**Files:**
- Modify: `equity_research/orchestration/aggregator.py`
- Modify: `tests/test_aggregator.py`

- [ ] **Step 1: Append failing tests to `tests/test_aggregator.py`**

```python
def test_pm_blends_score_and_uses_pm_narrative(tmp_path):
    import json

    cfg = _cfg()
    cfg.pm_enabled = True
    cfg.pm_weight = 0.5
    payload = json.dumps({"reasoning": "trend wins", "score": 1.0, "confidence": 0.9,
                          "narrative": "PM says buy."})
    client = OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                          chat_fn=lambda **k: {"message": {"content": payload}})
    agg = Aggregator(cfg, client)
    ops = [AgentOpinion(agent="fundamentals", stance="bearish", score=-0.4, confidence=0.5, rationale="r")]
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=[])
    # mech = -0.4 (single fundamentals, weight -> 1.0); blend 0.5*-0.4 + 0.5*1.0 = 0.3
    assert abs(v.score - 0.3) < 1e-9
    assert v.verdict == "buy"
    assert v.narrative == "PM says buy."


def test_pm_failure_falls_back_to_mechanical(tmp_path):
    cfg = _cfg()
    cfg.pm_enabled = True
    client = OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                          chat_fn=lambda **k: {"message": {"content": "not json"}}, max_retries=1)
    agg = Aggregator(cfg, client)
    ops = [AgentOpinion(agent="fundamentals", stance="bullish", score=0.8, confidence=0.9, rationale="r")]
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=[])
    assert abs(v.score - 0.8) < 1e-9   # mechanical fallback
    assert v.verdict == "buy"
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_aggregator.py -v -k "pm_blends or pm_failure"` (FAIL: PM path absent).

- [ ] **Step 3: Update `equity_research/orchestration/aggregator.py`**

Replace the block from `weights = self.config.normalized_weights(...)` down to the `narrative = ...` line with:

```python
        weights = self.config.normalized_weights([o.agent for o in directional])
        mech_score = sum(weights[o.agent] * o.score for o in directional)
        base_conf = sum(weights[o.agent] * o.confidence for o in directional)

        score, base_confidence, narrative = self._decide(ticker, mech_score, base_conf, directional)
        verdict = "buy" if score >= self.buy_th else "sell" if score <= self.sell_th else "hold"

        confidence = base_confidence
        caution = None
        if risk_op is not None:
            rl = risk_op.score
            confidence = max(0.0, min(1.0, base_confidence * (1 - self.config.risk["gate_strength"] * rl)))
            if rl >= self.config.risk["caution_threshold"]:
                caution = f"Elevated risk (level {rl:.0%}): {risk_op.rationale}"
```

(The `opinions_out = ...` and `return Verdict(...)` lines stay as they are.)

Add the `_decide` method to `Aggregator`:

```python
    def _decide(self, ticker, mech_score, base_conf, directional):
        """(score, confidence, narrative): LLM-PM blend when enabled, else mechanical.

        The mechanical score is the deterministic anchor; the PM only shifts it by
        (1 - pm_weight). Any PM failure falls back to the mechanical path.
        """
        if getattr(self.config, "pm_enabled", False):
            from equity_research.orchestration.portfolio_manager import run_pm
            try:
                pm = run_pm(self.client, ticker, mech_score, directional)
                w = self.config.pm_weight
                return (w * mech_score + (1 - w) * pm["score"], pm["confidence"], pm["narrative"])
            except Exception:
                pass  # PM unavailable/invalid -> mechanical fallback
        mech_verdict = "buy" if mech_score >= self.buy_th else "sell" if mech_score <= self.sell_th else "hold"
        narrative = self.client.generate_text(self._narrative_prompt(ticker, mech_verdict, mech_score, directional))
        return mech_score, base_conf, narrative
```

- [ ] **Step 4: Run to verify pass + full suite**

Run: `uv run pytest tests/test_aggregator.py -v` then `uv run pytest -q`
Expected: all pass (existing aggregator tests unchanged — `pm_enabled` defaults False → mechanical path identical).

- [ ] **Step 5: Commit**

```bash
git add equity_research/orchestration/aggregator.py tests/test_aggregator.py
git commit -m "feat(aggregator): LLM portfolio-manager blend with mechanical anchor"
```

---

### Task 4: Enable PM in `config.yaml`

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add** to `config.yaml` (top-level, near `model`):

```yaml
pm_enabled: true
pm_weight: 0.6
```

- [ ] **Step 2: Sanity-load** — `uv run python -c "from equity_research.config import Config; c=Config.load('config.yaml'); print(c.pm_enabled, c.pm_weight)"` → `True 0.6`.

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "config: enable portfolio-manager"
```

---

### Task 5 **[CONTROLLER-LIVE]**: Live check + re-capture verdicts golden

PM is now active, so verdicts change and the baselines are stale.

- [ ] **Step 1:** Run and confirm PM narrative + sensible verdicts:

```bash
uv run python -c "
from datetime import date
from equity_research.cli import analyze_ticker
for t in ['AAPL','MSFT','NVDA']:
    v = analyze_ticker(t, date.today(), 'config.yaml')
    print(t, v.status, v.verdict, round(v.score,3), {o.agent:o.stance for o in v.opinions})
    print('  narrative:', v.narrative[:140])
"
```
Expected: `status ok`, sensible verdicts, narrative reads like a PM decision (mentions agreement/disagreement).

- [ ] **Step 2:** Re-capture `tests/fixtures/golden/verdicts_golden.json` for AAPL, MSFT, NVDA at `date.today()`, hand-review, overwrite.

- [ ] **Step 3:** `uv run pytest -m integration tests/integration/test_verdicts_golden.py` → pass.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/golden/verdicts_golden.json
git commit -m "test(golden): re-capture verdict baselines under portfolio-manager"
```

---

## Completion
Merge `feature/q3-portfolio-manager` into `master` via **superpowers:finishing-a-development-branch**. All commits authored as the repo owner, NO `Co-Authored-By`.
