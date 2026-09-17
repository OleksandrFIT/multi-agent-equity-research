# P3 — Confidence calibration — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`). Tasks tagged **[CONTROLLER-LIVE]** need the backtest data / Ollama — the controller runs them.

**Goal:** Temper verdict confidence by measured historical hit-rate (only lowers it; never changes the verdict), with a transparent note. Off by default; enabled in config.yaml.

**Architecture:** `eval/calibration.py` builds a `{horizon: {buy, sell}}` hit-rate map and a factor `min(1, 2*hit)`. The aggregator multiplies confidence by that factor when a calibration map is supplied, and records a `calibration_note`. A `calibration.json` artifact is generated from the A3 backtest.

**Two repos / two branches:**
- **Part A** — backend `/Users/oleksandr/Documents/LLM/Multi-Agent Equity`, branch `feature/p3-calibration`.
- **Part B** — client `/Users/oleksandr/Documents/LLM/Multi-Agent Equity Client`, branch `feature/p3-calibration-ui`.

**Attribution:** commit ONLY as the repo author, NO `Co-Authored-By`. Client repo: no Claude/AI references.

---

## Part A — Backend

### Task A1: `eval/calibration.py`

**Files:**
- Create: `equity_research/eval/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write failing tests (`tests/test_calibration.py`)**

```python
from datetime import date

from equity_research.eval.backtest import BacktestRecord
from equity_research.eval.calibration import build_calibration, calibration_factor


def test_build_calibration_hit_rates():
    recs = [
        BacktestRecord("A", date(2024, 1, 5), "buy", 0.5, {21: 0.1}),    # buy, up -> hit
        BacktestRecord("B", date(2024, 1, 5), "buy", 0.3, {21: -0.1}),   # buy, down -> miss
        BacktestRecord("C", date(2024, 1, 5), "sell", -0.4, {21: 0.2}),  # sell, up -> miss
    ]
    c = build_calibration(recs, [21])
    assert c["21"]["buy"] == 0.5   # 1 of 2 buys up
    assert c["21"]["sell"] == 0.0  # sell went up -> not a hit


def test_calibration_factor_only_lowers():
    c = {"21": {"buy": 0.62, "sell": 0.14}}
    assert calibration_factor(c, "buy", 21) == 1.0             # min(1, 1.24)
    assert abs(calibration_factor(c, "sell", 21) - 0.28) < 1e-9
    assert calibration_factor(c, "hold", 21) == 1.0           # non-directional
    assert calibration_factor({}, "buy", 21) == 1.0           # missing -> no-op
    assert calibration_factor(c, "buy", 63) == 1.0            # horizon missing -> no-op
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_calibration.py -v` (module not found).

- [ ] **Step 3: Implement `equity_research/eval/calibration.py`**

```python
from __future__ import annotations

from equity_research.eval.backtest import BacktestRecord, metrics_for_horizon


def build_calibration(records: list[BacktestRecord], horizons: list[int]) -> dict:
    """Per-horizon directional hit-rate: {"<h>": {"buy": r, "sell": r}}."""
    return {str(h): metrics_for_horizon(records, h)["hit_rate"] for h in horizons}


def calibration_factor(calibration: dict, verdict: str, horizon: int) -> float:
    """Confidence multiplier in (0, 1]. Classes hitting >=50% are untouched;
    worse classes are tempered toward 0. Hold / missing data -> 1.0 (no-op)."""
    if verdict not in ("buy", "sell"):
        return 1.0
    hit = calibration.get(str(horizon), {}).get(verdict)
    if hit is None:
        return 1.0
    return min(1.0, 2.0 * hit)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_calibration.py -v`.

- [ ] **Step 5: Commit**

```bash
git add equity_research/eval/calibration.py tests/test_calibration.py
git commit -m "feat(eval): confidence calibration from measured hit-rate"
```

---

### Task A2: Config fields

**Files:**
- Modify: `equity_research/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Append a test to `tests/test_config.py`**

```python
def test_calibration_config_defaults():
    from equity_research.config import Config

    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".c",
                 edgar_user_agent="x", weights={"fundamentals": 1})
    assert cfg.calibration_enabled is False
    assert cfg.calibration_path == "calibration.json"
    assert cfg.calibration_horizon == 21
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_config.py -v -k calibration_config`.

- [ ] **Step 3: Add fields to `equity_research/config.py`** (after `self_critique_enabled`):

```python
    calibration_enabled: bool = False
    calibration_path: str = "calibration.json"
    calibration_horizon: int = 21
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_config.py -v`.

- [ ] **Step 5: Commit**

```bash
git add equity_research/config.py tests/test_config.py
git commit -m "feat(config): calibration settings"
```

---

### Task A3: Aggregator calibration hook + `Verdict.calibration_note`

**Files:**
- Modify: `equity_research/orchestration/aggregator.py`
- Modify: `tests/test_aggregator.py`

- [ ] **Step 1: Append failing tests to `tests/test_aggregator.py`**

```python
def test_calibration_tempers_sell_confidence(tmp_path):
    cfg = _cfg()
    calib = {"21": {"buy": 0.62, "sell": 0.14}}
    agg = Aggregator(cfg, _client(tmp_path), calibration=calib)
    ops = [AgentOpinion(agent="fundamentals", stance="bearish", score=-0.8, confidence=0.9, rationale="r")]
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=[])
    assert v.verdict == "sell"
    assert abs(v.confidence - 0.9 * 0.28) < 1e-9   # tempered by min(1, 2*0.14)
    assert v.calibration_note is not None


def test_calibration_no_op_for_buy_above_threshold(tmp_path):
    cfg = _cfg()
    calib = {"21": {"buy": 0.62, "sell": 0.14}}
    agg = Aggregator(cfg, _client(tmp_path), calibration=calib)
    ops = [AgentOpinion(agent="fundamentals", stance="bullish", score=0.8, confidence=0.9, rationale="r")]
    v = agg.aggregate("AAPL", date(2026, 9, 15), ops, skipped=[])
    assert v.verdict == "buy"
    assert abs(v.confidence - 0.9) < 1e-9   # buy hit >= 50% -> untouched
    assert v.calibration_note is None
```

(These assume `pm_enabled` is False in `_cfg()` — the mechanical path — so confidence is the raw `base_conf`; a single fundamentals opinion normalizes its weight to 1.0, so `base_conf = 0.9`.)

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_aggregator.py -v -k calibration` (`Aggregator` has no `calibration`).

- [ ] **Step 3: Update `equity_research/orchestration/aggregator.py`**

Add `calibration_note` to `Verdict` (after `caution`):
```python
    calibration_note: str | None = None
```

Extend `Aggregator.__init__` to accept the calibration map:
```python
    def __init__(self, config: Config, client: OllamaClient, buy_th: float = 0.2, sell_th: float = -0.2, calibration: dict | None = None):
        self.config = config
        self.client = client
        self.buy_th = buy_th
        self.sell_th = sell_th
        self.calibration = calibration
```

In `aggregate`, after the risk-gate block computes `confidence` and `caution`, and before building the final `Verdict`, insert:
```python
        calibration_note = None
        if self.calibration is not None:
            from equity_research.eval.calibration import calibration_factor
            factor = calibration_factor(self.calibration, verdict, self.config.calibration_horizon)
            if factor < 1.0:
                confidence = confidence * factor
                calibration_note = f"Confidence tempered ×{factor:.2f} by historical {verdict} accuracy"
```

Pass it into the final `Verdict(...)` (the normal path): add `calibration_note=calibration_note`. (The early-return insufficient/empty paths keep the default None.)

- [ ] **Step 4: Run to verify pass + full suite** — `uv run pytest tests/test_aggregator.py -v` then `uv run pytest -q` (existing tests unchanged: `calibration` defaults None → no tempering).

- [ ] **Step 5: Commit**

```bash
git add equity_research/orchestration/aggregator.py tests/test_aggregator.py
git commit -m "feat(aggregator): temper confidence by calibration, with a note"
```

---

### Task A4: Wire calibration + enable in config.yaml

**Files:**
- Modify: `equity_research/cli.py`
- Modify: `config.yaml`

- [ ] **Step 1: In `analyze_ticker`**, load the calibration map when enabled and pass it to the aggregator:

```python
    calibration = None
    if cfg.calibration_enabled:
        import json
        from pathlib import Path
        p = Path(cfg.calibration_path)
        if p.exists():
            calibration = json.loads(p.read_text())
```

Then build the aggregator with it: `Aggregator(cfg, narrative_client, calibration=calibration)` (keep the critic/Orchestrator wiring from Q1/Q2 intact — only the Aggregator construction gains `calibration=calibration`).

- [ ] **Step 2:** Leave `build_backtest_verdict` WITHOUT calibration — the backtest is the raw measurement that calibration is derived from; calibrating there would be circular (and confidence is not used by IC anyway). Add a one-line comment there noting this.

- [ ] **Step 3: Enable in `config.yaml`** — add top-level:

```yaml
calibration_enabled: true
```

- [ ] **Step 4: Sanity-load + full suite** — `uv run python -c "from equity_research.config import Config; c=Config.load('config.yaml'); print(c.calibration_enabled, c.calibration_path)"` → `True calibration.json`; `uv run pytest -q` → green. (No `calibration.json` yet → analyze runs without tempering; Part C generates it.)

- [ ] **Step 5: Commit**

```bash
git add equity_research/cli.py config.yaml
git commit -m "feat(cli): load and apply calibration in live analysis"
```

Merge `feature/p3-calibration` into `master` via **superpowers:finishing-a-development-branch** before Part B.

---

## Part B — Frontend

### Task B1: Calibration note on VerdictCard

**Files:**
- Modify: `src/api/types.ts`
- Modify: `src/components/VerdictCard.tsx`
- Modify: `src/components/VerdictCard.test.tsx`

- [ ] **Step 1: Add `calibration_note?: string | null` to `Verdict` in `src/api/types.ts`** (after `caution`).

- [ ] **Step 2: Append a test to `src/components/VerdictCard.test.tsx`**

```tsx
test('shows the calibration note when present', () => {
  render(<VerdictCard verdict={{ ...v, calibration_note: 'Confidence tempered ×0.28 by historical sell accuracy' }} />)
  expect(screen.getByText(/tempered ×0.28/i)).toBeInTheDocument()
})
```

(`v` is the existing fixture at the top of the file.)

- [ ] **Step 3: Run to verify fail** — `npm test -- src/components/VerdictCard.test.tsx`.

- [ ] **Step 4: Render the note in `src/components/VerdictCard.tsx`** — in the normal verdict branch, right after the confidence-meter row, add:

```tsx
        {verdict.calibration_note && (
          <div className="mt-2 text-xs" style={{ color: 'var(--text-mut)' }}>⚖ {verdict.calibration_note}</div>
        )}
```

- [ ] **Step 5: Run to verify pass + full suite + build** — `npm test && npm run build`.

- [ ] **Step 6: Commit**

```bash
git add src/api/types.ts src/components/VerdictCard.tsx src/components/VerdictCard.test.tsx
git commit -m "feat(ui): show confidence-calibration note on the verdict"
```

Merge `feature/p3-calibration-ui` into `master` via **superpowers:finishing-a-development-branch**.

---

## Part C — [CONTROLLER-LIVE]: Generate calibration + live check

- [ ] **Step 1: Generate `calibration.json`** from the existing A3 backtest report (no new long run):

```bash
uv run python -c "
import json
r = json.load(open('/tmp/a3_report.json'))
calib = {h: m['hit_rate'] for h, m in r['horizons'].items()}
open('calibration.json','w').write(json.dumps(calib, indent=2))
print(calib)
"
```
(If `/tmp/a3_report.json` is gone, re-run the A3 backtest first, or build from `render_backtest_json` records.)

- [ ] **Step 2: Commit the artifact**

```bash
git add calibration.json
git commit -m "data: calibration from 10x6 backtest (hit-rate by class)"
```

- [ ] **Step 3: Live check** — analyze a ticker that yields a `sell` and confirm confidence is tempered + the note appears; confirm a `buy` is untouched:

```bash
uv run python -c "
from datetime import date
from equity_research.cli import analyze_ticker
for t in ['XOM','AAPL']:
    v = analyze_ticker(t, date.today(), 'config.yaml')
    print(t, v.verdict, 'conf', round(v.confidence,3), '| note:', v.calibration_note)
"
```
Expected: any `sell` verdict shows a reduced confidence + note; `buy`/`hold` untouched. Verdict/stances unchanged → golden still valid (spot-check `-m integration tests/integration/test_verdicts_golden.py`).

- [ ] **Step 4: Live UI** — analyze, confirm the calibration note renders under the confidence meter. Stop servers.

---

## Completion
Both branches merged (backend first); `calibration.json` committed. All commits authored as the repo owner, NO `Co-Authored-By`; client repo carries no Claude/AI references.
