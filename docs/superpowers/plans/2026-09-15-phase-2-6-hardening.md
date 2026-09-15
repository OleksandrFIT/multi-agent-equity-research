# Phase 2.6 — Hardening (resilience + prompt calibration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the running 4-agent system more robust and better-calibrated: network timeouts + transient-retry on data/LLM calls, visible skip reasons in the report, an explicit scoring rubric with a stance↔score guard, and a one-shot output example.

**Architecture:** A small `resilient_call` helper (thread-based timeout + exponential-backoff retry) wraps the thin data fetchers and is configured at the composition root; skip reasons are captured in the orchestrator and flow through `Verdict` into the report; the judge prompt gains a calibration rubric and a one-shot example, and a pure post-parse guard makes stance agree with score sign.

**Tech Stack:** existing only (stdlib `concurrent.futures`, pydantic, pytest). No new dependencies.

**Scope:** Decided improvements only — timeouts+retry, skip reasons, scoring rubric + stance↔score, few-shot. Explicitly OUT (decided): friendly CLI error wrapping, explicit SEC throttle, system/user message split, confidence guidance. Cross-cutting refinement of the current `master`.

## File Structure

- `equity_research/util/__init__.py`, `equity_research/util/resilient.py` — `resilient_call`, `resilient`.
- `equity_research/config.py` + `config.yaml` — `net` block.
- `equity_research/cli.py` — wrap fetchers + ollama timeout.
- `equity_research/orchestration/orchestrator.py` — capture skip reasons.
- `equity_research/orchestration/aggregator.py` — `Verdict.skip_reasons`, thread reasons.
- `equity_research/reporting/report.py` — show skip reasons.
- `equity_research/agents/prompts.py` — scoring rubric + one-shot example.
- `equity_research/agents/judge.py` — stance↔score reconcile.
- Tests mirror each.

Naming contract: `resilient_call(fn, *, timeout, attempts, base_delay)`, `resilient(fn, net)`, `Config.net`, `Verdict.skip_reasons: dict[str, str]`, `Orchestrator.run` builds `skip_reasons`, `reconcile_stance(opinion)`.

---

## Task 1: resilient_call helper

**Files:**
- Create: `equity_research/util/__init__.py` (empty)
- Create: `equity_research/util/resilient.py`
- Test: `tests/test_resilient.py`

- [ ] **Step 1: Write the failing test**

`tests/test_resilient.py`:
```python
import time

import pytest

from equity_research.util.resilient import resilient, resilient_call


def test_succeeds_first_try():
    assert resilient_call(lambda: 42, timeout=1, attempts=3, base_delay=0) == 42


def test_retries_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert resilient_call(flaky, timeout=1, attempts=3, base_delay=0) == "ok"
    assert calls["n"] == 3


def test_raises_after_attempts():
    def always_fail():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        resilient_call(always_fail, timeout=1, attempts=2, base_delay=0)


def test_times_out():
    def slow():
        time.sleep(0.3)
        return "late"

    with pytest.raises(TimeoutError):
        resilient_call(slow, timeout=0.05, attempts=1, base_delay=0)


def test_resilient_wraps_callable():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("blip")
        return x * 2

    wrapped = resilient(flaky, {"data_timeout": 1, "data_attempts": 3, "data_base_delay": 0})
    assert wrapped(5) == 10
    assert calls["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_resilient.py -v`
Expected: FAIL — import error.

- [ ] **Step 3: Write minimal implementation**

`equity_research/util/__init__.py`: empty.

`equity_research/util/resilient.py`:
```python
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Callable, TypeVar

T = TypeVar("T")


def resilient_call(fn: Callable[[], T], *, timeout: float, attempts: int = 3, base_delay: float = 0.5) -> T:
    """Run fn with a per-attempt timeout and exponential-backoff retry.

    On timeout the underlying worker thread is abandoned (cannot be killed) and
    the next attempt starts fresh. Raises the last error after all attempts.
    """
    last: BaseException | None = None
    for i in range(attempts):
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
            last = TimeoutError(f"call timed out after {timeout}s")
            ex.shutdown(wait=False)
        except Exception as exc:  # transient network / library error
            last = exc
            ex.shutdown(wait=False)
        if i < attempts - 1 and base_delay:
            time.sleep(base_delay * (2 ** i))
    raise last if last is not None else RuntimeError("resilient_call failed")


def resilient(fn: Callable[..., T], net: dict) -> Callable[..., T]:
    """Wrap a callable so each invocation runs through resilient_call using net config."""

    def wrapped(*args, **kwargs) -> T:
        return resilient_call(lambda: fn(*args, **kwargs),
                              timeout=net["data_timeout"], attempts=net["data_attempts"],
                              base_delay=net["data_base_delay"])

    return wrapped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_resilient.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add equity_research/util tests/test_resilient.py
git commit -m "feat(util): resilient_call (timeout + backoff retry)"
```

---

## Task 2: config net block

**Files:**
- Modify: `equity_research/config.py`
- Modify: `config.yaml`
- Test: `tests/test_config.py` (add assertion)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:
```python
def test_config_has_net_defaults():
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=".cache",
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    assert cfg.net["data_timeout"] == 20
    assert cfg.net["data_attempts"] == 3
    assert cfg.net["ollama_timeout"] == 180
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_config_has_net_defaults -v`
Expected: FAIL — `Config` has no `net`.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/config.py`, add a `net` field to `Config` (after `rag`):
```python
    net: dict = Field(default_factory=lambda: {
        "data_timeout": 20,
        "data_attempts": 3,
        "data_base_delay": 0.5,
        "ollama_timeout": 180,
    })
```

In `config.yaml`, add:
```yaml
net:
  data_timeout: 20
  data_attempts: 3
  data_base_delay: 0.5
  ollama_timeout: 180
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (existing config tests + new one).

- [ ] **Step 5: Commit**

```bash
git add equity_research/config.py config.yaml tests/test_config.py
git commit -m "chore(config): net block (timeouts + retry)"
```

---

## Task 3: Wire resilience into the CLI

**Files:**
- Modify: `equity_research/cli.py`
- Test: `tests/test_cli.py` (add assertion that fetchers are wrapped)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:
```python
def test_analyze_ticker_wraps_fetchers_resiliently(monkeypatch):
    # analyze_ticker must wrap the yfinance adapter with resilient(): the wired
    # fetcher on the built technical agent must retry a once-failing adapter.
    import pandas as pd

    captured = {}

    class FakeOrch:
        def __init__(self, agents, aggregator):
            captured["agents"] = agents

        def run(self, ticker, as_of):
            from equity_research.orchestration.aggregator import Verdict
            return Verdict(ticker=ticker, as_of=as_of, verdict="hold", score=0.0,
                           confidence=0.0, narrative="n", opinions=[], skipped_agents=[])

    calls = {"n": 0}

    def flaky(_t):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("blip")
        idx = pd.date_range("2025-01-01", periods=3, freq="D")
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)

    monkeypatch.setattr(cli_module, "Orchestrator", FakeOrch)
    monkeypatch.setattr(cli_module, "fetch_yfinance", flaky)
    cli_module.analyze_ticker("AAPL", date(2026, 9, 15), "config.yaml")

    tech = next(a for a in captured["agents"] if a.name == "technical")
    df = tech.prices.fetch_yfinance("AAPL")  # the wired (resilient) fetcher
    assert df.iloc[-1]["Close"] == 3.0
    assert calls["n"] == 2  # retried once after the first failure
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py::test_analyze_ticker_wraps_fetchers_resiliently -v`
Expected: FAIL — the wired fetcher is the raw adapter (raises on first call / `calls["n"] == 1`), not a resilient-wrapped one. (`resilient`/`cfg.net` are added in Tasks 1-2.)

- [ ] **Step 3: Write minimal implementation**

In `equity_research/cli.py`, add the import:
```python
from equity_research.util.resilient import resilient
```

In `analyze_ticker`, wrap the fetchers and the edgar facts source with `cfg.net`, and pass the ollama timeout. Change the wiring block:
```python
    from ollama import Client

    chat_fn = Client(host=cfg.ollama_host, timeout=cfg.net["ollama_timeout"]).chat
    client = OllamaClient(model=cfg.model, cache=DiskCache(cfg.cache_dir),
                          seed=cfg.seed, temperature=cfg.temperature, chat_fn=chat_fn)
    prices = PriceProvider(fetch_yfinance=resilient(fetch_yfinance, cfg.net),
                           fetch_stooq=resilient(fetch_stooq, cfg.net))
    edgar = EdgarProvider(user_agent=cfg.edgar_user_agent)
    edgar.company_facts = resilient(edgar.company_facts, cfg.net)
```
And in the sentiment wiring, wrap the news fetcher:
```python
        ingest_fn=lambda t: ingest_news(resilient(fetch_news, cfg.net), news_store, t),
```
Also update the standalone `ingest` command to wrap its fetcher:
```python
    n = ingest_news(resilient(fetch_news, cfg.net), store, ticker.upper())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS. Confirm `uv run python -c "import equity_research.cli"` succeeds offline, and the full suite `uv run pytest -m "not integration" -p no:warnings -q` is green.

- [ ] **Step 5: Commit**

```bash
git add equity_research/cli.py tests/test_cli.py
git commit -m "feat(cli): resilient data fetchers + ollama timeout"
```

---

## Task 4: Skip reasons in report

**Files:**
- Modify: `equity_research/orchestration/orchestrator.py`
- Modify: `equity_research/orchestration/aggregator.py`
- Modify: `equity_research/reporting/report.py`
- Test: `tests/test_orchestrator.py`, `tests/test_report.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_orchestrator.py`:
```python
def test_failed_agent_reason_captured(tmp_path):
    agents = [StubAgent("fundamentals", 0.5), StubAgent("technical", 0.0, fail=True)]
    orch = Orchestrator(agents=agents, aggregator=_agg(tmp_path))
    v = orch.run("AAPL", as_of=date(2026, 9, 15))
    assert "technical" in v.skip_reasons
    assert "data unavailable" in v.skip_reasons["technical"]
```

Add to `tests/test_report.py`:
```python
def test_markdown_shows_skip_reason():
    v = _verdict().model_copy(update={"skipped_agents": ["technical"],
                                      "skip_reasons": {"technical": "TimeoutError: call timed out after 20s"}})
    md = render_markdown(v)
    assert "technical" in md
    assert "timed out" in md.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator.py::test_failed_agent_reason_captured tests/test_report.py::test_markdown_shows_skip_reason -v`
Expected: FAIL — `Verdict` has no `skip_reasons`; orchestrator does not capture reasons.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/orchestration/aggregator.py`, add a field to `Verdict` (after `skipped_agents`):
```python
    skip_reasons: dict[str, str] = {}
```
Add a `skip_reasons` parameter to `aggregate` (default None) and pass it into every `Verdict(...)` it returns. Change the signature and both `Verdict(...)` return sites:
```python
    def aggregate(self, ticker: str, as_of: date, opinions: list[AgentOpinion], skipped: list[str],
                  skip_reasons: dict[str, str] | None = None) -> Verdict:
        skip_reasons = dict(skip_reasons or {})
```
Then add `skip_reasons=skip_reasons,` to the empty-`directional` `Verdict(...)` return and the final `Verdict(...)` return.

In `equity_research/orchestration/orchestrator.py`, capture reasons and pass them:
```python
    def run(self, ticker: str, as_of: date) -> Verdict:
        opinions: list[AgentOpinion] = []
        skipped: list[str] = []
        skip_reasons: dict[str, str] = {}
        for agent in self.agents:
            try:
                evidence = agent.gather(ticker, as_of)
                opinions.append(agent.judge(evidence))
            except Exception as exc:
                logger.exception("agent %s failed during run", agent.name)
                skipped.append(agent.name)
                skip_reasons[agent.name] = f"{type(exc).__name__}: {exc}"
        return self.aggregator.aggregate(ticker, as_of, opinions, skipped, skip_reasons)
```

In `equity_research/reporting/report.py`, change the skipped-agents rendering to include reasons. Replace the existing block:
```python
    if verdict.skipped_agents:
        lines.append("")
        lines.append(f"_Skipped agents: {', '.join(verdict.skipped_agents)}_")
```
with:
```python
    if verdict.skipped_agents:
        lines.append("")
        rendered = [
            f"{a} ({verdict.skip_reasons[a]})" if a in verdict.skip_reasons else a
            for a in verdict.skipped_agents
        ]
        lines.append(f"_Skipped agents: {', '.join(rendered)}_")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_orchestrator.py tests/test_report.py tests/test_aggregator.py -v`
Expected: PASS (new + existing; existing aggregator/report tests unaffected since `skip_reasons` defaults to `{}`).

- [ ] **Step 5: Commit**

```bash
git add equity_research/orchestration/orchestrator.py equity_research/orchestration/aggregator.py equity_research/reporting/report.py tests/test_orchestrator.py tests/test_report.py
git commit -m "feat(orchestration): capture and surface agent skip reasons"
```

---

## Task 5: Scoring rubric + one-shot example in the judge prompt

**Files:**
- Modify: `equity_research/agents/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prompts.py`:
```python
def test_prompt_has_scoring_rubric_and_example():
    e = Evidence(ticker="AAPL", as_of=date(2026, 9, 15), metrics={"pe": 20.0})
    prompt = build_judge_prompt("fundamentals", e)
    assert "score must agree with your stance" in prompt.lower()
    assert "0.7" in prompt  # rubric anchor present
    assert '"stance"' in prompt and "example" in prompt.lower()  # one-shot example present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py::test_prompt_has_scoring_rubric_and_example -v`
Expected: FAIL — no rubric/example text yet.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/agents/prompts.py`, add two module constants after `OPINION_SCHEMA`:
```python
_RUBRIC = (
    "Scoring rubric: use score in [-1,1] where the sign is direction and the magnitude is "
    "conviction — strong ±0.7..1.0, moderate ±0.3..0.6, weak/neutral -0.2..0.2. Your score "
    "must agree with your stance: bullish => positive, bearish => negative, neutral => near 0. "
    "Set confidence lower when key data is missing."
)
_EXAMPLE = (
    'Example output: {"stance": "bearish", "score": -0.4, "confidence": 0.6, '
    '"rationale": "Valuation stretched versus modest growth.", "key_facts": ["P/E 40x", "growth 3%"]}'
)
```
Then include them in the returned prompt in `build_judge_prompt` — change the final instruction block so it reads (insert `_RUBRIC` and `_EXAMPLE` before the schema line):
```python
    return (
        f"You are {role} for {evidence.ticker} as of {evidence.as_of}.\n"
        f"Metrics:\n{metrics}\n"
        f"{notes_block}"
        f"{context_block}\n"
        "Return a JSON object with your stance (bullish/neutral/bearish), a score "
        "in [-1,1], a confidence in [0,1], a short rationale, and key_facts (a list "
        "of the specific figures you relied on). Base every fact only on the data above.\n"
        f"{_RUBRIC}\n"
        f"{_EXAMPLE}\n"
        f"JSON schema: {json.dumps(OPINION_SCHEMA)}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS (new test + existing prompt tests — the existing assertions on metrics/delimiting/notes are unaffected by the appended rubric/example).

- [ ] **Step 5: Commit**

```bash
git add equity_research/agents/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): scoring rubric + one-shot output example"
```

---

## Task 6: stance↔score reconcile guard

**Files:**
- Modify: `equity_research/agents/judge.py`
- Test: `tests/test_judge.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_judge.py`:
```python
from equity_research.agents.base import AgentOpinion
from equity_research.agents.judge import reconcile_stance


def test_reconcile_flips_contradictory_stance():
    op = AgentOpinion(agent="fundamentals", stance="bullish", score=-0.5, confidence=0.7, rationale="r")
    fixed = reconcile_stance(op)
    assert fixed.stance == "bearish"  # score is negative -> stance must be bearish


def test_reconcile_leaves_consistent_stance():
    op = AgentOpinion(agent="technical", stance="bullish", score=0.6, confidence=0.7, rationale="r")
    assert reconcile_stance(op).stance == "bullish"


def test_reconcile_leaves_neutral_small_score():
    op = AgentOpinion(agent="sentiment", stance="neutral", score=0.05, confidence=0.5, rationale="r")
    assert reconcile_stance(op).stance == "neutral"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_judge.py::test_reconcile_flips_contradictory_stance -v`
Expected: FAIL — `reconcile_stance` does not exist.

- [ ] **Step 3: Write minimal implementation**

In `equity_research/agents/judge.py`, add the function and apply it in the success branch:
```python
def reconcile_stance(opinion: AgentOpinion) -> AgentOpinion:
    """Make stance agree with the score sign; only fix hard contradictions."""
    if opinion.stance == "bullish" and opinion.score <= -0.1:
        return opinion.model_copy(update={"stance": "bearish"})
    if opinion.stance == "bearish" and opinion.score >= 0.1:
        return opinion.model_copy(update={"stance": "bullish"})
    return opinion
```
Change the success branch (currently `return ground(AgentOpinion(agent=agent, **raw), evidence)`) to reconcile first:
```python
        raw = client.generate_json(prompt, OPINION_SCHEMA)
        return ground(reconcile_stance(AgentOpinion(agent=agent, **raw)), evidence)
```
(`AgentOpinion` is already imported in judge.py.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_judge.py tests/test_agents.py tests/test_sentiment_agent.py -v`
Expected: PASS (new reconcile tests + existing judge/agent/sentiment tests; the reconcile is a no-op for consistent opinions). Note: `RiskAgent` builds its opinion directly (not via `judge_evidence`), so its `score=risk_level` in [0,1] is NOT reconciled — correct, since risk score is not directional.

- [ ] **Step 5: Commit**

```bash
git add equity_research/agents/judge.py tests/test_judge.py
git commit -m "feat(agents): reconcile stance to score sign after parse"
```

---

## Task 7: Full suite + live verification

**Files:** none (verification), unless a live issue needs fixing.

- [ ] **Step 1: Full unit suite**

Run: `uv run pytest -m "not integration" -p no:warnings -q`
Expected: all PASS.

- [ ] **Step 2: Live smoke**

Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run pytest -m integration -q`
Expected: PASS (analyze + RAG live).

- [ ] **Step 3: Eyeball a real run**

Run: `DISABLE_PANDERA_IMPORT_WARNING=True uv run python -m equity_research.cli analyze AAPL`
Expected: report renders; any skipped agent shows a reason in parentheses; agent stances agree with their score signs; confidence still gated by risk.

- [ ] **Step 4: Commit any live fix**

```bash
git add -A && git commit -m "fix: hardening live-integration adjustments"
```
(Skip if nothing changed.)

---

## Self-Review (completed during authoring)

- **Coverage of decided scope:** timeouts+retry (Tasks 1-3), skip reasons (Task 4), scoring rubric + stance↔score (Tasks 5-6), few-shot example (Task 5). Explicitly-excluded items (friendly CLI errors, SEC throttle, system/user split, confidence guidance) are absent by design.
- **Placeholder scan:** none — every code step is complete.
- **Type consistency:** `resilient_call(fn, *, timeout, attempts, base_delay)` and `resilient(fn, net)` used consistently in cli wiring; `Config.net` keys (`data_timeout`/`data_attempts`/`data_base_delay`/`ollama_timeout`) match between config, `resilient`, and cli; `Verdict.skip_reasons: dict[str,str]` set by `aggregate(..., skip_reasons=...)` from `Orchestrator.run`, read by `render_markdown`; `reconcile_stance(opinion)` applied in `judge_evidence` before `ground`; risk agent deliberately bypasses reconcile (builds opinion directly). All align with existing signatures (`AgentOpinion`, `Evidence`, `PriceProvider`, `OllamaClient`, `EdgarProvider`).
