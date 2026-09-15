import json
from datetime import datetime
from pathlib import Path

import pytest

FIX = Path(__file__).parent.parent / "fixtures" / "golden" / "verdicts_golden.json"


@pytest.mark.integration
def test_verdicts_golden_regression():
    """Real analyze pipeline over golden cases. Requires Ollama.

    Baselines are point-in-time; re-capture (new dated fixture) if data has moved.
    Run: uv run pytest -m integration tests/integration/test_verdicts_golden.py
    """
    from equity_research.cli import analyze_ticker

    cases = json.loads(FIX.read_text())
    for case in cases:
        as_of = datetime.strptime(case["as_of"], "%Y-%m-%d").date()
        v = analyze_ticker(case["ticker"], as_of, "config.yaml")
        assert v.status == "ok", f"{case['ticker']} status {v.status}"
        assert v.verdict == case["expected_verdict"], f"{case['ticker']} verdict {v.verdict}"
        assert -1.0 <= v.score <= 1.0 and 0.0 <= v.confidence <= 1.0
        stances = {o.agent: o.stance for o in v.opinions}
        for agent, expected_stance in case["expected_stances"].items():
            assert stances.get(agent) == expected_stance, f"{case['ticker']} {agent} {stances.get(agent)}"


@pytest.mark.integration
def test_verdicts_reproducible():
    """Same input -> same score/verdict (seed + temp 0 + cache)."""
    from equity_research.cli import analyze_ticker

    cases = json.loads(FIX.read_text())
    c = cases[0]
    as_of = datetime.strptime(c["as_of"], "%Y-%m-%d").date()
    a = analyze_ticker(c["ticker"], as_of, "config.yaml")
    b = analyze_ticker(c["ticker"], as_of, "config.yaml")
    assert a.verdict == b.verdict and abs(a.score - b.score) < 1e-9
