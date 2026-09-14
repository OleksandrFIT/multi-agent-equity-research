import json
import math
from pathlib import Path

from equity_research.analytics.fundamentals import compute_fundamental_metrics

FIX = Path(__file__).parent / "fixtures"


def test_aapl_metrics_match_hand_verified():
    data = json.loads((FIX / "aapl_facts.json").read_text())
    metrics = compute_fundamental_metrics(data["facts"], price=data["price"])
    for key, expected in data["expected"].items():
        assert math.isclose(metrics[key], expected, rel_tol=1e-3), key


def test_edge_cases():
    cases = json.loads((FIX / "edge_cases.json").read_text())
    for case in cases:
        m = compute_fundamental_metrics(case["facts"], price=case["price"])
        for key, expected in case["expected"].items():
            if expected == "nan":
                assert math.isnan(m[key]), f"{case['name']}:{key}"
            else:
                assert math.isclose(m[key], expected, rel_tol=1e-3), f"{case['name']}:{key}"
