import math

from equity_research.analytics.fundamentals import compute_fundamental_metrics


FACTS = {
    "net_income": 96995000000.0,
    "revenue": 383285000000.0,
    "revenue_prev": 394328000000.0,
    "equity": 62146000000.0,
    "total_debt": 111088000000.0,
    "eps_ttm": 6.13,
}


def test_pe_from_price_and_eps():
    m = compute_fundamental_metrics(FACTS, price=196.0)
    assert round(m["pe"], 2) == round(196.0 / 6.13, 2)


def test_roe():
    m = compute_fundamental_metrics(FACTS, price=196.0)
    assert round(m["roe"], 4) == round(96995000000.0 / 62146000000.0, 4)


def test_debt_to_equity():
    m = compute_fundamental_metrics(FACTS, price=196.0)
    assert round(m["debt_to_equity"], 4) == round(111088000000.0 / 62146000000.0, 4)


def test_revenue_growth_negative():
    m = compute_fundamental_metrics(FACTS, price=196.0)
    assert m["revenue_growth"] < 0  # revenue fell vs prev


def test_missing_eps_gives_nan_pe():
    facts = dict(FACTS)
    facts["eps_ttm"] = 0.0
    m = compute_fundamental_metrics(facts, price=196.0)
    assert math.isnan(m["pe"])


def test_richer_margins_and_ratio():
    facts = dict(FACTS, operating_income=114000000000.0, free_cash_flow=99000000000.0, current_ratio=0.99)
    m = compute_fundamental_metrics(facts, price=196.0)
    assert round(m["operating_margin"], 4) == round(114000000000.0 / 383285000000.0, 4)
    assert round(m["net_margin"], 4) == round(96995000000.0 / 383285000000.0, 4)
    assert round(m["fcf_margin"], 4) == round(99000000000.0 / 383285000000.0, 4)
    assert m["current_ratio"] == 0.99


def test_richer_metrics_nan_when_missing():
    m = compute_fundamental_metrics(FACTS, price=196.0)  # FACTS has no op_income/fcf/current_ratio
    assert math.isnan(m["operating_margin"])
    assert math.isnan(m["fcf_margin"])
    assert math.isnan(m["current_ratio"])
