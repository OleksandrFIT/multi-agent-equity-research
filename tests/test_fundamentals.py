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
