from equity_research.eval.metrics import (
    hit_rate_by_class,
    information_coefficient,
    long_short_curve,
    mean_return_by_class,
)


def _rec(verdict, score, fwd):
    return {"verdict": verdict, "score": score, "fwd_return": fwd}


def test_hit_rate_by_class():
    recs = [_rec("buy", 0.5, 0.1), _rec("buy", 0.4, -0.1), _rec("sell", -0.5, -0.2)]
    hr = hit_rate_by_class(recs)
    assert hr["buy"] == 0.5
    assert hr["sell"] == 1.0


def test_mean_return_by_class():
    recs = [_rec("buy", 0.5, 0.1), _rec("buy", 0.4, -0.1), _rec("hold", 0.0, 0.05)]
    mr = mean_return_by_class(recs)
    assert abs(mr["buy"] - 0.0) < 1e-9
    assert abs(mr["hold"] - 0.05) < 1e-9
    assert "sell" not in mr


def test_information_coefficient_monotonic_is_one():
    recs = [_rec("x", 0.1, 1.0), _rec("x", 0.2, 2.0), _rec("x", 0.3, 3.0)]
    assert abs(information_coefficient(recs) - 1.0) < 1e-9


def test_long_short_curve():
    recs = [_rec("buy", 0.5, 0.1), _rec("sell", -0.5, 0.2), _rec("hold", 0.0, 0.9)]
    curve = long_short_curve(recs)
    assert [round(c, 4) for c in curve] == [0.1, -0.1, -0.1]
