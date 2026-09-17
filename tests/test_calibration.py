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
