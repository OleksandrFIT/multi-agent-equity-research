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
