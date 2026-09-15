from __future__ import annotations

import json

from equity_research.eval.backtest import BacktestRecord, metrics_for_horizon

DISCLAIMER = "Backtest on a small sample; not investment advice. Past performance is not predictive."


def _horizon_dict(records: list[BacktestRecord], horizons: list[int]) -> dict:
    return {str(h): metrics_for_horizon(records, h) for h in horizons}


def render_backtest_json(records: list[BacktestRecord], horizons: list[int]) -> str:
    return json.dumps({
        "n_records": len(records),
        "horizons": _horizon_dict(records, horizons),
        "disclaimer": DISCLAIMER,
    }, indent=2)


def render_backtest_markdown(records: list[BacktestRecord], horizons: list[int]) -> str:
    lines = [f"# Backtest report ({len(records)} verdicts)", ""]
    for h in horizons:
        m = metrics_for_horizon(records, h)
        lines.append(f"## Horizon {h} trading days (n={m['n']})")
        ic = "n/a" if m["ic"] is None else f"{m['ic']:+.3f}"
        lines.append(f"- Information coefficient (score vs return): {ic}")
        lines.append("- Hit rate: " + (", ".join(f"{k} {v:.0%}" for k, v in m["hit_rate"].items()) or "n/a"))
        lines.append("- Mean forward return: " + (", ".join(f"{k} {v:+.2%}" for k, v in m["mean_return"].items()) or "n/a"))
        curve = m["long_short_curve"]
        final = curve[-1] if curve else 0.0
        lines.append(f"- Naive long-short cumulative return: {final:+.2%}")
        lines.append("")
    lines += ["---", f"> {DISCLAIMER}"]
    return "\n".join(lines)
