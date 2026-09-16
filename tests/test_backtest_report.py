import json
from datetime import date

from equity_research.eval.backtest import BacktestRecord
from equity_research.eval.report import render_backtest_json, render_backtest_markdown


def _recs():
    return [
        BacktestRecord("AAA", date(2023, 1, 5), "buy", 0.5, {21: 0.1, 63: 0.2}),
        BacktestRecord("BBB", date(2023, 1, 5), "sell", -0.4, {21: -0.05, 63: None}),
    ]


def test_markdown_has_horizons_and_metrics():
    md = render_backtest_markdown(_recs(), horizons=[21, 63])
    assert "21" in md and "63" in md
    assert "hit rate" in md.lower()
    assert "information coefficient" in md.lower()


def test_json_roundtrips_metrics():
    data = json.loads(render_backtest_json(_recs(), horizons=[21, 63]))
    assert "21" in data["horizons"]
    assert data["horizons"]["21"]["n"] == 2
    assert data["n_records"] == 2


def test_json_includes_per_record_rows():
    data = json.loads(render_backtest_json(_recs(), horizons=[21, 63]))
    assert len(data["records"]) == 2
    row = data["records"][0]
    assert row["ticker"] == "AAA" and row["verdict"] == "buy" and abs(row["score"] - 0.5) < 1e-9
    assert row["fwd_returns"]["21"] == 0.1 and row["fwd_returns"]["63"] == 0.2
    assert data["records"][1]["fwd_returns"]["63"] is None


def test_markdown_has_records_table():
    md = render_backtest_markdown(_recs(), horizons=[21, 63])
    assert "| Ticker |" in md
    assert "AAA" in md and "BBB" in md
