from datetime import date

import pytest


@pytest.mark.integration
def test_analyze_aapl_live():
    """Requires Ollama running with the configured model, plus network for edgar/yfinance.
    Run explicitly: uv run pytest -m integration
    """
    from equity_research.cli import analyze_ticker

    verdict = analyze_ticker("AAPL", date.today(), "config.yaml")
    assert verdict.ticker == "AAPL"
    assert verdict.verdict in {"buy", "hold", "sell"}
    assert len(verdict.opinions) >= 1
