from datetime import date

import pandas as pd

from equity_research.agents.risk import RiskAgent
from equity_research.data.prices import PriceProvider
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient

RISK_CFG = {"gate_strength": 0.5, "caution_threshold": 0.6,
            "vol_low": 0.15, "vol_high": 0.60, "beta_high": 1.5, "drawdown_high": 0.40}


def _prices(closes):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": [float(c) for c in closes]}, index=idx)


def _client(tmp_path):
    return OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                        chat_fn=lambda **k: {"message": {"content": "Risk narrative."}})


def _provider(closes):
    frame = _prices(closes)
    return PriceProvider(fetch_yfinance=lambda t: frame, fetch_stooq=lambda t: frame)


def test_gather_computes_risk_metrics(tmp_path):
    closes = [100, 110, 99, 120, 90, 130, 100, 140]
    agent = RiskAgent(prices=_provider(closes), client=_client(tmp_path), benchmark="SPY", risk_cfg=RISK_CFG)
    ev = agent.gather("AAPL", as_of=date(2025, 3, 1))
    assert "volatility" in ev.metrics
    assert "max_drawdown" in ev.metrics
    assert "beta" in ev.metrics


def test_judge_score_is_risk_level_and_neutral(tmp_path):
    closes = [100, 160, 80, 200, 60, 220, 70, 240]  # very volatile -> high risk
    agent = RiskAgent(prices=_provider(closes), client=_client(tmp_path), benchmark="SPY", risk_cfg=RISK_CFG)
    op = agent.judge(agent.gather("AAPL", as_of=date(2025, 3, 1)))
    assert op.agent == "risk"
    assert op.stance == "neutral"
    assert 0.0 <= op.score <= 1.0
    assert op.rationale == "Risk narrative."


def test_beta_nan_is_tolerated(tmp_path):
    flat = _prices([100, 100, 100, 100])
    provider = PriceProvider(fetch_yfinance=lambda t: flat, fetch_stooq=lambda t: flat)
    agent = RiskAgent(prices=provider, client=_client(tmp_path), benchmark="SPY", risk_cfg=RISK_CFG)
    ev = agent.gather("AAPL", as_of=date(2025, 3, 1))
    op = agent.judge(ev)
    assert op.score == 0.0  # flat prices: vol 0, drawdown 0, beta nan -> level 0
