from datetime import date
from pathlib import Path

import pandas as pd

from equity_research.agents.fundamentals import FundamentalsAgent
from equity_research.agents.technical import TechnicalAgent
from equity_research.data.prices import PriceProvider
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient


def _prices(closes):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=idx)


def _client(tmp_path):
    content = '{"stance":"neutral","score":0.0,"confidence":0.5,"rationale":"r","key_facts":[]}'
    return OllamaClient(
        model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
        chat_fn=lambda **k: {"message": {"content": content}},
    )


def test_technical_agent_gathers_indicator_metrics(tmp_path):
    closes = [float(x) for x in range(100, 360)]
    provider = PriceProvider(fetch_yfinance=lambda t: _prices(closes), fetch_stooq=lambda t: _prices(closes))
    agent = TechnicalAgent(prices=provider, client=_client(tmp_path))
    ev = agent.gather("AAPL", as_of=date(2025, 9, 1))
    assert "rsi14" in ev.metrics and "sma50" in ev.metrics


def test_fundamentals_agent_uses_facts_and_price(tmp_path):
    closes = [float(x) for x in range(100, 360)]
    provider = PriceProvider(fetch_yfinance=lambda t: _prices(closes), fetch_stooq=lambda t: _prices(closes))
    facts = {"net_income": 100.0, "revenue": 1000.0, "revenue_prev": 900.0,
             "equity": 500.0, "total_debt": 250.0, "eps_ttm": 5.0}
    agent = FundamentalsAgent(
        facts_source=type("F", (), {"company_facts": staticmethod(lambda t: facts)})(),
        prices=provider, client=_client(tmp_path),
    )
    ev = agent.gather("AAPL", as_of=date(2025, 9, 1))
    assert "pe" in ev.metrics and "roe" in ev.metrics


def test_agent_judge_returns_opinion(tmp_path):
    closes = [float(x) for x in range(100, 360)]
    provider = PriceProvider(fetch_yfinance=lambda t: _prices(closes), fetch_stooq=lambda t: _prices(closes))
    agent = TechnicalAgent(prices=provider, client=_client(tmp_path))
    ev = agent.gather("AAPL", as_of=date(2025, 9, 1))
    op = agent.judge(ev)
    assert op.agent == "technical"
