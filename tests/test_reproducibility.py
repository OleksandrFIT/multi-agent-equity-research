from datetime import date
from pathlib import Path

import pandas as pd

from equity_research.agents.technical import TechnicalAgent
from equity_research.config import Config
from equity_research.data.prices import PriceProvider
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient
from equity_research.orchestration.aggregator import Aggregator
from equity_research.orchestration.orchestrator import Orchestrator


def _prices():
    closes = [float(x) for x in range(100, 360)]
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=idx)


def test_same_input_same_score(tmp_path: Path):
    cfg = Config(model="m", temperature=0.0, seed=1, cache_dir=str(tmp_path / "c"),
                 edgar_user_agent="x x@x.com",
                 weights={"fundamentals": 0.4, "technical": 0.25, "sentiment": 0.15, "risk": 0.2})
    content = '{"stance":"bullish","score":0.3,"confidence":0.6,"rationale":"r","key_facts":[]}'

    def build():
        client = OllamaClient(model="m", cache=DiskCache(cfg.cache_dir), seed=1, temperature=0.0,
                              chat_fn=lambda **k: {"message": {"content": content}})
        prices = PriceProvider(fetch_yfinance=lambda t: _prices(), fetch_stooq=lambda t: _prices())
        agents = [TechnicalAgent(prices=prices, client=client)]
        return Orchestrator(agents=agents, aggregator=Aggregator(cfg, client))

    v1 = build().run("AAPL", date(2025, 9, 1))
    v2 = build().run("AAPL", date(2025, 9, 1))
    assert abs(v1.score - v2.score) < 1e-9
    assert v1.verdict == v2.verdict
