from __future__ import annotations

from datetime import date

import typer

from equity_research.agents.fundamentals import FundamentalsAgent
from equity_research.agents.technical import TechnicalAgent
from equity_research.config import Config
from equity_research.data.adapters import fetch_stooq, fetch_yfinance
from equity_research.data.edgar import EdgarProvider
from equity_research.data.prices import PriceProvider
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient
from equity_research.orchestration.aggregator import Aggregator, Verdict
from equity_research.orchestration.orchestrator import Orchestrator
from equity_research.reporting.report import render_markdown

app = typer.Typer(help="Multi-agent equity research")


def analyze_ticker(ticker: str, as_of: date, cfg_path: str) -> Verdict:
    cfg = Config.load(cfg_path)
    from ollama import Client

    chat_fn = Client(host=cfg.ollama_host).chat
    client = OllamaClient(model=cfg.model, cache=DiskCache(cfg.cache_dir),
                          seed=cfg.seed, temperature=cfg.temperature, chat_fn=chat_fn)
    prices = PriceProvider(fetch_yfinance=fetch_yfinance, fetch_stooq=fetch_stooq)
    edgar = EdgarProvider(user_agent=cfg.edgar_user_agent)
    agents = [
        FundamentalsAgent(facts_source=edgar, prices=prices, client=client),
        TechnicalAgent(prices=prices, client=client),
    ]
    orch = Orchestrator(agents=agents, aggregator=Aggregator(cfg, client))
    return orch.run(ticker, as_of)


@app.command()
def analyze(ticker: str, config: str = "config.yaml"):
    verdict = analyze_ticker(ticker.upper(), date.today(), config)
    typer.echo(render_markdown(verdict))


if __name__ == "__main__":
    app()
