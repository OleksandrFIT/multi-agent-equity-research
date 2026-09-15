from __future__ import annotations

from datetime import date

import typer

from equity_research.agents.fundamentals import FundamentalsAgent
from equity_research.agents.risk import RiskAgent
from equity_research.agents.sentiment import SentimentAgent
from equity_research.agents.technical import TechnicalAgent
from equity_research.config import Config
from equity_research.data.adapters import fetch_stooq, fetch_yfinance
from equity_research.data.edgar import EdgarProvider
from equity_research.data.prices import PriceProvider
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient
from equity_research.orchestration.aggregator import Aggregator, Verdict
from equity_research.orchestration.orchestrator import Orchestrator
from equity_research.rag.adapters import fetch_news
from equity_research.rag.chroma_store import ChromaVectorStore
from equity_research.rag.ingest import ingest_news
from equity_research.rag.reranker import default_scorer
from equity_research.rag.retrieve import NewsRetriever
from equity_research.rag.store import NewsStore
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
    news_store = NewsStore(ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"],
                                             embed_model=cfg.rag["embed_model"]))
    retriever = NewsRetriever(news_store, scorer=default_scorer(cfg.rag["rerank_model"]))
    sentiment = SentimentAgent(
        retriever=retriever,
        ingest_fn=lambda t: ingest_news(fetch_news, news_store, t),
        client=client, k=cfg.rag["retrieve_k"], candidate_k=cfg.rag["candidate_k"],
    )
    agents = [
        FundamentalsAgent(facts_source=edgar, prices=prices, client=client),
        TechnicalAgent(prices=prices, client=client),
        sentiment,
        RiskAgent(prices=prices, client=client, benchmark=cfg.benchmark, risk_cfg=cfg.risk),
    ]
    orch = Orchestrator(agents=agents, aggregator=Aggregator(cfg, client))
    return orch.run(ticker, as_of)


@app.command()
def analyze(ticker: str, config: str = "config.yaml"):
    verdict = analyze_ticker(ticker.upper(), date.today(), config)
    typer.echo(render_markdown(verdict))


@app.command()
def ingest(ticker: str, config: str = "config.yaml"):
    cfg = Config.load(config)
    store = NewsStore(ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"], embed_model=cfg.rag["embed_model"]))
    n = ingest_news(fetch_news, store, ticker.upper())
    typer.echo(f"Ingested {n} news items for {ticker.upper()}")


if __name__ == "__main__":
    app()
