from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer

from equity_research.agents.fundamentals import FundamentalsAgent
from equity_research.agents.risk import RiskAgent
from equity_research.agents.sentiment import SentimentAgent
from equity_research.agents.technical import TechnicalAgent
from equity_research.config import Config
from equity_research.data.adapters import fetch_stooq, fetch_yfinance, fetch_yfinance_long
from equity_research.data.company import company_name
from equity_research.data.edgar import EdgarProvider
from equity_research.data.prices import PriceProvider, PriceValidationError
from equity_research.eval.backtest import run_backtest
from equity_research.eval.report import render_backtest_json, render_backtest_markdown
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient
from equity_research.orchestration.aggregator import Aggregator, Verdict
from equity_research.orchestration.orchestrator import Orchestrator
from equity_research.rag.adapters import fetch_news
from equity_research.rag.chroma_store import ChromaVectorStore
from equity_research.rag.filing_retrieve import FilingRetriever
from equity_research.rag.filing_store import FilingStore
from equity_research.rag.filings import fetch_filing_sections
from equity_research.rag.ingest import ingest_news
from equity_research.rag.parent_store import ParentStore
from equity_research.rag.reranker import default_scorer
from equity_research.rag.retrieve import NewsRetriever
from equity_research.rag.store import NewsStore
from equity_research.reporting.report import render_markdown
from equity_research.util.resilient import resilient

app = typer.Typer(help="Multi-agent equity research")


def _build_filing_pieces(cfg, vs):
    filing_store = FilingStore(vs, ParentStore(cfg.rag["parent_dir"]),
                               parent_max_chars=cfg.rag["filing_parent_max_chars"])
    retriever = FilingRetriever(filing_store, scorer=default_scorer(cfg.rag["rerank_model"]))

    def ingest_fn(ticker):
        sections = resilient(fetch_filing_sections, cfg.net)(ticker, date.today(), cfg.edgar_user_agent)
        filing_store.upsert(sections)

    return retriever, ingest_fn


def _unknown_ticker_verdict(prices, ticker: str, as_of) -> "Verdict | None":
    """Return an unknown-ticker Verdict if there is no price history, else None.

    A ticker with no price data (e.g. a typo like APPL) cannot be valued; we
    short-circuit before running any agent so we don't surface generic,
    misleading news for a symbol that does not exist.
    """
    try:
        prices.history(ticker, as_of)
    except PriceValidationError:
        return Verdict(
            ticker=ticker, as_of=as_of, verdict="hold", score=0.0, confidence=0.0,
            status="unknown_ticker",
            narrative=f"No price data for {ticker}; it may be an unknown or delisted ticker.",
            opinions=[],
        )
    except Exception:
        return None  # transient/other error: let agents run and degrade to insufficient_data
    return None


def analyze_ticker(ticker: str, as_of: date, cfg_path: str, on_event=None) -> Verdict:
    cfg = Config.load(cfg_path)
    from ollama import Client

    chat_fn = Client(host=cfg.ollama_host, timeout=cfg.net["ollama_timeout"]).chat
    client = OllamaClient(model=cfg.model, cache=DiskCache(cfg.cache_dir),
                          seed=cfg.seed, temperature=cfg.temperature, chat_fn=chat_fn)
    prices = PriceProvider(fetch_yfinance=resilient(fetch_yfinance, cfg.net),
                           fetch_stooq=resilient(fetch_stooq, cfg.net))
    edgar = EdgarProvider(user_agent=cfg.edgar_user_agent)
    edgar.company_facts = resilient(edgar.company_facts, cfg.net)
    vs = ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"], embed_model=cfg.rag["embed_model"], net=cfg.net)
    news_store = NewsStore(vs)
    retriever = NewsRetriever(news_store, scorer=default_scorer(cfg.rag["rerank_model"]))
    sentiment = SentimentAgent(
        retriever=retriever,
        ingest_fn=lambda t: ingest_news(resilient(fetch_news, cfg.net), news_store, t),
        client=client, k=cfg.rag["retrieve_k"], candidate_k=cfg.rag["candidate_k"],
        name_fn=company_name,
    )
    filing_retriever, filing_ingest = _build_filing_pieces(cfg, vs)
    agents = [
        FundamentalsAgent(facts_source=edgar, prices=prices, client=client,
                          filing_retriever=filing_retriever, filing_ingest_fn=filing_ingest,
                          k=cfg.rag["filing_retrieve_k"], candidate_k=cfg.rag["filing_candidate_k"]),
        TechnicalAgent(prices=prices, client=client),
        sentiment,
        RiskAgent(prices=prices, client=client, benchmark=cfg.benchmark, risk_cfg=cfg.risk),
    ]
    unknown = _unknown_ticker_verdict(prices, ticker, as_of)
    if unknown is not None:
        return unknown
    orch = Orchestrator(agents=agents, aggregator=Aggregator(cfg, client))
    return orch.run(ticker, as_of, on_event=on_event)


@app.command()
def analyze(ticker: str, config: str = "config.yaml"):
    verdict = analyze_ticker(ticker.upper(), date.today(), config)
    typer.echo(render_markdown(verdict))


@app.command()
def ingest(ticker: str, config: str = "config.yaml"):
    cfg = Config.load(config)
    vs = ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"], embed_model=cfg.rag["embed_model"], net=cfg.net)
    n = ingest_news(resilient(fetch_news, cfg.net), NewsStore(vs), ticker.upper())
    sections = resilient(fetch_filing_sections, cfg.net)(ticker.upper(), date.today(), cfg.edgar_user_agent)
    FilingStore(vs, ParentStore(cfg.rag["parent_dir"]),
                parent_max_chars=cfg.rag["filing_parent_max_chars"]).upsert(sections)
    typer.echo(f"Ingested {n} news items and {len(sections)} filing sections for {ticker.upper()}")


def build_backtest_verdict(cfg: Config, client):
    prices = PriceProvider(fetch_yfinance=resilient(fetch_yfinance_long, cfg.net),
                           fetch_stooq=resilient(fetch_stooq, cfg.net))
    edgar = EdgarProvider(user_agent=cfg.edgar_user_agent)
    edgar.company_facts = resilient(edgar.company_facts, cfg.net)
    vs = ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"], embed_model=cfg.rag["embed_model"], net=cfg.net)
    filing_retriever, filing_ingest = _build_filing_pieces(cfg, vs)

    def run_verdict(ticker, as_of):
        agents = [
            FundamentalsAgent(facts_source=edgar, prices=prices, client=client,
                              filing_retriever=filing_retriever, filing_ingest_fn=filing_ingest,
                              k=cfg.rag["filing_retrieve_k"], candidate_k=cfg.rag["filing_candidate_k"]),
            TechnicalAgent(prices=prices, client=client),
            RiskAgent(prices=prices, client=client, benchmark=cfg.benchmark, risk_cfg=cfg.risk),
        ]
        return Orchestrator(agents=agents, aggregator=Aggregator(cfg, client)).run(ticker, as_of)

    return run_verdict


def _full_close(ticker: str):
    df = fetch_yfinance_long(ticker)
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df["Close"]


@app.command()
def backtest(config: str = "config.yaml"):
    cfg = Config.load(config)
    from ollama import Client

    chat_fn = Client(host=cfg.ollama_host, timeout=cfg.net["ollama_timeout"]).chat
    client = OllamaClient(model=cfg.model, cache=DiskCache(cfg.cache_dir),
                          seed=cfg.seed, temperature=cfg.temperature, chat_fn=chat_fn)
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in cfg.backtest["dates"]]
    records = run_backtest(cfg.backtest["universe"], dates, cfg.backtest["horizons"],
                           build_backtest_verdict(cfg, client), _full_close)
    md = render_backtest_markdown(records, cfg.backtest["horizons"])
    Path(cfg.backtest["report_path"]).write_text(md)
    Path(cfg.backtest["report_path"].replace(".md", ".json")).write_text(
        render_backtest_json(records, cfg.backtest["horizons"]))
    typer.echo(md)


if __name__ == "__main__":
    app()
