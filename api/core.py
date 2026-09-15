from __future__ import annotations

from datetime import date, datetime

CONFIG_PATH = "config.yaml"


def health() -> dict:
    from equity_research.config import Config

    cfg = Config.load(CONFIG_PATH)
    reachable = True
    try:
        from ollama import Client

        Client(host=cfg.ollama_host, timeout=5).list()
    except Exception:
        reachable = False
    return {"ok": reachable, "model": cfg.model, "ollama_reachable": reachable}


def run_analyze(ticker: str, on_event):
    from equity_research.cli import analyze_ticker

    return analyze_ticker(ticker.upper(), date.today(), CONFIG_PATH, on_event=on_event)


def ingest_ticker(ticker: str) -> int:
    from equity_research.config import Config
    from equity_research.rag.adapters import fetch_news
    from equity_research.rag.chroma_store import ChromaVectorStore
    from equity_research.rag.ingest import ingest_news
    from equity_research.rag.store import NewsStore
    from equity_research.util.resilient import resilient

    cfg = Config.load(CONFIG_PATH)
    store = NewsStore(ChromaVectorStore(persist_dir=cfg.rag["chroma_dir"], embed_model=cfg.rag["embed_model"]))
    return ingest_news(resilient(fetch_news, cfg.net), store, ticker.upper())


def backtest_config() -> dict:
    from equity_research.config import Config

    cfg = Config.load(CONFIG_PATH)
    return {k: cfg.backtest[k] for k in ("universe", "dates", "horizons")}


def run_backtest_records(on_progress):
    from equity_research.cli import _full_close, build_backtest_verdict
    from equity_research.config import Config
    from equity_research.eval.backtest import run_backtest
    from equity_research.llm.cache import DiskCache
    from equity_research.llm.ollama_client import OllamaClient

    cfg = Config.load(CONFIG_PATH)
    from ollama import Client

    chat_fn = Client(host=cfg.ollama_host, timeout=cfg.net["ollama_timeout"]).chat
    client = OllamaClient(model=cfg.model, cache=DiskCache(cfg.cache_dir),
                          seed=cfg.seed, temperature=cfg.temperature, chat_fn=chat_fn)
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in cfg.backtest["dates"]]
    run_verdict = build_backtest_verdict(cfg, client)

    def run_verdict_progress(ticker, as_of):
        on_progress({"ticker": ticker, "as_of": str(as_of)})
        return run_verdict(ticker, as_of)

    records = run_backtest(cfg.backtest["universe"], dates, cfg.backtest["horizons"],
                           run_verdict_progress, _full_close)
    return records, cfg.backtest["horizons"]
