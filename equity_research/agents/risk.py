from __future__ import annotations

from datetime import date

from equity_research.agents.base import AgentOpinion
from equity_research.analytics.risk import annualized_volatility, beta, max_drawdown, risk_level
from equity_research.data.models import Evidence
from equity_research.data.prices import PriceProvider
from equity_research.llm.ollama_client import OllamaClient


class RiskAgent:
    name = "risk"

    def __init__(self, prices: PriceProvider, client: OllamaClient, benchmark: str, risk_cfg: dict):
        self.prices = prices
        self.client = client
        self.benchmark = benchmark
        self.risk_cfg = risk_cfg

    def gather(self, ticker: str, as_of: date) -> Evidence:
        hist, note = self.prices.history(ticker, as_of)
        close = hist["Close"]
        metrics = {
            "volatility": annualized_volatility(close),
            "max_drawdown": max_drawdown(close),
        }
        try:
            bench, _ = self.prices.history(self.benchmark, as_of)
            metrics["beta"] = beta(close, bench["Close"])
        except Exception:
            metrics["beta"] = float("nan")
        notes = [note] if note else []
        return Evidence(ticker=ticker, as_of=as_of, metrics=metrics, notes=notes)

    def judge(self, evidence: Evidence) -> AgentOpinion:
        level = risk_level(evidence.metrics, self.risk_cfg)
        narrative = self.client.generate_text(self._prompt(evidence, level))
        key_facts = [f"{k}: {v:.4f}" for k, v in evidence.metrics.items()] + [f"risk_level: {level:.2f}"]
        return AgentOpinion(
            agent=self.name, stance="neutral", score=level, confidence=level,
            rationale=narrative, key_facts=key_facts,
        )

    def _prompt(self, evidence: Evidence, level: float) -> str:
        metrics = "\n".join(f"- {k}: {v:.4f}" for k, v in evidence.metrics.items())
        return (
            f"You are a risk analyst for {evidence.ticker}. Given these price-risk metrics, "
            f"write 2-3 sentences describing the risk profile. Do not give a buy/sell opinion.\n"
            f"{metrics}\n- computed risk level (0=low, 1=high): {level:.2f}"
        )
