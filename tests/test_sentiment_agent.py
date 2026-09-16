from datetime import date

from equity_research.agents.sentiment import SentimentAgent
from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient


class FakeRetriever:
    def __init__(self, texts):
        self._texts = texts

    def retrieve(self, ticker, query, as_of, k, candidate_k):
        return self._texts


def _client(tmp_path, content):
    return OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
                        chat_fn=lambda **k: {"message": {"content": content}})


def test_gather_puts_news_into_context(tmp_path):
    retr = FakeRetriever(["Apple beats estimates.", "Supply chain concerns."])
    ingested = {}
    agent = SentimentAgent(retriever=retr, ingest_fn=lambda t: ingested.setdefault(t, True),
                           client=_client(tmp_path, "x"))
    ev = agent.gather("AAPL", as_of=date(2026, 9, 15))
    assert "Apple beats estimates." in ev.context
    assert ingested["AAPL"] is True  # fresh news ingested before retrieval


def test_judge_returns_sentiment_opinion(tmp_path):
    retr = FakeRetriever(["Apple beats estimates."])
    content = ('{"reasoning":"good news","stance":"bullish","score":0.6,"confidence":0.7,'
              '"rationale":"good news","key_facts":["Apple beats estimates"]}')
    agent = SentimentAgent(retriever=retr, ingest_fn=lambda t: None, client=_client(tmp_path, content))
    op = agent.judge(agent.gather("AAPL", as_of=date(2026, 9, 15)))
    assert op.agent == "sentiment"
    assert op.stance == "bullish"
    assert "Apple beats estimates" in op.key_facts  # grounded against context


def test_no_news_degrades_without_calling_llm(tmp_path):
    calls = {"n": 0}

    def chat(**k):
        calls["n"] += 1
        return {"message": {"content": '{"stance":"bullish","score":0.9,"confidence":0.9,"rationale":"r","key_facts":[]}'}}

    from equity_research.llm.ollama_client import OllamaClient
    from equity_research.llm.cache import DiskCache
    client = OllamaClient(model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0, chat_fn=chat)
    agent = SentimentAgent(retriever=FakeRetriever([]), ingest_fn=lambda t: None, client=client)
    op = agent.judge(agent.gather("AAPL", as_of=date(2026, 9, 15)))
    assert op.stance == "neutral"
    assert op.confidence == 0.0
    assert calls["n"] == 0  # LLM not called when there is no news


def test_sentiment_filters_news_by_company_relevance():
    from datetime import date

    from equity_research.agents.sentiment import SentimentAgent

    class FakeRetriever:
        def retrieve(self, ticker, query, as_of, k, candidate_k):
            return ["Apple unveils new iPhone", "Generic market wrap: indexes edge lower"]

    agent = SentimentAgent(retriever=FakeRetriever(), ingest_fn=lambda t: None,
                           client=None, name_fn=lambda t: "Apple Inc")
    ev = agent.gather("AAPL", date(2026, 9, 15))
    assert ev.context == ["Apple unveils new iPhone"]  # generic wrap dropped


def test_sentiment_keeps_all_when_no_company_name():
    from datetime import date

    from equity_research.agents.sentiment import SentimentAgent

    class FakeRetriever:
        def retrieve(self, ticker, query, as_of, k, candidate_k):
            return ["Generic market wrap", "Another headline"]

    agent = SentimentAgent(retriever=FakeRetriever(), ingest_fn=lambda t: None,
                           client=None, name_fn=lambda t: None)
    ev = agent.gather("AAPL", date(2026, 9, 15))
    assert ev.context == ["Generic market wrap", "Another headline"]  # no name -> no filtering
