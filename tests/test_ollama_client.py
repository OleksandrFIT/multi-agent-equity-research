from pathlib import Path

import pytest

from equity_research.llm.cache import DiskCache
from equity_research.llm.ollama_client import OllamaClient

SCHEMA = {
    "type": "object",
    "properties": {"stance": {"type": "string"}, "score": {"type": "number"}},
    "required": ["stance", "score"],
}


def make_chat(responses):
    seq = list(responses)
    calls = []

    def chat_fn(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": seq.pop(0)}}

    chat_fn.calls = calls
    return chat_fn


def test_generate_json_parses_valid(tmp_path: Path):
    client = OllamaClient(
        model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
        chat_fn=make_chat(['{"stance": "bullish", "score": 0.5}']),
    )
    out = client.generate_json("prompt", SCHEMA)
    assert out == {"stance": "bullish", "score": 0.5}


def test_generate_json_retries_then_succeeds(tmp_path: Path):
    chat = make_chat(["not json", '{"stance": "neutral", "score": 0.0}'])
    client = OllamaClient(
        model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
        chat_fn=chat,
        max_retries=2,
    )
    out = client.generate_json("prompt", SCHEMA)
    assert out["stance"] == "neutral"
    assert "not valid JSON" in chat.calls[1]["messages"][0]["content"]


def test_generate_json_raises_after_retries(tmp_path: Path):
    client = OllamaClient(
        model="m", cache=DiskCache(tmp_path), seed=1, temperature=0.0,
        chat_fn=make_chat(["nope", "still nope"]),
        max_retries=2,
    )
    with pytest.raises(ValueError):
        client.generate_json("prompt", SCHEMA)


def test_generate_json_cached(tmp_path: Path):
    calls = {"n": 0}

    def chat_fn(**kwargs):
        calls["n"] += 1
        return {"message": {"content": '{"stance": "bullish", "score": 0.5}'}}

    cache = DiskCache(tmp_path)
    client = OllamaClient(model="m", cache=cache, seed=1, temperature=0.0, chat_fn=chat_fn)
    client.generate_json("same prompt", SCHEMA)
    client.generate_json("same prompt", SCHEMA)
    assert calls["n"] == 1  # second call served from cache
