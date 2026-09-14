from pathlib import Path

from equity_research.llm.cache import DiskCache


def test_get_or_set_persists_and_reuses(tmp_path: Path):
    cache = DiskCache(tmp_path)
    calls = {"n": 0}

    def produce():
        calls["n"] += 1
        return {"value": 42}

    first = cache.get_or_set("k1", produce)
    second = cache.get_or_set("k1", produce)
    assert first == second == {"value": 42}
    assert calls["n"] == 1  # producer ran once, second read from disk


def test_different_keys_isolated(tmp_path: Path):
    cache = DiskCache(tmp_path)
    assert cache.get_or_set("a", lambda: 1) == 1
    assert cache.get_or_set("b", lambda: 2) == 2
