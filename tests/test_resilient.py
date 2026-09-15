import time

import pytest

from equity_research.util.resilient import resilient, resilient_call


def test_succeeds_first_try():
    assert resilient_call(lambda: 42, timeout=1, attempts=3, base_delay=0) == 42


def test_retries_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert resilient_call(flaky, timeout=1, attempts=3, base_delay=0) == "ok"
    assert calls["n"] == 3


def test_raises_after_attempts():
    def always_fail():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        resilient_call(always_fail, timeout=1, attempts=2, base_delay=0)


def test_times_out():
    def slow():
        time.sleep(0.3)
        return "late"

    with pytest.raises(TimeoutError):
        resilient_call(slow, timeout=0.05, attempts=1, base_delay=0)


def test_resilient_wraps_callable():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("blip")
        return x * 2

    wrapped = resilient(flaky, {"data_timeout": 1, "data_attempts": 3, "data_base_delay": 0})
    assert wrapped(5) == 10
    assert calls["n"] == 2
