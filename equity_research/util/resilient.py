from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Callable, TypeVar

T = TypeVar("T")


def resilient_call(fn: Callable[[], T], *, timeout: float, attempts: int = 3, base_delay: float = 0.5) -> T:
    """Run fn with a per-attempt timeout and exponential-backoff retry.

    On timeout the underlying worker thread is abandoned (cannot be killed) and
    the next attempt starts fresh. Raises the last error after all attempts.
    """
    last: BaseException | None = None
    for i in range(attempts):
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
            last = TimeoutError(f"call timed out after {timeout}s")
            ex.shutdown(wait=False)
        except Exception as exc:  # transient network / library error
            last = exc
            ex.shutdown(wait=False)
        if i < attempts - 1 and base_delay:
            time.sleep(base_delay * (2 ** i))
    raise last if last is not None else RuntimeError("resilient_call failed")


def resilient(fn: Callable[..., T], net: dict) -> Callable[..., T]:
    """Wrap a callable so each invocation runs through resilient_call using net config."""

    def wrapped(*args, **kwargs) -> T:
        return resilient_call(lambda: fn(*args, **kwargs),
                              timeout=net["data_timeout"], attempts=net["data_attempts"],
                              base_delay=net["data_base_delay"])

    return wrapped
