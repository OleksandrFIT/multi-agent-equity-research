from __future__ import annotations

import hashlib
import json
from typing import Callable

from equity_research.llm.cache import DiskCache


class OllamaClient:
    def __init__(
        self,
        model: str,
        cache: DiskCache,
        seed: int,
        temperature: float,
        chat_fn: Callable[..., dict],
        max_retries: int = 3,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.model = model
        self.cache = cache
        self.seed = seed
        self.temperature = temperature
        self.chat_fn = chat_fn
        self.max_retries = max_retries

    def _cache_key(self, prompt: str, fmt: str) -> str:
        payload = json.dumps([self.model, self.seed, self.temperature, fmt, prompt])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _call(self, prompt: str, fmt) -> str:
        options = {"temperature": self.temperature, "seed": self.seed}
        resp = self.chat_fn(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=fmt,
            options=options,
        )
        return resp["message"]["content"]

    def generate_json(self, prompt: str, schema: dict, *, seed=None, temperature=None) -> dict:
        key = self._cache_key(prompt, json.dumps(schema, sort_keys=True))

        def produce() -> dict:
            last_err = ""
            attempt_prompt = prompt
            for _ in range(self.max_retries):
                raw = self._call(attempt_prompt, schema)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    last_err = str(exc)
                    attempt_prompt = (
                        f"{prompt}\n\nYour previous output was not valid JSON "
                        f"({last_err}). Return ONLY valid JSON matching the schema."
                    )
            raise ValueError(f"invalid JSON after {self.max_retries} retries: {last_err}")

        return self.cache.get_or_set(key, produce)

    def generate_text(self, prompt: str, *, seed=None, temperature=None) -> str:
        key = self._cache_key(prompt, "text")
        return self.cache.get_or_set(key, lambda: self._call(prompt, ""))
