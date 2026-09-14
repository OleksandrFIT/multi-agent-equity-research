from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    model: str
    temperature: float
    seed: int
    cache_dir: str
    edgar_user_agent: str
    weights: dict[str, float]
    ollama_host: str = "http://localhost:11434"

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)

    def normalized_weights(self, available: list[str]) -> dict[str, float]:
        subset = {k: self.weights[k] for k in available if k in self.weights}
        total = sum(subset.values())
        if total == 0:
            raise ValueError("weights sum to zero for available agents")
        return {k: v / total for k, v in subset.items()}
