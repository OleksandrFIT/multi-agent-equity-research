from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


class DiskCache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.dir / f"{digest}.json"

    def get_or_set(self, key: str, producer: Callable[[], T]) -> T:
        path = self._path(key)
        if path.exists():
            return json.loads(path.read_text())
        value = producer()
        path.write_text(json.dumps(value))
        return value
