from __future__ import annotations

from pathlib import Path


class ParentStore:
    """File-backed store mapping a parent id to its full section text."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, parent_id: str) -> Path:
        return self.dir / f"{parent_id}.txt"

    def put(self, parent_id: str, text: str) -> None:
        self._path(parent_id).write_text(text)

    def get(self, parent_id: str) -> str:
        return self._path(parent_id).read_text()

    def has(self, parent_id: str) -> bool:
        return self._path(parent_id).exists()
