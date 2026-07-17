"""Append-only JSONL checkpointing so interrupted runs resume without rework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse one JSON object per line, skipping blanks; missing file is empty."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class JsonlCheckpoint:
    """Appends keyed records to a JSONL file and remembers which keys are done."""

    def __init__(self, path: Path, key_field: str) -> None:
        self._path = path
        self._key_field = key_field
        self._done = {str(record[key_field]) for record in load_jsonl(path)}
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def done(self) -> set[str]:
        return set(self._done)

    def __contains__(self, key: str) -> bool:
        return key in self._done

    def append(self, record: dict[str, Any]) -> None:
        with self._path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        self._done.add(str(record[self._key_field]))
