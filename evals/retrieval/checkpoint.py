"""Append-only JSONL checkpointing so interrupted runs resume without rework."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Marks the provenance record rather than a completed item.
FINGERPRINT_FIELD = "_checkpoint"


class CheckpointMismatchError(RuntimeError):
    """The checkpoint on disk was produced by a different run."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse one JSON object per line, skipping blanks; missing file is empty.

    A process killed mid-append leaves a partial final line. Appends are
    ordered, so only the last line can be torn: it is dropped with a warning so
    a resume loses one item's work rather than failing to parse the whole file.
    An unparseable line anywhere earlier is real corruption and still raises.
    """
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            print(
                f"{path}: dropping a truncated final line, most likely a run killed "
                "mid-append; that item will be redone",
                file=sys.stderr,
            )
    return records


def load_items(path: Path) -> list[dict[str, Any]]:
    """Checkpointed items only, skipping the provenance record.

    Every reader of a checkpointed file must use this rather than load_jsonl,
    since the provenance row carries no item key and would be parsed as a
    malformed item.
    """
    return [record for record in load_jsonl(path) if FINGERPRINT_FIELD not in record]


class JsonlCheckpoint:
    """Appends keyed records to a JSONL file and remembers which keys are done.

    ``fingerprint`` binds the file to the run that created it. Without it the
    file records only item ids, so pointing a second arm at a path that already
    holds the first arm's rows skips every item as already done and yields a
    complete, plausible artifact for arm B built entirely from arm A's data. The
    same silence covers resuming after a depth or question-set change, which
    mixes two configurations into one file. A resume whose fingerprint disagrees
    is refused rather than adopted.
    """

    def __init__(
        self, path: Path, key_field: str, *, fingerprint: dict[str, Any] | None = None
    ) -> None:
        self._path = path
        self._key_field = key_field
        self._fingerprint = fingerprint
        records = load_jsonl(path)
        stored = next(
            (record[FINGERPRINT_FIELD] for record in records if FINGERPRINT_FIELD in record),
            None,
        )
        if fingerprint is not None and records:
            self._require_match(stored, fingerprint, path)
        self._done = {
            str(record[key_field]) for record in records if FINGERPRINT_FIELD not in record
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        if fingerprint is not None and stored is None:
            self._write({FINGERPRINT_FIELD: fingerprint})

    @staticmethod
    def _require_match(stored: dict[str, Any] | None, expected: dict[str, Any], path: Path) -> None:
        if stored is None:
            raise CheckpointMismatchError(
                f"{path} has no provenance record, so it cannot be shown to belong to "
                f"this run ({expected}); start a fresh checkpoint or delete it"
            )
        if stored != expected:
            differing = sorted(
                key for key in set(stored) | set(expected) if stored.get(key) != expected.get(key)
            )
            raise CheckpointMismatchError(
                f"{path} was produced by a different run: {differing} differ "
                f"(file {stored}, this run {expected}). Resuming it would attribute "
                "one run's results to another"
            )

    @property
    def done(self) -> set[str]:
        return set(self._done)

    def __contains__(self, key: str) -> bool:
        return key in self._done

    def _write(self, record: dict[str, Any]) -> None:
        with self._path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def append(self, record: dict[str, Any]) -> None:
        self._write(record)
        self._done.add(str(record[self._key_field]))
