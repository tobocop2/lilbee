"""Sidecar records of files a sync holds out: ingestion failures and user removals.

``skipped_sources.json`` maps a filename to the file hash it is held out at.
``_plan_file_changes`` treats a file whose current hash matches its marker as
unchanged, so a failed extract is paid once and a removal stays out. Editing the
file changes its hash and re-arms it. ``skip_reasons.json`` records why each
file is held out, and ``skip_kinds.json`` records its ``SkipKind``. A record
with no stored kind reads as a removal when its reason is ``REMOVED_SKIP_REASON``.
Production writes go through ``update_skip_records`` and ``clear_skip_markers``,
both under one cross-process lock.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from lilbee.core.security import file_lock_or_warn
from lilbee.data.types import SkippedSource

log = logging.getLogger(__name__)

SKIP_MARKER_FILENAME = "skipped_sources.json"
SKIP_REASON_FILENAME = "skip_reasons.json"
SKIP_KIND_FILENAME = "skip_kinds.json"
DEFAULT_SKIP_REASON = "held out by an earlier sync"
REMOVED_SKIP_REASON = "removed via remove (re-add the source or run retry-skipped to restore)"
# A sync, a /delete, and a reset from another process all change these records.
_RECORDS_LOCK_TIMEOUT_S = 10.0


class SkipKind(StrEnum):
    """Why a skip marker holds a file out of the sync."""

    FAILED = "failed"
    REMOVED = "removed"


@dataclass
class SkipRecords:
    """The skip markers with the reason and the kind of each."""

    markers: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    kinds: dict[str, SkipKind] = field(default_factory=dict)


def _load_str_map(path: Path) -> dict[str, str]:
    """Load a ``{str: str}`` JSON file, or empty dict on any read/parse error."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.debug("Sidecar %s unreadable, treating as empty: %s", path.name, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def _write_str_map(path: Path, data: dict[str, str]) -> None:
    """Replace *path* atomically with a ``{str: str}`` JSON map. Best-effort."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("Failed to persist %s: %s", path, exc)
        with contextlib.suppress(OSError):
            tmp.unlink()


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("Could not remove %s: %s", path, exc)


def load_skip_markers(data_root: Path) -> dict[str, str]:
    """Load the filename → failed-hash map, or empty dict on any read error."""
    return _load_str_map(data_root / SKIP_MARKER_FILENAME)


def write_skip_markers(data_root: Path, markers: dict[str, str]) -> None:
    """Replace the marker file atomically. Best-effort: errors are logged, not raised."""
    _write_str_map(data_root / SKIP_MARKER_FILENAME, markers)


def load_skip_reasons(data_root: Path) -> dict[str, str]:
    """Load the filename → skip-reason map (informational), empty on any read error."""
    return _load_str_map(data_root / SKIP_REASON_FILENAME)


def write_skip_reasons(data_root: Path, reasons: dict[str, str]) -> None:
    """Replace the reasons sidecar atomically. Best-effort: errors are logged, not raised."""
    _write_str_map(data_root / SKIP_REASON_FILENAME, reasons)


def _kind_of(stored: str, reason: str | None) -> SkipKind:
    """The stored kind, or for a record without one: REMOVED when its reason is the removal text."""
    try:
        return SkipKind(stored)
    except ValueError:
        return SkipKind.REMOVED if reason == REMOVED_SKIP_REASON else SkipKind.FAILED


def _load_records(data_root: Path) -> SkipRecords:
    """Read the three sidecars; a marker without a stored kind is FAILED unless removed."""
    markers = load_skip_markers(data_root)
    reasons = load_skip_reasons(data_root)
    stored = _load_str_map(data_root / SKIP_KIND_FILENAME)
    kinds = {name: _kind_of(stored.get(name, ""), reasons.get(name)) for name in markers}
    return SkipRecords(markers, reasons, kinds)


def _write_records(data_root: Path, records: SkipRecords) -> None:
    """Replace the three sidecars. Best-effort: errors are logged, not raised."""
    write_skip_markers(data_root, records.markers)
    write_skip_reasons(data_root, records.reasons)
    write_skip_kinds(data_root, records.kinds)


def load_skip_kinds(data_root: Path) -> dict[str, SkipKind]:
    """The kind of every marked file; a record without a stored kind is FAILED unless removed."""
    return _load_records(data_root).kinds


def write_skip_kinds(data_root: Path, kinds: Mapping[str, SkipKind]) -> None:
    """Replace the kinds sidecar atomically. Best-effort: errors are logged, not raised."""
    _write_str_map(
        data_root / SKIP_KIND_FILENAME, {name: str(kind) for name, kind in kinds.items()}
    )


def update_skip_records(data_root: Path, change: Callable[[SkipRecords], None]) -> None:
    """Apply *change* to the records as they are on disk, under a cross-process lock.

    Reasons and kinds whose marker is gone are dropped in the same write.
    """
    with file_lock_or_warn(data_root / SKIP_MARKER_FILENAME, _RECORDS_LOCK_TIMEOUT_S):
        records = _load_records(data_root)
        before = SkipRecords(dict(records.markers), dict(records.reasons), dict(records.kinds))
        change(records)
        records.reasons = {k: v for k, v in records.reasons.items() if k in records.markers}
        records.kinds = {k: v for k, v in records.kinds.items() if k in records.markers}
        if records != before:
            _write_records(data_root, records)


def held_out_names(data_root: Path) -> list[str]:
    """Every file held out by an ingestion failure, sorted; removed sources are not listed."""
    kinds = load_skip_kinds(data_root)
    return sorted(name for name, kind in kinds.items() if kind is SkipKind.FAILED)


def mark_removed(data_root: Path, hashes: Mapping[str, str]) -> None:
    """Hold each file in *hashes* out of every sync as a user removal, at the given hash."""

    def _hold(records: SkipRecords) -> None:
        records.markers.update(hashes)
        records.reasons.update(dict.fromkeys(hashes, REMOVED_SKIP_REASON))
        records.kinds.update(dict.fromkeys(hashes, SkipKind.REMOVED))

    update_skip_records(data_root, _hold)


def clear_failed_markers(data_root: Path) -> list[str]:
    """Drop the record of every file an ingestion failure holds out; removals stay. Returns them."""
    dropped: list[str] = []

    def _drop(records: SkipRecords) -> None:
        dropped.extend(name for name, kind in records.kinds.items() if kind is SkipKind.FAILED)
        for name in dropped:
            records.markers.pop(name)

    update_skip_records(data_root, _drop)
    return dropped


def describe_skips(data_root: Path, names: Iterable[str]) -> list[SkippedSource]:
    """Pair each name with its recorded reason, in order; ``DEFAULT_SKIP_REASON`` when none."""
    reasons = load_skip_reasons(data_root)
    return [
        SkippedSource(filename=name, reason=reasons.get(name, DEFAULT_SKIP_REASON))
        for name in names
    ]


def clear_skip_markers(data_root: Path) -> None:
    """Delete the marker file and both sidecars. No-op if absent."""
    with file_lock_or_warn(data_root / SKIP_MARKER_FILENAME, _RECORDS_LOCK_TIMEOUT_S):
        _unlink(data_root / SKIP_MARKER_FILENAME)
        _unlink(data_root / SKIP_REASON_FILENAME)
        _unlink(data_root / SKIP_KIND_FILENAME)
