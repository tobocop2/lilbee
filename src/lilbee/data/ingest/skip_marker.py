"""Sidecar records of files a sync holds out: ingestion failures and user removals.

``skipped_sources.json`` maps a filename to the file hash it is held out at.
``_plan_file_changes`` treats a file whose current hash matches its marker as
unchanged, so a failed extract is paid once and a removal stays out. Editing the
file changes its hash and re-arms it. ``skip_reasons.json`` records why each
file is held out, and ``skip_kinds.json`` records its ``SkipKind`` with the
hash and reason it was written for. A stored kind counts only while both still
match; otherwise the record reads as a removal when its reason is
``REMOVED_SKIP_REASON`` and as a failure when it is not. Production writes go
through ``update_skip_records`` and ``clear_skip_markers``, both under one
cross-process lock. A write first replaces ``skip_records.pending.json`` with
the three new texts, and a reader takes them from there until the write ends,
so the three files change together.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypedDict

from filelock import FileLock

from lilbee.data.types import SkippedSource

log = logging.getLogger(__name__)

SKIP_MARKER_FILENAME = "skipped_sources.json"
SKIP_REASON_FILENAME = "skip_reasons.json"
SKIP_KIND_FILENAME = "skip_kinds.json"
SKIP_PENDING_FILENAME = "skip_records.pending.json"
_RECORD_FILENAMES = (SKIP_MARKER_FILENAME, SKIP_REASON_FILENAME, SKIP_KIND_FILENAME)
DEFAULT_SKIP_REASON = "held out by an earlier sync"
REMOVED_SKIP_REASON = "removed via remove (re-add the source or run retry-skipped to restore)"
# A sync, a /delete, and a reset from another process all change these records.
# The total wait; filelock polls the lock until it runs out.
_RECORDS_LOCK_TIMEOUT_S = 10.0
_RECORDS_LOCKED = (
    "Could not lock {path}, so the list of held-out files was not changed. "
    "If no other lilbee process is running, delete that file and try again."
)


class SkipRecordsLockError(RuntimeError):
    """Raised when the records lock cannot be taken; nothing was read or written."""


class SkipKind(StrEnum):
    """Why a skip marker holds a file out of the sync."""

    FAILED = "failed"
    REMOVED = "removed"


class _StoredKind(TypedDict):
    """One ``skip_kinds.json`` entry: the kind and the marker it was written for."""

    kind: SkipKind
    hash: str
    reason: str | None


@dataclass
class SkipRecords:
    """The skip markers with the reason and the kind of each."""

    markers: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    kinds: dict[str, SkipKind] = field(default_factory=dict)


@dataclass(frozen=True)
class _PendingWrite:
    """A write of the three sidecars that has started: what each becomes and what it replaces."""

    after: dict[str, str]
    """The text each sidecar gets, by file name."""
    before: dict[str, str | None]
    """The digest of the bytes each sidecar held, None for a file that was absent."""


def _json_map(text: str, name: str) -> dict[str, object]:
    """The JSON object *text* holds, or an empty dict when it holds none."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        log.debug("Sidecar %s unreadable, treating as empty: %s", name, exc)
        return {}
    if not isinstance(raw, dict):  # the file is untyped JSON
        return {}
    return {str(k): v for k, v in raw.items()}


def _read_text(path: Path) -> str | None:
    """The text of *path*, or None when it cannot be read; an absent file is not opened."""
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.debug("Sidecar %s unreadable, treating as empty: %s", path.name, exc)
        return None


def _digest(text: str | None) -> str | None:
    """A digest of *text*, None for a file that is absent."""
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pending_write(data_root: Path) -> _PendingWrite | None:
    """The write a pending file records, or None without one that can be read."""
    text = _read_text(data_root / SKIP_PENDING_FILENAME)
    if text is None:
        return None
    raw = _json_map(text, SKIP_PENDING_FILENAME)
    after, before = raw.get("after"), raw.get("before")
    # the pending file is untyped JSON
    if not isinstance(after, dict) or not isinstance(before, dict):
        return None
    if not all(isinstance(after.get(name), str) for name in _RECORD_FILENAMES):
        return None
    return _PendingWrite(after, {name: before.get(name) for name in _RECORD_FILENAMES})


def _record_texts(
    data_root: Path, names: tuple[str, ...] = _RECORD_FILENAMES
) -> dict[str, str | None]:
    """The text of each sidecar in *names*, by file name: from a pending write while it stands.

    Without a pending file only the sidecars in *names* are read.
    """
    on_disk = {name: _read_text(data_root / name) for name in names}
    # Checked after the read, so a write that started during it is seen.
    if not (data_root / SKIP_PENDING_FILENAME).exists():
        return on_disk
    return _texts_through_pending_write(data_root)


def _texts_through_pending_write(data_root: Path) -> dict[str, str | None]:
    """The text of every sidecar while a pending file is there.

    A pending write stands while each sidecar holds what the write replaces or
    what it writes. Another text means a lilbee without this file wrote since,
    and the sidecars are then the newer state.
    """
    on_disk = {name: _read_text(data_root / name) for name in _RECORD_FILENAMES}
    pending = _pending_write(data_root)
    if pending is None:
        return on_disk
    for name, text in on_disk.items():
        if _digest(text) not in (pending.before[name], _digest(pending.after[name])):
            return on_disk
    return dict(pending.after)


def _load_json_map(data_root: Path, filename: str) -> dict[str, object]:
    """The JSON object the sidecar *filename* holds, or an empty dict on any read error."""
    text = _record_texts(data_root, (filename,))[filename]
    return {} if text is None else _json_map(text, filename)


def _str_map(raw: Mapping[str, object]) -> dict[str, str]:
    """The entries of *raw* whose value is a string."""
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def _replace_text(path: Path, text: str) -> bool:
    """Replace *path* atomically with *text*; False when it could not be written."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("Failed to persist %s: %s", path, exc)
        with contextlib.suppress(OSError):
            tmp.unlink()
        return False
    return True


def _write_json_map(path: Path, data: Mapping[str, object]) -> None:
    """Replace *path* atomically with a JSON object. Best-effort."""
    _replace_text(path, json.dumps(data, sort_keys=True))


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("Could not remove %s: %s", path, exc)


def load_skip_markers(data_root: Path) -> dict[str, str]:
    """Load the filename → failed-hash map, or empty dict on any read error."""
    return _str_map(_load_json_map(data_root, SKIP_MARKER_FILENAME))


def write_skip_markers(data_root: Path, markers: dict[str, str]) -> None:
    """Replace the marker file atomically. Best-effort: errors are logged, not raised."""
    _write_json_map(data_root / SKIP_MARKER_FILENAME, markers)


def load_skip_reasons(data_root: Path) -> dict[str, str]:
    """Load the filename → skip-reason map (informational), empty on any read error."""
    return _str_map(_load_json_map(data_root, SKIP_REASON_FILENAME))


def write_skip_reasons(data_root: Path, reasons: dict[str, str]) -> None:
    """Replace the reasons sidecar atomically. Best-effort: errors are logged, not raised."""
    _write_json_map(data_root / SKIP_REASON_FILENAME, reasons)


def _kind_of(stored: object, marker: str, reason: str | None) -> SkipKind:
    """The stored kind while it names this marker and reason; else REMOVED for the removal text."""
    # the kinds file is untyped JSON
    if isinstance(stored, dict) and (stored.get("hash"), stored.get("reason")) == (marker, reason):
        with contextlib.suppress(ValueError):
            return SkipKind(stored.get("kind", ""))
    return SkipKind.REMOVED if reason == REMOVED_SKIP_REASON else SkipKind.FAILED


def _load_records(data_root: Path) -> SkipRecords:
    """Read the three sidecars; a marker without a matching stored kind reads by its reason."""
    texts = _record_texts(data_root)
    maps = {name: {} if text is None else _json_map(text, name) for name, text in texts.items()}
    markers = _str_map(maps[SKIP_MARKER_FILENAME])
    reasons = _str_map(maps[SKIP_REASON_FILENAME])
    stored = maps[SKIP_KIND_FILENAME]
    kinds = {
        name: _kind_of(stored.get(name), marker, reasons.get(name))
        for name, marker in markers.items()
    }
    return SkipRecords(markers, reasons, kinds)


def write_skip_records(data_root: Path, records: SkipRecords) -> None:
    """Replace the three sidecars together; each kind names the hash and reason it is for.

    The pending file is written first and holds the three texts, so a write
    that stops part way reads as complete. Best-effort: errors are logged.
    """
    stored = {
        name: _StoredKind(kind=kind, hash=marker, reason=records.reasons.get(name))
        for name, kind in records.kinds.items()
        if (marker := records.markers.get(name)) is not None
    }
    maps = (records.markers, records.reasons, stored)
    after = {
        name: json.dumps(data, sort_keys=True)
        for name, data in zip(_RECORD_FILENAMES, maps, strict=True)
    }
    before = {name: _digest(_read_text(data_root / name)) for name in _RECORD_FILENAMES}
    pending = data_root / SKIP_PENDING_FILENAME
    _replace_text(pending, json.dumps(asdict(_PendingWrite(after, before)), sort_keys=True))
    _apply(data_root, after)


def _apply(data_root: Path, texts: Mapping[str, str]) -> None:
    """Give each sidecar its text of a pending write, then end the write."""
    written = [_replace_text(data_root / name, text) for name, text in texts.items()]
    if all(written):
        _unlink(data_root / SKIP_PENDING_FILENAME)


def _settle(data_root: Path) -> None:
    """Finish or drop a write that did not end; the caller holds the records lock."""
    if not (data_root / SKIP_PENDING_FILENAME).exists():
        return
    pending = _pending_write(data_root)
    if pending is not None and _record_texts(data_root) == pending.after:
        _apply(data_root, pending.after)
    else:
        _unlink(data_root / SKIP_PENDING_FILENAME)


def load_skip_records(data_root: Path) -> SkipRecords:
    """The skip markers with the reason and the kind of each, as they are on disk."""
    return _load_records(data_root)


def load_skip_kinds(data_root: Path) -> dict[str, SkipKind]:
    """The kind of every marked file."""
    return _load_records(data_root).kinds


@contextlib.contextmanager
def skip_records_lock(data_root: Path) -> Iterator[None]:
    """Hold the cross-process lock on the records, or raise without entering.

    Re-entrant within a thread, so an operation can take it before its first
    change and still call the record helpers inside.
    """
    lock = FileLock(str(data_root / SKIP_MARKER_FILENAME) + ".lock", is_singleton=True)
    try:
        lock.acquire(timeout=_RECORDS_LOCK_TIMEOUT_S)
    except OSError as error:  # filelock's Timeout is an OSError too
        raise SkipRecordsLockError(_RECORDS_LOCKED.format(path=lock.lock_file)) from error
    try:
        yield
    finally:
        lock.release()


def update_skip_records(data_root: Path, change: Callable[[SkipRecords], None]) -> None:
    """Apply *change* to the records as they are on disk, under a cross-process lock.

    Reasons and kinds whose marker is gone are dropped in the same write.
    """
    with skip_records_lock(data_root):
        _settle(data_root)
        records = _load_records(data_root)
        before = SkipRecords(dict(records.markers), dict(records.reasons), dict(records.kinds))
        change(records)
        records.reasons = {k: v for k, v in records.reasons.items() if k in records.markers}
        records.kinds = {k: v for k, v in records.kinds.items() if k in records.markers}
        if records != before:
            write_skip_records(data_root, records)


def held_out_names(data_root: Path) -> list[str]:
    """Every file held out by an ingestion failure, sorted; removed sources are not listed."""
    kinds = load_skip_kinds(data_root)
    return sorted(name for name, kind in kinds.items() if kind is SkipKind.FAILED)


def mark_removed(
    data_root: Path, hashes: Mapping[str, str], unreachable: Iterable[str] = ()
) -> None:
    """Hold files out of every sync as user removals.

    Each file in *hashes* is held at the given hash; each marked file in
    *unreachable* keeps the hash its marker already records.
    """

    def _hold(records: SkipRecords) -> None:
        kept = {name: records.markers[name] for name in unreachable if name in records.markers}
        held = {**kept, **hashes}
        records.markers.update(held)
        records.reasons.update(dict.fromkeys(held, REMOVED_SKIP_REASON))
        records.kinds.update(dict.fromkeys(held, SkipKind.REMOVED))

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


def describe_failures(data_root: Path, names: Iterable[str]) -> list[SkippedSource]:
    """Each of *names* an ingestion failure holds out, with its reason, in order."""
    records = _load_records(data_root)
    return [
        SkippedSource(filename=name, reason=records.reasons.get(name, DEFAULT_SKIP_REASON))
        for name in names
        if records.kinds.get(name) is SkipKind.FAILED
    ]


def clear_skip_markers(data_root: Path) -> None:
    """Delete the marker file and both sidecars. No-op if absent."""
    with skip_records_lock(data_root):
        _settle(data_root)
        _unlink(data_root / SKIP_MARKER_FILENAME)
        _unlink(data_root / SKIP_REASON_FILENAME)
        _unlink(data_root / SKIP_KIND_FILENAME)
