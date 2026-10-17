"""The held-out records an older lilbee kept in JSON files at the data root."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path

from .types import REMOVED_REASON, HeldOut, HeldRecord

log = logging.getLogger(__name__)

_MARKERS = "skipped_sources.json"
_REASONS = "skip_reasons.json"
_KINDS = "skip_kinds.json"
_PENDING = "skip_records.pending.json"
_RECORDS = (_MARKERS, _REASONS, _KINDS)
# Every file the older format kept, the lock of its writers included.
LEGACY_FILES = (*_RECORDS, _PENDING, f"{_MARKERS}.lock")
# The reason an older lilbee wrote for a removal.
_OLD_REMOVED_REASON = "removed via remove (re-add the source or run retry-skipped to restore)"
_DEFAULT_REASON = "held out by an earlier sync"


def legacy_files(data_root: Path) -> list[Path]:
    """The files of the older format that are still at *data_root*."""
    return [path for name in LEGACY_FILES if (path := data_root / name).exists()]


def _text(path: Path) -> str | None:
    """The text of *path*, or None when it is absent or cannot be read."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        log.debug("Cannot read %s: %s", path.name, exc)
        return None


def _object(text: str | None) -> dict[str, object]:
    """The JSON object *text* holds, or an empty one."""
    if text is None:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return {}
    # the files are untyped JSON
    return {str(key): value for key, value in raw.items()} if isinstance(raw, dict) else {}


def _texts(data_root: Path) -> dict[str, str | None]:
    """The text of each record file, from a write that did not end when one is recorded."""
    after = _object(_text(data_root / _PENDING)).get("after")
    # the pending file is untyped JSON
    if isinstance(after, dict) and all(isinstance(after.get(name), str) for name in _RECORDS):
        return {name: after[name] for name in _RECORDS}
    return {name: _text(data_root / name) for name in _RECORDS}


def _kind(stored: object, marker: str, reason: str | None) -> HeldOut:
    """The stored kind while it names this marker and reason; else the kind its reason gives."""
    # the kinds file is untyped JSON
    if isinstance(stored, dict) and (stored.get("hash"), stored.get("reason")) == (marker, reason):
        with contextlib.suppress(ValueError):
            return HeldOut(stored.get("kind", ""))
    return HeldOut.REMOVED if reason == _OLD_REMOVED_REASON else HeldOut.FAILED


def read_legacy_records(data_root: Path) -> dict[str, HeldRecord]:
    """The record of each file the JSON files at *data_root* hold out, by path."""
    texts = _texts(data_root)
    markers, reasons, kinds = (_object(texts[name]) for name in _RECORDS)
    records: dict[str, HeldRecord] = {}
    for name, marker in markers.items():
        if not isinstance(marker, str):
            continue
        reason = reasons.get(name)
        text = reason if isinstance(reason, str) else None
        kind = _kind(kinds.get(name), marker, text)
        shown = REMOVED_REASON if kind is HeldOut.REMOVED else text or _DEFAULT_REASON
        records[name] = HeldRecord(kind, marker, shown)
    return records


def remove_legacy_files(data_root: Path) -> None:
    """Delete the files of the older format; one that cannot go is read again next open."""
    for path in legacy_files(data_root):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not delete %s: %s", path, exc)
