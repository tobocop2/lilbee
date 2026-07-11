"""Sidecar records of files that produced no chunks, so a sync can skip them.

A file that yields zero chunks (Tesseract timeout, decode failure, no usable
text) gets a marker here keyed by the file hash that failed.
``_plan_file_changes`` treats a file whose current hash matches its marker as
unchanged, so the per-file extract cost (30-60s for a stubborn scanned PDF) is
paid once, not on every sync. The marker is a small JSON file in
``cfg.data_root``; editing the file changes its hash and re-arms it, and
``retry_skipped`` / ``force_rebuild`` drop the file from the marker set.

A second sidecar (``skip_reasons.json``) records filename → human-readable
reason, so a report can say WHY a file was skipped (the exception message, or
"no text extracted"), not just that it was. It is informational only -- the
hash-keyed markers above drive the resume logic -- and is cleared alongside them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

SKIP_MARKER_FILENAME = "skipped_sources.json"
SKIP_REASON_FILENAME = "skip_reasons.json"


def _load_str_map(path: Path) -> dict[str, str]:
    """Load a ``{str: str}`` JSON file, or empty dict on any read/parse error."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
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


def clear_skip_markers(data_root: Path) -> None:
    """Delete both the marker file and the reasons sidecar. No-op if absent."""
    _unlink(data_root / SKIP_MARKER_FILENAME)
    _unlink(data_root / SKIP_REASON_FILENAME)
