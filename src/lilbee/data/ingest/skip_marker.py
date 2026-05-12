"""Sidecar record of files that produced no chunks, so a sync can skip them.

A file that yields zero chunks (Tesseract timeout, decode failure, no usable
text) gets a marker here keyed by the file hash that failed.
``_plan_file_changes`` treats a file whose current hash matches its marker as
unchanged, so the per-file extract cost (30-60s for a stubborn scanned PDF) is
paid once, not on every sync. The marker is a small JSON file in
``cfg.data_root``; editing the file changes its hash and re-arms it, and
``retry_skipped`` / ``force_rebuild`` drop the file from the marker set.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

SKIP_MARKER_FILENAME = "skipped_sources.json"


def _marker_path(data_root: Path) -> Path:
    return data_root / SKIP_MARKER_FILENAME


def load_skip_markers(data_root: Path) -> dict[str, str]:
    """Load the filename → failed-hash map, or empty dict on any read error."""
    path = _marker_path(data_root)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("Skip-marker file unreadable, treating as empty: %s", exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def write_skip_markers(data_root: Path, markers: dict[str, str]) -> None:
    """Replace the marker file atomically. Best-effort: errors are logged, not raised."""
    path = _marker_path(data_root)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(markers, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("Failed to persist skip markers to %s: %s", path, exc)
        with contextlib.suppress(OSError):
            tmp.unlink()


def clear_skip_markers(data_root: Path) -> None:
    """Delete the marker file. No-op if absent."""
    path = _marker_path(data_root)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("Could not remove skip-marker file %s: %s", path, exc)
