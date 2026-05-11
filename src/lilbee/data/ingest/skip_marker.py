"""Sidecar persistence for "this file failed last time, don't retry on every sync".

Without this, the diff in ``_plan_file_changes`` re-discovers every file that
produced zero chunks (Tesseract timeout, decode failure, no usable text) and
retries it every sync. On corpora with a few stubborn scanned PDFs that costs
30-60 seconds per file per sync forever.

The marker is a tiny JSON file in ``cfg.data_root`` mapping filename to the
file hash that last failed. When a file's current hash matches its marker, the
pipeline skips re-processing. ``/sync --force-rebuild`` clears the markers.
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
