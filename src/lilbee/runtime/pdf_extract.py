"""Subprocess entry point for vision-OCR PDF extraction.

Runs ``extract_pdf_vision`` in a child process so pdfium's internal
JPEG-2000 decoder cannot saturate the foreground TUI's scheduler bucket
on macOS. The parent reads JSON args from this process's stdin and
returns ``page_texts`` (or an error) as JSON on stdout. Per-page
progress lines are written to stderr in the form
``progress: page=N total=M`` for the parent to surface in the TaskBar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _emit_progress(page: int, total: int) -> None:
    sys.stderr.write(f"progress: page={page} total={total}\n")
    sys.stderr.flush()


def _on_progress(event_type: object, data: object) -> None:
    """Bridge ``DetailedProgressCallback`` events to stderr progress lines."""
    from lilbee.runtime.progress import BatchProgressEvent, EventType

    if event_type != EventType.BATCH_PROGRESS or not isinstance(data, BatchProgressEvent):
        return
    _emit_progress(data.current, data.total)


def main() -> int:
    """Read JSON args from stdin; write JSON result to stdout. Exit 0 on success."""
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read() or "{}")
        path = Path(payload["path"])
        model = str(payload["vision_model"])
        quiet = bool(payload.get("quiet", True))
        timeout_raw = payload.get("timeout")
        timeout: float | None = float(timeout_raw) if timeout_raw is not None else None
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        sys.stdout.write(json.dumps({"error": f"invalid args: {exc}"}))
        return 1

    try:
        from lilbee.vision import extract_pdf_vision

        page_texts = extract_pdf_vision(
            path,
            model,
            quiet=quiet,
            timeout=timeout,
            on_progress=_on_progress,
        )
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}))
        return 1

    sys.stdout.write(json.dumps({"page_texts": [list(pt) for pt in page_texts]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
