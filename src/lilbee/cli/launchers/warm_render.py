"""Render chat-model warm progress from the server's SSE stream.

A launcher consumes ``/api/warm/stream`` and drives a rich progress display so
the user sees a real read-phase byte bar, then an engine-load spinner, while a
large chat model loads, instead of a frozen line. Designed to degrade cleanly:
when the stream can't be opened (server still binding, a transient transport
error, or the endpoint absent) the caller falls back to a plain readiness poll.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from lilbee.catalog.formatting import display_label_for_ref
from lilbee.cli.app import console
from lilbee.providers.warm_progress import WarmPhase, WarmProgress

_WARM_STREAM_PATH = "/api/warm/stream"
_SSE_DATA_PREFIX = "data:"
_STREAM_CONNECT_TIMEOUT_S = 5.0
_DEFAULT_MODEL_LABEL = "chat model"


def _model_label(ref: str | None) -> str:
    """The canonical UI label for the warming model, or a generic fallback."""
    return display_label_for_ref(ref) if ref else _DEFAULT_MODEL_LABEL


def _iter_warm_events(base_url: str, timeout_s: float) -> Iterator[WarmProgress]:
    """Yield ``WarmProgress`` snapshots parsed from the SSE warm stream.

    Raises ``httpx.HTTPError`` if the stream cannot be opened (e.g. a server
    without the endpoint), so the caller can fall back to a plain poll.
    """
    timeout = httpx.Timeout(timeout_s, connect=_STREAM_CONNECT_TIMEOUT_S)
    with httpx.stream("GET", f"{base_url}{_WARM_STREAM_PATH}", timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith(_SSE_DATA_PREFIX):
                continue
            payload = line[len(_SSE_DATA_PREFIX) :].strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            # The terminal ``done`` event carries ``{}`` (no phase); only real
            # snapshots are yielded, so the renderer ignores it naturally.
            if isinstance(data, dict) and "phase" in data:
                yield WarmProgress.model_validate(data)


def _apply(progress: Progress, task_id: TaskID, snap: WarmProgress) -> None:
    """Reflect one warm snapshot onto the rich progress task."""
    if snap.phase is WarmPhase.READING_WEIGHTS:
        progress.update(
            task_id,
            description=f"Reading {_model_label(snap.model_ref)} weights",
            total=snap.bytes_total or None,
            completed=snap.bytes_done,
            detail=snap.detail or "",
        )
    elif snap.phase is WarmPhase.LOADING_ENGINE:
        # No byte signal during the VRAM load: drop to an indeterminate spinner.
        progress.update(
            task_id,
            description="Loading engine",
            total=None,
            detail=snap.detail or "",
        )
    elif snap.phase is WarmPhase.READY:
        total = snap.bytes_total or None
        progress.update(task_id, description="Chat model ready", total=total, completed=total or 0)
    elif snap.phase is WarmPhase.ERROR:
        progress.update(task_id, description="Chat model failed to load", detail=snap.error or "")
    else:  # STARTING
        progress.update(task_id, description="Preparing chat model", total=None, detail="")


def render_warm(base_url: str, timeout_s: float) -> bool | None:
    """Drive a progress display from the warm stream.

    Returns ``True`` once the chat engine reports ready, ``False`` on an error
    phase or if the stream ends before ready (the caller proceeds either way),
    and ``None`` when the stream could not be used at all so the caller falls
    back to a plain readiness poll.
    """
    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TextColumn("{task.fields[detail]}"),
        TimeElapsedColumn(),
    )
    saw_event = False
    try:
        with Progress(*columns, console=console, transient=True) as progress:
            task_id = progress.add_task("Preparing chat model", total=None, detail="")
            reached_ready = False
            for snap in _iter_warm_events(base_url, timeout_s):
                saw_event = True
                _apply(progress, task_id, snap)
                if snap.phase is WarmPhase.READY:
                    reached_ready = True
                    break
                if snap.phase is WarmPhase.ERROR:
                    return False
            return reached_ready
    except httpx.HTTPError:
        # Only None means "stream never ran, poll instead". If it opened and
        # yielded events before dropping, report not-ready so the caller does not
        # double-spend the full warm budget on a second poll.
        return None if not saw_event else False
