"""Shared wiki generation worker for TUI screens.

Extracts the wiki background generation logic so both ChatScreen and WikiScreen
can trigger wiki generation without coupling to each other.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.services import get_services

if TYPE_CHECKING:
    from textual.dom import DOMNode

log = logging.getLogger(__name__)

WIKI_STAGE_PREPARING = "preparing"
WIKI_STAGE_GENERATING = "generating"
WIKI_STAGE_FAITHFULNESS = "faithfulness_check"
WIKI_STAGE_FAILED = "failed"

WIKI_STAGE_FRACTIONS: dict[str, float] = {
    WIKI_STAGE_PREPARING: 0.0,
    WIKI_STAGE_GENERATING: 0.33,
    WIKI_STAGE_FAITHFULNESS: 0.67,
}


def resolve_wiki_targets(requested: str | None = None) -> list[str] | None:
    """Resolve source names for wiki generation.

    Callers must check ``cfg.wiki`` before calling. Returns a list of source
    names, or None when no sources are indexed or *requested* is not found.
    """
    try:
        sources = get_services().store.get_sources()
    except Exception:
        log.warning("Failed to list sources for wiki", exc_info=True)
        return None
    names = [s["filename"] for s in sources if s.get("filename")]
    if not names:
        return None
    if requested is not None:
        if requested not in names:
            return None
        return [requested]
    return names


def _make_progress_callback(
    source_name: str,
    source_idx: int,
    total: int,
    widget: DOMNode,
    update_task: Callable[[str, int, str], None],
    task_id: str,
    errors: list[str],
) -> Callable[[str, dict[str, object]], None]:
    """Build a progress callback for a single source's wiki generation."""

    def _on_progress(stage: str, _data: dict[str, object]) -> None:
        if stage == WIKI_STAGE_FAILED:
            errors.append(str(_data.get("error", msg.CMD_WIKI_UNKNOWN_ERROR)))
            return
        fraction = WIKI_STAGE_FRACTIONS.get(stage, 0.0)
        pct = int((source_idx + fraction) * 100 / total)
        call_from_thread(
            widget,
            update_task,
            task_id,
            pct,
            msg.CMD_WIKI_PROGRESS.format(name=source_name, stage=stage),
        )

    return _on_progress


def _report_result(
    generated: int,
    total: int,
    errors: list[str],
    widget: DOMNode,
    task_id: str,
    complete_task: Callable[[str], None],
    fail_task: Callable[[str, str], None],
    notify: Callable[..., None],
    on_complete: Callable[[], None] | None,
) -> None:
    """Notify the user of wiki generation results."""
    if generated > 0:
        call_from_thread(widget, complete_task, task_id)
        call_from_thread(
            widget, notify, msg.CMD_WIKI_SUCCESS.format(generated=generated, total=total)
        )
        if on_complete is not None:
            call_from_thread(widget, on_complete)
    else:
        fail_reason = errors[-1] if errors else msg.CMD_WIKI_NO_PAGES
        call_from_thread(widget, fail_task, task_id, fail_reason)
        call_from_thread(
            widget, notify, msg.CMD_WIKI_NONE_GENERATED.format(total=total), severity="warning"
        )


def _process_source(
    source: str,
    idx: int,
    total: int,
    widget: DOMNode,
    update_task: Callable[[str, int, str], None],
    task_id: str,
    errors: list[str],
) -> bool:
    """Generate a wiki page for one source. Returns True if a page was produced."""
    from lilbee.wiki.gen import generate_summary_page

    svc = get_services()
    call_from_thread(
        widget,
        update_task,
        task_id,
        int(idx * 100 / total),
        msg.CMD_WIKI_PROGRESS.format(name=source, stage=WIKI_STAGE_PREPARING),
    )
    chunks = svc.store.get_chunks_by_source(source)
    if not chunks:
        return False
    progress_cb = _make_progress_callback(source, idx, total, widget, update_task, task_id, errors)
    # Suppress wiki gen warnings during TUI mode to prevent stderr corruption.
    wiki_logger = logging.getLogger("lilbee.wiki.gen")
    original_level = wiki_logger.level
    wiki_logger.setLevel(logging.ERROR)
    try:
        result = generate_summary_page(
            source, chunks, svc.provider, svc.store, on_progress=progress_cb
        )
    finally:
        wiki_logger.setLevel(original_level)
    return result is not None


def run_wiki_generation(
    sources: list[str],
    task_id: str,
    widget: DOMNode,
    update_task: Callable[[str, int, str], None],
    complete_task: Callable[[str], None],
    fail_task: Callable[[str, str], None],
    notify: Callable[..., None],
    on_complete: Callable[[], None] | None = None,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> None:
    """Run wiki generation for the given sources (call from a background thread)."""
    total = len(sources)
    generated = 0
    errors: list[str] = []

    try:
        for idx, source in enumerate(sources):
            if is_cancelled():
                break
            if _process_source(source, idx, total, widget, update_task, task_id, errors):
                generated += 1

        _report_result(
            generated,
            total,
            errors,
            widget,
            task_id,
            complete_task,
            fail_task,
            notify,
            on_complete,
        )
    except Exception as exc:
        log.warning("Wiki generation failed", exc_info=True)
        call_from_thread(widget, fail_task, task_id, str(exc))
        call_from_thread(widget, notify, msg.CMD_WIKI_FAILED.format(error=exc), severity="error")
