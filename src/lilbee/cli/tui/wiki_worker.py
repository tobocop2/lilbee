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
from lilbee.config import cfg
from lilbee.services import get_services

if TYPE_CHECKING:
    from textual.dom import DOMNode

log = logging.getLogger(__name__)

WIKI_SUBCMD_GENERATE = "generate"
WIKI_STAGE_PREPARING = "preparing"

WIKI_STAGE_FRACTIONS: dict[str, float] = {
    WIKI_STAGE_PREPARING: 0.0,
    "generating": 0.33,
    "faithfulness_check": 0.67,
}


def resolve_wiki_targets(requested: str | None = None) -> list[str] | None:
    """Resolve source names for wiki generation.

    Returns a list of source names, or None if validation fails (caller should
    notify the user with the appropriate message).
    """
    if not cfg.wiki:
        return None
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
    """Run wiki generation for the given sources (call from a background thread).

    Parameters
    ----------
    sources:
        List of source filenames to generate wiki pages for.
    task_id:
        The task queue ID for progress updates.
    widget:
        The Textual widget used as the context for ``call_from_thread``.
    update_task, complete_task, fail_task:
        Task bar callbacks for progress, completion, and failure.
    notify:
        Notification callback (e.g., ``screen.notify``).
    on_complete:
        Optional callback fired after successful generation (e.g., reload wiki sidebar).
    is_cancelled:
        Returns True if the worker has been cancelled.
    """
    from lilbee.wiki.gen import generate_summary_page

    svc = get_services()
    total = len(sources)
    generated = 0
    last_error: str = ""

    try:
        for idx, source in enumerate(sources):
            if is_cancelled():
                break
            base_pct = int(idx * 100 / total)
            call_from_thread(
                widget,
                update_task,
                task_id,
                base_pct,
                msg.CMD_WIKI_PROGRESS.format(name=source, stage=WIKI_STAGE_PREPARING),
            )
            chunks = svc.store.get_chunks_by_source(source)
            if not chunks:
                continue

            def _on_progress(
                stage: str,
                _data: dict[str, object],
                source_name: str = source,
                source_idx: int = idx,
            ) -> None:
                nonlocal last_error
                if stage == "failed":
                    last_error = str(_data.get("error", msg.CMD_WIKI_UNKNOWN_ERROR))
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

            result = generate_summary_page(
                source, chunks, svc.provider, svc.store, on_progress=_on_progress
            )
            if result is not None:
                generated += 1

        if generated > 0:
            call_from_thread(widget, complete_task, task_id)
            call_from_thread(
                widget,
                notify,
                msg.CMD_WIKI_SUCCESS.format(generated=generated, total=total),
            )
            if on_complete is not None:
                call_from_thread(widget, on_complete)
        else:
            fail_reason = last_error or msg.CMD_WIKI_NO_PAGES
            call_from_thread(widget, fail_task, task_id, fail_reason)
            call_from_thread(
                widget,
                notify,
                msg.CMD_WIKI_NONE_GENERATED.format(total=total),
                severity="warning",
            )
    except Exception as exc:
        log.warning("Wiki generation failed", exc_info=True)
        call_from_thread(widget, fail_task, task_id, str(exc))
        call_from_thread(widget, notify, msg.CMD_WIKI_FAILED.format(error=exc), severity="error")
