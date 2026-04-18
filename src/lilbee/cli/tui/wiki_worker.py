"""Shared wiki generation worker for TUI screens.

Extracts the wiki background generation logic so both ChatScreen and WikiScreen
can trigger wiki generation through ``TaskBarController.start_task``. The
worker receives a ``ProgressReporter`` and writes progress through it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lilbee.cli.tui import messages as msg
from lilbee.services import get_services

if TYPE_CHECKING:
    from lilbee.cli.tui.widgets.task_bar import ProgressReporter

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


def _process_source(
    source: str, idx: int, total: int, reporter: ProgressReporter, errors: list[str]
) -> bool:
    """Generate a wiki page for one source. Returns True if a page was produced."""
    from lilbee.wiki.gen import generate_summary_page

    svc = get_services()
    reporter.update(
        int(idx * 100 / total),
        msg.CMD_WIKI_PROGRESS.format(name=source, stage=WIKI_STAGE_PREPARING),
    )
    chunks = svc.store.get_chunks_by_source(source)
    if not chunks:
        return False

    def _on_stage(stage: str, data: dict[str, object]) -> None:
        if stage == WIKI_STAGE_FAILED:
            errors.append(str(data.get("error", msg.CMD_WIKI_UNKNOWN_ERROR)))
            return
        fraction = WIKI_STAGE_FRACTIONS.get(stage, 0.0)
        pct = int((idx + fraction) * 100 / total)
        reporter.update(pct, msg.CMD_WIKI_PROGRESS.format(name=source, stage=stage))

    # Suppress wiki gen warnings during TUI mode to prevent stderr corruption.
    wiki_logger = logging.getLogger("lilbee.wiki.gen")
    original_level = wiki_logger.level
    wiki_logger.setLevel(logging.ERROR)
    try:
        result = generate_summary_page(
            source, chunks, svc.provider, svc.store, on_progress=_on_stage
        )
    finally:
        wiki_logger.setLevel(original_level)
    return result is not None


def generate_wiki_pages(sources: list[str], reporter: ProgressReporter) -> int:
    """Generate wiki pages for *sources*.

    Runs on a daemon worker thread; ``reporter`` is the
    ``TaskBarController`` reporter for this task. Returns the number of
    pages generated. Raises ``RuntimeError`` (caught by the controller and
    surfaced as a task failure) when zero pages were produced.
    """
    total = len(sources)
    generated = 0
    errors: list[str] = []

    for idx, source in enumerate(sources):
        reporter.check_cancelled()
        if _process_source(source, idx, total, reporter, errors):
            generated += 1

    if generated == 0:
        reason = errors[-1] if errors else msg.CMD_WIKI_NO_PAGES
        raise RuntimeError(reason)
    return generated
