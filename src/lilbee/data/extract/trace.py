"""Structured per-file extraction tracing, for sharing ingest diagnostics.

Every xberg extraction emits one machine-parseable line on the ``lilbee.ingest.trace``
logger: filename, wall-clock, chunk and page counts, and how many pages fell through
to OCR. Files that needed the vision model also emit a line on ``lilbee.ingest.vision``
so ``grep vision`` yields exactly the set of scanned files. Enable with
``LILBEE_INGEST_TRACE=1`` (sets both loggers to DEBUG).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

trace_log = logging.getLogger("lilbee.ingest.trace")
vision_log = logging.getLogger("lilbee.ingest.vision")

_TRACE_ENV = "LILBEE_INGEST_TRACE"


@dataclass(frozen=True)
class ExtractionTrace:
    """One xberg extraction's measured outcome."""

    source: str
    content_type: str
    elapsed_s: float
    page_count: int
    chunk_count: int
    ocr_pages: int
    vision_configured: bool

    @property
    def used_vision(self) -> bool:
        """A page fell through to OCR and the OCR backend is the vision model."""
        return self.ocr_pages > 0 and self.vision_configured

    def as_line(self) -> str:
        """A stable key=value line, easy to grep, diff, and hand to the xberg author."""
        return (
            f"extract source={self.source!r} type={self.content_type} "
            f"elapsed_ms={self.elapsed_s * 1000:.0f} pages={self.page_count} "
            f"chunks={self.chunk_count} ocr_pages={self.ocr_pages} "
            f"vision={'yes' if self.used_vision else 'no'}"
        )


def configure_from_env() -> None:
    """Enable trace/vision logging per LILBEE_INGEST_TRACE; mirror to a file when
    LILBEE_INGEST_TRACE_FILE is set.

    The file handler exists because host apps own the root handlers: the TUI
    logs WARNING+ to its file, so INFO trace lines vanish there even with the
    loggers enabled. A dedicated handler on these two loggers makes the trace
    destination independent of whichever front-end is running."""
    if os.environ.get("LILBEE_INGEST_TRACE", "").strip().lower() not in {"1", "true", "yes"}:
        return
    trace_log.setLevel(logging.DEBUG)
    vision_log.setLevel(logging.INFO)
    target = os.environ.get("LILBEE_INGEST_TRACE_FILE", "").strip()
    if not target:
        return
    resolved = str(Path(target).absolute())
    for logger in (trace_log, vision_log):
        if any(
            isinstance(h, logging.FileHandler) and h.baseFilename == resolved
            for h in logger.handlers
        ):
            continue
        handler = logging.FileHandler(resolved)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)


def trace_extraction(trace: ExtractionTrace) -> None:
    """Log one extraction's outcome, plus a dedicated line if it needed vision."""
    trace_log.info("%s", trace.as_line())
    if trace.used_vision:
        vision_log.info(
            "vision-ocr source=%r ocr_pages=%d elapsed_ms=%.0f",
            trace.source,
            trace.ocr_pages,
            trace.elapsed_s * 1000,
        )
