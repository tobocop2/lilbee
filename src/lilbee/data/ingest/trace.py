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
    """Turn tracing on when LILBEE_INGEST_TRACE is truthy.

    lilbee's root logger defaults to WARNING, which would swallow the INFO trace
    lines. Setting the two named loggers to INFO lets their records through the
    isEnabledFor gate; propagation then hands them to the root handler (added by
    basicConfig at NOTSET), so they surface regardless of the root level.
    """
    if os.environ.get(_TRACE_ENV, "").lower() in ("1", "true", "yes"):
        trace_log.setLevel(logging.DEBUG)
        vision_log.setLevel(logging.INFO)


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
