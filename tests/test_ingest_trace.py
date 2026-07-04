"""The extraction trace: one parseable line per file, a vision line when OCR fired."""

from __future__ import annotations

import logging

import pytest

from lilbee.data.ingest.trace import (
    ExtractionTrace,
    configure_from_env,
    trace_extraction,
    trace_log,
    vision_log,
)


def _trace(**kw: object) -> ExtractionTrace:
    base = {
        "source": "doj-ds05/EFTA00000123.pdf",
        "content_type": "application/pdf",
        "elapsed_s": 1.234,
        "page_count": 10,
        "chunk_count": 42,
        "ocr_pages": 0,
        "vision_configured": True,
    }
    base.update(kw)
    return ExtractionTrace(**base)  # type: ignore[arg-type]


def test_line_carries_filename_timing_and_counts() -> None:
    line = _trace().as_line()
    assert "source='doj-ds05/EFTA00000123.pdf'" in line
    assert "elapsed_ms=1234" in line
    assert "pages=10" in line
    assert "chunks=42" in line
    assert "ocr_pages=0" in line
    assert "vision=no" in line


def test_used_vision_requires_ocr_pages_and_a_configured_model() -> None:
    assert _trace(ocr_pages=5, vision_configured=True).used_vision is True
    assert _trace(ocr_pages=5, vision_configured=False).used_vision is False  # tesseract
    assert _trace(ocr_pages=0, vision_configured=True).used_vision is False  # all native


def test_native_only_file_emits_no_vision_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="lilbee.ingest.vision")
    caplog.set_level(logging.INFO, logger="lilbee.ingest.trace")
    trace_extraction(_trace(ocr_pages=0))
    assert any(r.name == "lilbee.ingest.trace" for r in caplog.records)
    assert not any(r.name == "lilbee.ingest.vision" for r in caplog.records)


def test_scanned_file_emits_a_dedicated_vision_line(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="lilbee.ingest.vision")
    trace_extraction(_trace(ocr_pages=7, vision_configured=True))
    vision_records = [r for r in caplog.records if r.name == "lilbee.ingest.vision"]
    assert len(vision_records) == 1
    assert "ocr_pages=7" in vision_records[0].getMessage()
    assert "doj-ds05/EFTA00000123.pdf" in vision_records[0].getMessage()


def test_env_flag_enables_both_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_log.setLevel(logging.WARNING)
    vision_log.setLevel(logging.WARNING)
    monkeypatch.setenv("LILBEE_INGEST_TRACE", "1")
    configure_from_env()
    assert trace_log.level == logging.DEBUG  # includes the extract-start debug line
    assert vision_log.level == logging.INFO  # vision lines are INFO, and emit


def test_trace_surfaces_under_a_warning_root(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real failure mode: root at WARNING (lilbee's default) silently drops
    # INFO trace lines. The env flag must lift the named loggers so records emit.
    logging.getLogger().setLevel(logging.WARNING)
    trace_log.setLevel(logging.WARNING)
    vision_log.setLevel(logging.WARNING)
    monkeypatch.setenv("LILBEE_INGEST_TRACE", "1")
    configure_from_env()
    assert trace_log.isEnabledFor(logging.INFO)
    assert vision_log.isEnabledFor(logging.INFO)
