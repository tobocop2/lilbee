"""The CLI status output lists held-out files, matching /api/status."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from lilbee.app.status import StatusConfig, StatusResult
from lilbee.cli.helpers import render_status_result
from lilbee.data.types import SkippedSource


def _status(skipped: list[SkippedSource] | None = None, skipped_total: int = 0) -> StatusResult:
    return StatusResult(
        document_count=0,
        config=StatusConfig(
            documents_dir="docs",
            data_dir="data",
            chat_model="chat:latest",
            embedding_model="embed:latest",
        ),
        sources=[],
        total_chunks=0,
        skipped=skipped or [],
        skipped_total=skipped_total,
    )


def _texts(status: StatusResult) -> tuple[list[Table], list[str]]:
    tables = [r for r in render_status_result(status) if isinstance(r, Table)]
    strings = [
        r.plain if isinstance(r, Text) else r
        for r in render_status_result(status)
        if isinstance(r, (str, Text))
    ]
    return tables, strings


def test_held_out_files_are_listed_with_their_reasons() -> None:
    status = _status(
        skipped=[SkippedSource(filename="notes/Solo.md", reason="no text extracted (0 chunks)")],
        skipped_total=1,
    )
    tables, strings = _texts(status)
    assert any(t.title == "Held out of the index" for t in tables)
    assert any("1" in s and "held out" in s and "--retry-skipped" in s for s in strings)


def test_a_bracketed_filename_and_reason_render_literally() -> None:
    """The held-out table must not treat a filename or reason as markup, and
    a Windows path's backslash before a bracket must survive."""
    status = _status(
        skipped=[SkippedSource(filename="notes\\[draft].md", reason="unsupported: [scan]")],
        skipped_total=1,
    )
    tables, _strings = _texts(status)
    table = next(t for t in tables if t.title == "Held out of the index")
    filename_cell, reason_cell = (col._cells[0] for col in table.columns)
    assert filename_cell.plain == "notes\\[draft].md"
    assert reason_cell.plain == "unsupported: [scan]"


def test_the_held_out_summary_names_what_the_cap_hid() -> None:
    status = _status(
        skipped=[SkippedSource(filename="a.md", reason="no text extracted (0 chunks)")],
        skipped_total=12,
    )
    _tables, strings = _texts(status)
    assert any("12" in s and "11 more not shown" in s for s in strings)


def test_nothing_held_out_prints_no_table() -> None:
    tables, _strings = _texts(_status())
    assert not any(t.title == "Held out of the index" for t in tables)


def test_the_embedder_that_built_the_index_is_shown() -> None:
    from lilbee.app.status import IndexStatus

    status = _status()
    status.index = IndexStatus(embedding_model="old:latest", embedding_dim=768)
    _tables, strings = _texts(status)
    assert any("old:latest" in s and "768" in s for s in strings)


def test_no_index_line_before_the_first_sync() -> None:
    _tables, strings = _texts(_status())
    assert not any("Index built with" in s for s in strings)


def test_a_bracketed_documents_dir_and_model_ref_render_literally() -> None:
    """Every config value on the status header is user-configured and may carry a bracket."""
    status = _status()
    status.config.documents_dir = "notes/[draft]"
    status.config.chat_model = "org/model[q4].gguf"
    _tables, strings = _texts(status)
    assert any("notes/[draft]" in s for s in strings)
    assert any("org/model[q4].gguf" in s for s in strings)


def test_the_ocr_off_warning_is_printed_when_status_carries_one() -> None:
    status = _status()
    status.ocr_warning = "OCR is off (enable_ocr = false), so the vision model v is not used."
    _tables, strings = _texts(status)
    assert any("OCR is off" in s for s in strings)
    assert not any("OCR is off" in s for s in _texts(_status())[1])
