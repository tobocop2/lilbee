"""The CLI status output lists held-out files, matching /api/status."""

from __future__ import annotations

from rich.table import Table

from lilbee.app.status import StatusConfig, StatusResult
from lilbee.cli.helpers import render_status_result
from lilbee.data.types import SkippedSource


def _status(skipped: list[SkippedSource] | None = None, skipped_total: int = 0) -> StatusResult:
    return StatusResult(
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
    strings = [r for r in render_status_result(status) if isinstance(r, str)]
    return tables, strings


def test_held_out_files_are_listed_with_their_reasons() -> None:
    status = _status(
        skipped=[SkippedSource(filename="notes/Solo.md", reason="no text extracted (0 chunks)")],
        skipped_total=1,
    )
    tables, strings = _texts(status)
    assert any(t.title == "Held out of the index" for t in tables)
    assert any("1" in s and "held out" in s and "--retry-skipped" in s for s in strings)


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
