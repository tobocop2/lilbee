"""The status screen pages through every source as the user scrolls."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from textual.widgets import DataTable, Static

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens import status as status_screen
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.data.store import DOCUMENTS_PAGE_DEFAULT, SourceSummary
from tests._async_wait import wait_for_widget, wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conftest import make_mock_services

PAGE = DOCUMENTS_PAGE_DEFAULT
TOTAL = 2 * PAGE + PAGE // 2
SIZE = (120, 40)


def _record(i: int) -> dict:
    return {
        "filename": f"doc-{i:04d}.md",
        "file_hash": "h",
        "chunk_count": 3,
        "ingested_at": "2026-01-01",
    }


class _Library:
    """A list-backed stand-in for the store's paged listing; it records every read."""

    def __init__(self, total: int) -> None:
        self.records = [_record(i) for i in range(total)]
        self.reads: list[dict] = []
        self.gate: threading.Event | None = None
        self.gate_offset: int | None = None

    def get_sources(self, *, search=None, limit=None, offset=0):
        self.reads.append({"limit": limit, "offset": offset})
        if self.gate is not None and offset == self.gate_offset:
            self.gate.wait(timeout=10)
        end = len(self.records) if limit is None else offset + limit
        return self.records[offset:end]

    def count_sources(self, *, search=None):
        return len(self.records)


class _StatusApp(LilbeeAppHost):
    CSS = ""

    def on_mount(self) -> None:
        self.push_screen(StatusScreen())


@pytest.fixture(autouse=True)
def _no_held_out(monkeypatch):
    monkeypatch.setattr(status_screen, "held_out_sources", lambda: ([], 0))


def _install(library: _Library) -> None:
    store = MagicMock()
    store.get_sources.side_effect = library.get_sources
    store.count_sources.side_effect = library.count_sources
    store.source_summary.return_value = SourceSummary(0, 0, {}, [])
    set_services(make_mock_services(store=store))


@pytest.fixture(autouse=True)
def _reset_services():
    yield
    set_services(None)


def _names(table: DataTable) -> list[str]:
    return [str(table.get_row_at(i)[0]) for i in range(table.row_count)]


def _hint(app) -> Static:
    return app.screen.query_one("#docs-more", Static)


async def _loaded(pilot, table: DataTable, rows: int) -> None:
    assert await wait_until(pilot, lambda: table.row_count == rows, timeout=5.0)


async def test_scrolling_to_the_end_loads_each_page_until_every_source_shows():
    library = _Library(TOTAL)
    _install(library)
    app = _StatusApp()
    async with app.run_test(size=SIZE) as pilot:
        table = await wait_for_widget(pilot, app, "#docs-table", DataTable)
        await _loaded(pilot, table, PAGE)
        assert str(_hint(app).render()) == msg.STATUS_DOCS_MORE.format(count=PAGE)
        assert _hint(app).display is True

        table.scroll_end(animate=False)
        await _loaded(pilot, table, 2 * PAGE)
        assert str(_hint(app).render()) == msg.STATUS_DOCS_MORE.format(count=2 * PAGE)

        table.scroll_end(animate=False)
        await _loaded(pilot, table, TOTAL)
        assert await wait_until(pilot, lambda: _hint(app).display is False)
        assert _names(table) == [r["filename"] for r in library.records]
        storage = str(app.screen.query_one("#storage-info", Static).render())
        assert str(TOTAL) in storage


async def test_moving_the_cursor_near_the_end_loads_the_next_page():
    library = _Library(TOTAL)
    _install(library)
    app = _StatusApp()
    async with app.run_test(size=SIZE) as pilot:
        table = await wait_for_widget(pilot, app, "#docs-table", DataTable)
        await _loaded(pilot, table, PAGE)

        table.move_cursor(row=PAGE - 1)
        await _loaded(pilot, table, 2 * PAGE)


async def test_each_load_asks_for_one_page_never_all():
    library = _Library(TOTAL)
    _install(library)
    app = _StatusApp()
    async with app.run_test(size=SIZE) as pilot:
        table = await wait_for_widget(pilot, app, "#docs-table", DataTable)
        await _loaded(pilot, table, PAGE)
        table.scroll_end(animate=False)
        await _loaded(pilot, table, 2 * PAGE)
        table.scroll_end(animate=False)
        await _loaded(pilot, table, TOTAL)
    assert library.reads == [
        {"limit": PAGE, "offset": 0},
        {"limit": PAGE, "offset": PAGE},
        {"limit": PAGE, "offset": 2 * PAGE},
    ]


async def test_two_quick_scroll_events_start_one_load():
    library = _Library(TOTAL)
    library.gate = threading.Event()
    library.gate_offset = PAGE
    _install(library)
    app = _StatusApp()
    async with app.run_test(size=SIZE) as pilot:
        table = await wait_for_widget(pilot, app, "#docs-table", DataTable)
        await _loaded(pilot, table, PAGE)

        table.scroll_end(animate=False)
        await wait_until(pilot, lambda: len(library.reads) == 2)
        table.move_cursor(row=PAGE - 1)
        table.scroll_home(animate=False)
        table.scroll_end(animate=False)
        await pilot.pause()
        library.gate.set()
        await _loaded(pilot, table, 2 * PAGE)

    assert [r["offset"] for r in library.reads] == [0, PAGE]


async def test_an_empty_library_keeps_its_message_and_shows_no_hint():
    _install(_Library(0))
    app = _StatusApp()
    async with app.run_test(size=SIZE) as pilot:
        table = await wait_for_widget(pilot, app, "#docs-table", DataTable)
        assert await wait_until(
            pilot, lambda: table.row_count == 1 and msg.STATUS_DOCS_EMPTY in _names(table)
        )
        assert _hint(app).display is False


async def test_a_failed_first_read_keeps_its_message_and_shows_no_hint():
    store = MagicMock()
    store.count_sources.side_effect = OSError("disk")
    store.source_summary.side_effect = OSError("disk")
    set_services(make_mock_services(store=store))
    app = _StatusApp()
    async with app.run_test(size=SIZE) as pilot:
        table = await wait_for_widget(pilot, app, "#docs-table", DataTable)
        assert await wait_until(
            pilot, lambda: table.row_count == 1 and msg.STATUS_DOCS_LOAD_FAILED in _names(table)
        )
        assert _hint(app).display is False


async def test_a_failed_page_keeps_the_loaded_rows_and_says_so():
    library = _Library(TOTAL)
    _install(library)
    app = _StatusApp()
    async with app.run_test(size=SIZE) as pilot:
        table = await wait_for_widget(pilot, app, "#docs-table", DataTable)
        await _loaded(pilot, table, PAGE)

        def refuse(*, search=None, limit=None, offset=0):
            raise OSError("disk")

        get_services().store.get_sources.side_effect = refuse
        table.scroll_end(animate=False)
        assert await wait_until(
            pilot, lambda: str(_hint(app).render()) == msg.STATUS_DOCS_LOAD_FAILED
        )
        assert table.row_count == PAGE


async def test_a_page_that_comes_back_empty_ends_the_paging():
    library = _Library(TOTAL)
    _install(library)
    app = _StatusApp()
    async with app.run_test(size=SIZE) as pilot:
        table = await wait_for_widget(pilot, app, "#docs-table", DataTable)
        await _loaded(pilot, table, PAGE)
        library.records = library.records[:PAGE]
        table.scroll_end(animate=False)
        assert await wait_until(pilot, lambda: _hint(app).display is False)
        assert table.row_count == PAGE
