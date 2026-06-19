"""Pilot-driven tests for the ``/memories`` management screen."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.app import ComposeResult
from textual.widgets import DataTable, Footer

from lilbee.app.services import set_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.memories import MemoriesScreen
from lilbee.core.config import cfg
from lilbee.data.store import LOCAL_OWNER, MemoryKind, MemoryRow, MemorySource
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conftest import make_mock_services


def _row(
    text: str,
    *,
    memory_id: str = "id0",
    kind: MemoryKind = MemoryKind.FACT,
    shared: bool = False,
) -> MemoryRow:
    return MemoryRow(
        id=memory_id,
        owner=LOCAL_OWNER,
        shared=shared,
        kind=kind,
        source=MemorySource.MANUAL,
        text=text,
        vector=[0.1],
        created_at="t",
        updated_at="t",
    )


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    cfg.memory_enabled = True
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def store():
    store = MagicMock()
    store.get_memories.return_value = []
    store.update_memory.return_value = True
    store.delete_memory.return_value = True
    set_services(make_mock_services(store=store))
    yield store
    set_services(None)


@pytest.fixture
def notes(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(
        MemoriesScreen,
        "notify",
        lambda self, message, **kwargs: captured.append(message),
    )
    return captured


class MemoriesTestApp(LilbeeAppHost):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(MemoriesScreen())


def _table(app: MemoriesTestApp) -> DataTable:
    return app.screen.query_one("#memories-table", DataTable)


async def test_loads_memories_into_table(store):
    store.get_memories.return_value = [_row("uses rust"), _row("likes terse", memory_id="id1")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert _table(app).row_count == 2


async def test_empty_notifies(store, notes):
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert _table(app).row_count == 0
        assert msg.MEMORIES_EMPTY in notes


async def test_disabled_notifies(store, notes):
    cfg.memory_enabled = False
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert _table(app).row_count == 0
        assert msg.MEMORIES_DISABLED in notes
        store.get_memories.assert_not_called()


async def test_load_failure_notifies(store, notes):
    store.get_memories.side_effect = RuntimeError("boom")
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert any("boom" in n for n in notes)


async def test_delete_confirmed_removes_memory(store, notes):
    store.get_memories.return_value = [_row("uses rust")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # After delete the reload returns an empty list.
        store.get_memories.return_value = []
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        store.delete_memory.assert_called_once_with("id0", owner=LOCAL_OWNER)
        assert msg.MEMORIES_DELETED in notes


async def test_delete_missing_memory_reports_not_found(store, notes):
    store.get_memories.return_value = [_row("uses rust")]
    store.delete_memory.return_value = False
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        assert msg.MEMORIES_DELETE_NOT_FOUND in notes


async def test_delete_cancelled_keeps_memory(store):
    store.get_memories.return_value = [_row("uses rust")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
        store.delete_memory.assert_not_called()


async def test_toggle_shared_flips_flag(store, notes):
    store.get_memories.return_value = [_row("uses rust", shared=False)]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        store.update_memory.assert_called_once_with("id0", shared=True, owner=LOCAL_OWNER)
        assert msg.MEMORIES_SHARED_ON in notes


async def test_filter_narrows_rows(store):
    store.get_memories.return_value = [_row("uses rust"), _row("likes terse", memory_id="id1")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen._filter = "rust"
        app.screen._load_memories()
        await pilot.pause()
        assert _table(app).row_count == 1


async def test_actions_noop_on_empty_table(store):
    """Delete/toggle with no rows must not call the store."""
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.press("s")
        await pilot.pause()
        store.delete_memory.assert_not_called()
        store.update_memory.assert_not_called()


async def test_toggle_shared_noop_when_row_unknown(store):
    """A highlighted id missing from the loaded list short-circuits."""
    store.get_memories.return_value = [_row("uses rust")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen._memories = []  # table still has the row, cache does not
        await pilot.press("s")
        await pilot.pause()
        store.update_memory.assert_not_called()


async def test_delete_failure_notifies(store, notes, monkeypatch):
    from lilbee.cli.tui.screens import memories as memories_mod

    store.get_memories.return_value = [_row("uses rust")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(memories_mod, "forget", MagicMock(side_effect=RuntimeError("disk")))
        app.screen._do_delete("id0")
        await pilot.pause()
        assert any("disk" in n for n in notes)


async def test_shared_failure_notifies(store, notes, monkeypatch):
    from lilbee.cli.tui.screens import memories as memories_mod

    store.get_memories.return_value = [_row("uses rust")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(
            memories_mod, "set_memory_shared", MagicMock(side_effect=RuntimeError("boom"))
        )
        await pilot.press("s")
        await pilot.pause()
        assert any("boom" in n for n in notes)


async def test_escape_clears_search_then_backs_out(store):
    from textual.widgets import Input

    store.get_memories.return_value = [_row("uses rust")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        search = app.screen.query_one("#memories-search", Input)
        search.value = "ru"
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert search.value == ""
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, MemoriesScreen)


async def test_focus_search_and_vim_keys(store):
    from textual.widgets import Input

    store.get_memories.return_value = [_row("a"), _row("b", memory_id="id1")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press("slash")
        await pilot.pause()
        assert screen.query_one("#memories-search", Input).has_focus
        # Input focused: _table_or_none returns None, action bodies skip.
        screen.action_cursor_down()
        screen.action_cursor_up()
        screen.action_jump_top()
        screen.action_jump_bottom()
        _table(app).focus()
        await pilot.pause()
        screen.action_cursor_down()
        screen.action_cursor_up()
        screen.action_jump_bottom()
        screen.action_jump_top()
        await pilot.pause()


async def test_highlighted_id_handles_coordinate_error(store, monkeypatch):
    store.get_memories.return_value = [_row("uses rust")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        table = _table(app)
        monkeypatch.setattr(
            table, "coordinate_to_cell_key", MagicMock(side_effect=RuntimeError("no coord"))
        )
        assert app.screen._highlighted_id() is None


async def test_highlighted_id_handles_null_row_key(store, monkeypatch):
    from types import SimpleNamespace

    store.get_memories.return_value = [_row("uses rust")]
    app = MemoriesTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        table = _table(app)
        monkeypatch.setattr(
            table,
            "coordinate_to_cell_key",
            lambda _coord: (SimpleNamespace(value=None), None),
        )
        assert app.screen._highlighted_id() is None


def test_go_back_guarded_against_empty_stack():
    from unittest.mock import PropertyMock, patch

    screen = MemoriesScreen()
    fake_app = MagicMock()
    fake_app.screen_stack = [screen]
    with patch.object(MemoriesScreen, "app", new_callable=PropertyMock, return_value=fake_app):
        screen.action_go_back()
        fake_app.pop_screen.assert_not_called()
        fake_app.screen_stack = [object(), screen]
        screen.action_go_back()
        fake_app.pop_screen.assert_called_once()
