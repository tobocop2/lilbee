"""The Settings Engines tab asks before it removes the build that the embedder runs on."""

from __future__ import annotations

from unittest import mock

import pytest
from textual.widgets import Static

from lilbee.app import engine_service
from lilbee.cli.tui.engine_commands import EngineCommandResult
from lilbee.cli.tui.screens import settings_engines
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from tests._async_wait import wait_until
from tests.test_tui_settings_engines import (  # noqa: F401
    _config,
    _open_engines,
    _select_row,
    _SettingsApp,
    _toasts,
    service,
)

pytestmark = pytest.mark.usefixtures("service")


def _asks_then_removes(engine: str, version: str, backend: str, confirmed: bool = False) -> str:
    if not confirmed:
        raise engine_service.EmbedConfirmRequiredError("lib", 7)
    return "tei 0.9 (cpu)"


async def _embed_question_open(app: _SettingsApp, pilot) -> bool:
    """Wait for the second dialog, not the removal dialog that is still closing."""
    return await wait_until(
        pilot,
        lambda: (
            isinstance(app.screen, ConfirmDialog)
            and "Continue?" in str(app.screen.query_one("#confirm-message", Static).render())
        ),
    )


async def test_the_tab_asks_again_when_the_removal_changes_the_embedding_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remove = mock.Mock(side_effect=_asks_then_removes)
    monkeypatch.setattr(engine_service, "remove_build", remove)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "tei")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("y")
        assert await _embed_question_open(app, pilot)
        text = str(app.screen.query_one("#confirm-message", Static).render())
        assert text == (
            "Library lib holds 7 documents. tei 0.9 (cpu) runs the embedding model. "
            "Without it the vectors can become unusable by the index. Continue?"
        )
        assert remove.call_count == 1
        await pilot.press("y")
        assert await wait_until(pilot, lambda: remove.call_count == 2)
        assert remove.call_args_list[1] == mock.call("tei", "0.9", "cpu", confirmed=True)
        assert await wait_until(pilot, lambda: "Removed engine tei 0.9 (cpu)." in _toasts())


async def test_declining_the_embed_question_keeps_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remove = mock.Mock(side_effect=_asks_then_removes)
    monkeypatch.setattr(engine_service, "remove_build", remove)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "tei")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("y")
        assert await _embed_question_open(app, pilot)
        await pilot.press("n")
        assert await wait_until(pilot, lambda: "The engine is unchanged." in _toasts())
        assert remove.call_count == 1
        assert table.row_count == 4


async def test_declining_the_embed_question_reports_a_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_service, "remove_build", mock.Mock(side_effect=_asks_then_removes))
    shown: list[EngineCommandResult] = []
    monkeypatch.setattr(settings_engines, "announce", lambda _app, result: shown.append(result))
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot)
        await _select_row(pilot, table, "tei")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("y")
        assert await _embed_question_open(app, pilot)
        await pilot.press("n")
        assert await wait_until(pilot, lambda: bool(shown))
    assert [(r.ok, r.text) for r in shown] == [(False, "The engine is unchanged.")]
