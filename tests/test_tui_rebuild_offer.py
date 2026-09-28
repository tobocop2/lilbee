"""The Settings screen offers a rebuild after a save or reset that changes the index."""

from __future__ import annotations

from unittest import mock

import pytest
from textual.pilot import Pilot
from textual.widgets import Input, Label, Static

from lilbee.app.settings import apply_settings_update
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog, ConfirmPill
from lilbee.core.config import cfg
from tests._async_wait import press_widget, wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests._profile_fixtures import (
    gate_releases_at_once,  # noqa: F401 -- autouse fixture, applied by import
    isolated_cfg,  # noqa: F401 -- autouse fixture, applied by import
)
from tests.test_tui_profile_tab import _dialog, _loaded, _press, _SettingsApp, _text, _until

# A dialog that is coming is pushed within a few message-loop ticks of the save.
_ABSENT_PAUSES = 30
_REBUILD_KEY = "chunk_size"
_PLAIN_KEY = "top_k"


async def _editors(app: LilbeeAppHost, pilot: Pilot) -> SettingsScreen:
    """Open every settings pane and wait for the two editors these tests use."""
    screen = await _loaded(app, pilot, "Default")
    screen.populate_all_panes()
    assert await _until(
        pilot,
        lambda: bool(screen.query(f"#ed-{_REBUILD_KEY}") and screen.query(f"#ed-{_PLAIN_KEY}")),
    )
    return screen


async def _reset(pilot: Pilot, screen: SettingsScreen, key: str) -> None:
    """Focus *key*'s editor and press the reset key on it."""
    await press_widget(pilot, screen.query_one(f"#ed-{key}", Input), "ctrl+r", max_pauses=300)


async def _answer_offer(app: LilbeeAppHost, pilot: Pilot, pill_id: str) -> None:
    """Wait for the rebuild offer, check what it says, and answer it with *pill_id*."""
    dialog = await _dialog(app, pilot, ConfirmDialog)
    assert _text(dialog.query_one("#confirm-title", Static)) == msg.CMD_REBUILD_CONFIRM_TITLE
    message = dialog.query_one("#confirm-message", Label)
    assert str(message.render()) == msg.SETTINGS_REINDEX_MESSAGE
    await _press(pilot, dialog.query_one(f"#{pill_id}", ConfirmPill))


async def _no_offer(app: LilbeeAppHost, pilot: Pilot) -> bool:
    return not await wait_until(
        pilot, lambda: isinstance(app.screen, ConfirmDialog), max_pauses=_ABSENT_PAUSES
    )


@pytest.mark.parametrize(("pill_id", "rebuilds"), [("confirm-yes", 1), ("confirm-no", 0)])
async def test_saving_a_rebuild_setting_offers_a_rebuild(pill_id: str, rebuilds: int) -> None:
    app = _SettingsApp()
    with mock.patch.object(LilbeeAppHost, "start_rebuild") as rebuild:
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _editors(app, pilot)
            field = screen.query_one(f"#ed-{_REBUILD_KEY}", Input)
            field.value = "900"
            await _press(pilot, field)
            await _answer_offer(app, pilot, pill_id)
            assert await _until(pilot, lambda: app.screen is screen)
    assert rebuild.call_count == rebuilds
    assert cfg.chunk_size == 900


@pytest.mark.parametrize(("pill_id", "rebuilds"), [("confirm-yes", 1), ("confirm-no", 0)])
async def test_resetting_a_rebuild_setting_offers_a_rebuild(pill_id: str, rebuilds: int) -> None:
    apply_settings_update({_REBUILD_KEY: 900})
    app = _SettingsApp()
    with mock.patch.object(LilbeeAppHost, "start_rebuild") as rebuild:
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _editors(app, pilot)
            await _reset(pilot, screen, _REBUILD_KEY)
            await _answer_offer(app, pilot, pill_id)
            assert await _until(pilot, lambda: app.screen is screen)
    assert rebuild.call_count == rebuilds
    assert cfg.chunk_size == 512


async def test_saving_a_setting_that_needs_no_rebuild_offers_none() -> None:
    app = _SettingsApp()
    with mock.patch.object(LilbeeAppHost, "start_rebuild") as rebuild:
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _editors(app, pilot)
            field = screen.query_one(f"#ed-{_PLAIN_KEY}", Input)
            field.value = "9"
            await _press(pilot, field)
            assert await _until(pilot, lambda: cfg.top_k == 9)
            assert await _no_offer(app, pilot)
    rebuild.assert_not_called()


async def test_resetting_a_setting_that_needs_no_rebuild_offers_none() -> None:
    apply_settings_update({_PLAIN_KEY: 9})
    app = _SettingsApp()
    with mock.patch.object(LilbeeAppHost, "start_rebuild") as rebuild:
        async with app.run_test(size=(120, 40)) as pilot:
            screen = await _editors(app, pilot)
            await _reset(pilot, screen, _PLAIN_KEY)
            assert await _until(pilot, lambda: cfg.top_k != 9)
            assert await _no_offer(app, pilot)
    rebuild.assert_not_called()
