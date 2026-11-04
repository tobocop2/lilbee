"""The Settings screen offers one rebuild when the user leaves it, for the values now in force."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest import mock

import pytest
from textual.app import ComposeResult
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widgets import Footer, Input, Label, Static

from conftest import TEST_EMBED_REF
from lilbee.app.services import get_services
from lilbee.app.settings import apply_settings_update
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.profile_dialogs import ApplyProfileDialog
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog, ConfirmPill
from lilbee.core.config import cfg
from lilbee.data.store.types import HeldOut, HeldRecord
from tests import test_tui_embed_swap as embed_swap
from tests._async_wait import press_widget, wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests._profile_fixtures import (
    gate_releases_at_once,  # noqa: F401 -- autouse fixture, applied by import
    isolated_cfg,  # noqa: F401 -- autouse fixture, applied by import
)
from tests.test_tui_embed_swap import _NEW
from tests.test_tui_profile_tab import _dialog, _loaded, _pick, _press, _text, _until

# A dialog that is coming is pushed within a few message-loop ticks of the trigger.
_ABSENT_PAUSES = 30
_REBUILD_KEY = "chunk_size"
_PLAIN_KEY = "top_k"
_LIMIT_KEY = "max_chunks_per_file"
_BOTH_MESSAGE = f"{msg.SETTINGS_REINDEX_MESSAGE} {msg.PROFILE_APPLY_HELD_OUT}"
# The embedder-swap services fixture, by its name, for the swaps made from Settings.
services = embed_swap.services


@pytest.fixture
def a_file_held_out() -> None:
    """One file the chunk limit holds out of the index."""
    held = {"long.md": HeldRecord(HeldOut.FAILED, "hash", "14 chunks exceed the limit")}
    get_services().store.held_records.return_value = held


class _VisitApp(LilbeeAppHost):
    """Settings over a stand-in Chat view, so Escape goes back the way the app does."""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        self.install_screen(Screen(), name="chat")
        self.install_screen(SettingsScreen(), name="settings")
        self.push_screen("settings")


class _Visit:
    """Run a Settings visit with each pushed ConfirmDialog and each rebuild start counted."""

    def __init__(self) -> None:
        self.app = _VisitApp()
        self.dialogs: list[ConfirmDialog] = []
        self.cleanups: list[Callable[[], object]] = []

    async def __aenter__(self) -> _Visit:
        self._rebuild = mock.patch.object(LilbeeAppHost, "start_rebuild")
        self.rebuild = self._rebuild.start()
        self._context = self.app.run_test(size=(120, 40))
        self.pilot = await self._context.__aenter__()
        real_push = self.app.push_screen

        def _push(screen: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(screen, ConfirmDialog):
                self.dialogs.append(screen)
            return real_push(screen, *args, **kwargs)

        self._push = mock.patch.object(self.app, "push_screen", _push)
        self._push.start()
        self.screen = await _loaded(self.app, self.pilot, "Default")
        self.screen.populate_all_panes()
        assert await _until(
            self.pilot,
            lambda: bool(
                self.screen.query(f"#ed-{_REBUILD_KEY}") and self.screen.query(f"#ed-{_PLAIN_KEY}")
            ),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._push.stop()
        await self._context.__aexit__(*exc)
        for cleanup in self.cleanups:
            cleanup()
        self._rebuild.stop()

    async def save(self, key: str, value: str) -> None:
        """Type *value* into *key*'s editor and submit it."""
        field = self.screen.query_one(f"#ed-{key}", Input)
        field.value = value
        await _press(self.pilot, field)
        assert await _until(self.pilot, lambda: str(getattr(cfg, key)) == value)

    async def reset(self, key: str) -> None:
        """Press the reset key on *key*'s editor."""
        field = self.screen.query_one(f"#ed-{key}", Input)
        await press_widget(self.pilot, field, "ctrl+r", max_pauses=300)

    async def leave(self) -> None:
        """Press Escape until the Settings screen is gone, the way a user goes back."""
        app = self.app
        await self.pilot.press("escape")
        await self.pilot.pause()
        if app.screen is self.screen:
            await self.pilot.press("escape")
        assert await _until(self.pilot, lambda: self.screen not in app.screen_stack)

    async def no_dialog(self, shown: int = 0) -> bool:
        """True when no ConfirmDialog past the *shown* ones comes within the wait."""
        return not await wait_until(
            self.pilot, lambda: len(self.dialogs) > shown, max_pauses=_ABSENT_PAUSES
        )

    async def answer_offer(self, pill_id: str, text: str = msg.SETTINGS_REINDEX_MESSAGE) -> None:
        """Wait for the rebuild offer, check it says *text*, and answer it with *pill_id*."""
        dialog = await _dialog(self.app, self.pilot, ConfirmDialog)
        assert _text(dialog.query_one("#confirm-title", Static)) == msg.CMD_REBUILD_CONFIRM_TITLE
        message = dialog.query_one("#confirm-message", Label)
        assert str(message.render()) == text
        await _press(self.pilot, dialog.query_one(f"#{pill_id}", ConfirmPill))
        assert await _until(self.pilot, lambda: not isinstance(self.app.screen, ConfirmDialog))


@pytest.mark.parametrize(("pill_id", "rebuilds"), [("confirm-yes", 1), ("confirm-no", 0)])
async def test_one_edit_offers_one_rebuild_on_leaving_and_none_before(
    index_holds_chunks, pill_id: str, rebuilds: int
) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        assert await visit.no_dialog()
        await visit.leave()
        await visit.answer_offer(pill_id)
        assert len(visit.dialogs) == 1
        assert visit.rebuild.call_count == rebuilds


async def test_two_edits_offer_one_rebuild_on_leaving(index_holds_chunks) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        await visit.save(_REBUILD_KEY, "700")
        assert await visit.no_dialog()
        await visit.leave()
        await visit.answer_offer("confirm-yes")
        assert len(visit.dialogs) == 1
        assert visit.rebuild.call_count == 1
    assert cfg.chunk_size == 700


async def test_an_edit_then_discard_offers_no_rebuild(index_holds_chunks) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        await _press(visit.pilot, visit.screen.query_one("#profile-discard", ConfirmPill))
        assert await _until(visit.pilot, lambda: cfg.chunk_size == 512)
        await visit.leave()
        assert await visit.no_dialog()
    visit.rebuild.assert_not_called()


async def test_an_edit_then_a_reset_to_the_built_value_offers_no_rebuild(
    index_holds_chunks,
) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        await visit.reset(_REBUILD_KEY)
        assert await _until(visit.pilot, lambda: cfg.chunk_size == 512)
        await visit.leave()
        assert await visit.no_dialog()


async def test_a_reset_away_from_the_built_value_offers_one_rebuild_on_leaving(
    index_holds_chunks,
) -> None:
    apply_settings_update({_REBUILD_KEY: 900})
    async with _Visit() as visit:
        await visit.reset(_REBUILD_KEY)
        assert await _until(visit.pilot, lambda: cfg.chunk_size == 512)
        assert await visit.no_dialog()
        await visit.leave()
        await visit.answer_offer("confirm-no")
        assert len(visit.dialogs) == 1


async def test_no_edit_offers_no_rebuild(index_holds_chunks) -> None:
    async with _Visit() as visit:
        await visit.leave()
        assert await visit.no_dialog()


async def test_an_edit_that_needs_no_rebuild_offers_none(index_holds_chunks) -> None:
    async with _Visit() as visit:
        await visit.save(_PLAIN_KEY, "9")
        await visit.leave()
        assert await visit.no_dialog()


async def test_an_edit_on_an_empty_index_offers_no_rebuild(index_is_empty) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        await visit.leave()
        assert await visit.no_dialog()
    assert cfg.chunk_size == 900


async def test_a_second_visit_asks_only_about_its_own_edits(index_holds_chunks) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        await visit.leave()
        await visit.answer_offer("confirm-no")
        visit.app.switch_screen("settings")
        assert await _until(visit.pilot, lambda: visit.app.screen is visit.screen)
        await visit.leave()
        assert await visit.no_dialog(shown=1)


async def test_a_rebuild_started_during_the_visit_leaves_nothing_to_ask(
    index_holds_chunks,
) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        chat = mock.Mock()
        with mock.patch.object(visit.app, "chat_screen", return_value=chat):
            LilbeeApp.start_rebuild(visit.app)
        chat.run_sync.assert_called_once_with(force_rebuild=True)
        await visit.leave()
        assert await visit.no_dialog()


@pytest.mark.parametrize(
    ("edited", "message"), [(True, msg.SETTINGS_REINDEX_MESSAGE), (False, None)]
)
async def test_a_quit_from_settings_asks_nothing_and_names_an_owed_rebuild(
    index_holds_chunks, edited: bool, message: str | None
) -> None:
    async with _Visit() as visit:
        if edited:
            await visit.save(_REBUILD_KEY, "900")
        with mock.patch.object(visit.app, "exit") as exit_app:
            await visit.app.action_quit()
        exit_app.assert_called_once_with(message=message)
        assert await visit.no_dialog()


def _dialogs(app: LilbeeAppHost) -> list[ConfirmDialog]:
    return [s for s in app.screen_stack if isinstance(s, ConfirmDialog)]


async def _one_dialog_saying(app: LilbeeAppHost, pilot: Pilot, message: str) -> ConfirmDialog:
    """Wait for a dialog with *message* on top, and check it is the only one open."""

    def _shown() -> bool:
        top = app.screen
        return isinstance(top, ConfirmDialog) and (
            str(top.query_one("#confirm-message", Label).render()) == message
        )

    assert await _until(pilot, _shown), message
    assert len(_dialogs(app)) == 1
    return _dialogs(app)[0]


async def test_a_reset_that_turns_the_wiki_off_asks_the_wipe_then_the_rebuild_on_leaving(
    index_holds_chunks,
) -> None:
    apply_settings_update({_REBUILD_KEY: 900, "wiki": True})
    with (
        mock.patch("lilbee.cli.tui.screens.wiki.wiki_has_content", return_value=True),
        mock.patch("lilbee.cli.tui.screens.wiki.start_wiki_wipe") as wipe,
    ):
        async with _Visit() as visit:
            app, pilot = visit.app, visit.pilot
            field = visit.screen.query_one(f"#ed-{_PLAIN_KEY}", Input)
            await press_widget(pilot, field, "ctrl+shift+r", max_pauses=300)
            reset_all = await _one_dialog_saying(app, pilot, msg.SETTINGS_RESET_ALL_CONFIRM_MESSAGE)
            await _press(pilot, reset_all.query_one("#confirm-yes", ConfirmPill))
            wipe_offer = await _one_dialog_saying(app, pilot, msg.WIKI_WIPE_DISABLED_MESSAGE)
            await _press(pilot, wipe_offer.query_one("#confirm-no", ConfirmPill))
            assert await _until(pilot, lambda: app.screen is visit.screen)
            await visit.leave()
            await visit.answer_offer("confirm-no")
            visit.rebuild.assert_not_called()
    assert cfg.wiki is False
    wipe.assert_not_called()


@pytest.mark.parametrize(("pill_id", "rebuilds"), [("confirm-yes", 1), ("confirm-no", 0)])
async def test_dropping_a_limit_that_holds_a_file_out_offers_the_retry_once_on_leaving(
    index_holds_chunks, a_file_held_out, pill_id: str, rebuilds: int
) -> None:
    async with _Visit() as visit:
        await visit.save(_LIMIT_KEY, "0")
        assert await visit.no_dialog()
        await visit.leave()
        await visit.answer_offer(pill_id, msg.PROFILE_APPLY_HELD_OUT)
        assert len(visit.dialogs) == 1
        assert visit.rebuild.call_count == rebuilds


async def test_a_discard_that_drops_a_limit_offers_the_retry_once_on_leaving(
    index_holds_chunks, a_file_held_out
) -> None:
    apply_settings_update({_LIMIT_KEY: 10})
    async with _Visit() as visit:
        await _press(visit.pilot, visit.screen.query_one("#profile-discard", ConfirmPill))
        assert await _until(visit.pilot, lambda: cfg.max_chunks_per_file == 3000)
        assert await visit.no_dialog()
        await visit.leave()
        await visit.answer_offer("confirm-yes", msg.PROFILE_APPLY_HELD_OUT)
        assert len(visit.dialogs) == 1
        assert visit.rebuild.call_count == 1


async def test_an_index_value_and_a_limit_changed_give_one_dialog_for_both(
    index_holds_chunks, a_file_held_out
) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        await visit.save(_LIMIT_KEY, "0")
        await visit.leave()
        await visit.answer_offer("confirm-yes", _BOTH_MESSAGE)
        assert await visit.no_dialog(shown=1)
        assert visit.rebuild.call_count == 1


async def test_a_held_out_file_with_no_change_offers_no_retry(
    index_holds_chunks, a_file_held_out
) -> None:
    async with _Visit() as visit:
        await visit.leave()
        assert await visit.no_dialog()


async def test_a_tighter_limit_admits_no_held_out_file_and_offers_no_retry(
    index_holds_chunks, a_file_held_out
) -> None:
    async with _Visit() as visit:
        await visit.save(_LIMIT_KEY, "10")
        await visit.leave()
        assert await visit.no_dialog()


async def test_a_quit_after_dropping_a_limit_names_the_retry(
    index_holds_chunks, a_file_held_out
) -> None:
    async with _Visit() as visit:
        await visit.save(_LIMIT_KEY, "0")
        with mock.patch.object(visit.app, "exit") as exit_app:
            await visit.app.action_quit()
        exit_app.assert_called_once_with(message=msg.PROFILE_APPLY_HELD_OUT)
        # The quit ended the visit: nothing is left to ask.
        assert visit.screen.leave() is None


async def _press_unfocused(visit: _Visit, key: str) -> None:
    """Press *key* with no editor focused, the way a user browses the screen."""
    visit.app.set_focus(None)
    await visit.pilot.pause()
    await visit.pilot.press(key)


async def _leave_by_escape(visit: _Visit) -> None:
    await visit.leave()


async def _leave_by_palette(visit: _Visit) -> None:
    from textual.command import CommandList, CommandPalette

    app = visit.app
    app.action_command_palette()
    assert await _until(visit.pilot, lambda: isinstance(app.screen, CommandPalette))
    await visit.pilot.press(*"task center")
    assert await _until(visit.pilot, lambda: app.screen.query_one(CommandList).option_count > 0)
    await visit.pilot.press("down", "enter")


async def _leave_by_tab_click(visit: _Visit) -> None:
    await visit.pilot.click("#view-tab-chat")


def _leave_by_key(key: str) -> Any:
    async def _leave(visit: _Visit) -> None:
        await _press_unfocused(visit, key)

    _leave.__name__ = f"_leave_by_{key}"
    return _leave


async def _open_catalog(visit: _Visit) -> Screen:
    """Open the catalog from the Settings model field, the way its browse choice does."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.widgets.model_pick import _open_catalog_for_key

    _open_catalog_for_key(visit.screen, "chat_model")
    assert await _until(visit.pilot, lambda: isinstance(visit.app.screen, CatalogScreen))
    return visit.app.screen


async def _leave_through_catalog_back(visit: _Visit) -> None:
    await _open_catalog(visit)
    await _press_unfocused(visit, "escape")
    # The catalog goes back to the screen that opened it; the visit goes on.
    assert await _until(visit.pilot, lambda: visit.app.screen is visit.screen)
    assert await visit.no_dialog()
    await visit.leave()


async def _leave_through_catalog_pick(visit: _Visit) -> None:
    from tests._lilbee_app_test_host import patch_model_write

    catalog = await _open_catalog(visit)
    with patch_model_write():
        catalog._activate("chat_model", "other/model", "picked")  # type: ignore[attr-defined]
        await visit.pilot.pause()
    await _press_unfocused(visit, "c")


_LEAVE_PATHS = [
    _leave_by_escape,
    _leave_by_key("q"),
    _leave_by_key("left_square_bracket"),
    _leave_by_key("right_square_bracket"),
    _leave_by_key("c"),
    _leave_by_key("t"),
    _leave_by_key("m"),
    _leave_by_tab_click,
    _leave_by_palette,
    _leave_through_catalog_back,
    _leave_through_catalog_pick,
]


@pytest.mark.parametrize("leave", _LEAVE_PATHS, ids=lambda f: f.__name__.removeprefix("_leave_"))
async def test_every_way_out_of_settings_offers_one_rebuild_after_an_index_edit(
    index_holds_chunks, leave: Any
) -> None:
    async with _Visit() as visit:
        visit.app.active_view = "Settings"
        await visit.save(_REBUILD_KEY, "900")
        await leave(visit)
        assert await _until(visit.pilot, lambda: visit.screen not in visit.app.screen_stack)
        await visit.answer_offer("confirm-no")
        assert len(visit.dialogs) == 1


async def test_a_dialog_over_settings_asks_nothing(index_holds_chunks) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        visit.app.push_screen(ConfirmDialog("A question", "Over Settings"))
        await _press(
            visit.pilot,
            (await _dialog(visit.app, visit.pilot, ConfirmDialog)).query_one(
                "#confirm-no", ConfirmPill
            ),
        )
        assert await _until(visit.pilot, lambda: visit.app.screen is visit.screen)
        assert await visit.no_dialog(shown=1)
        await visit.leave()
        await visit.answer_offer("confirm-no")
        assert len(visit.dialogs) == 2


@pytest.mark.parametrize("key", ["left_square_bracket", "right_square_bracket"])
async def test_a_view_key_does_not_close_the_rebuild_offer_unanswered(
    index_holds_chunks, key: str
) -> None:
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        await visit.leave()
        dialog = await _dialog(visit.app, visit.pilot, ConfirmDialog)
        view = visit.app.active_view
        await visit.pilot.press(key)
        await visit.pilot.pause()
        assert visit.app.screen is dialog
        assert visit.app.active_view == view
        await visit.answer_offer("confirm-no")
        assert len(visit.dialogs) == 1
    visit.rebuild.assert_not_called()


def _rebuild_for_real(visit: _Visit) -> mock.Mock:
    """Run the app's own rebuild start over a stand-in Chat view; returns that view."""
    chat = mock.Mock()
    visit.rebuild.side_effect = lambda: LilbeeApp.start_rebuild(visit.app)
    patch = mock.patch.object(visit.app, "chat_screen", return_value=chat)
    patch.start()
    visit.cleanups.append(patch.stop)
    return chat


@pytest.mark.parametrize(("pill_id", "rebuilds"), [("apply-reindex", 1), ("apply-apply", 0)])
async def test_a_profile_apply_answers_the_rebuild_and_leaving_asks_nothing(
    index_holds_chunks, pill_id: str, rebuilds: int
) -> None:
    async with _Visit() as visit:
        chat = _rebuild_for_real(visit)
        _pick(visit.screen, "Research papers")
        dialog = await _dialog(visit.app, visit.pilot, ApplyProfileDialog)
        await _press(visit.pilot, dialog.query_one(f"#{pill_id}", ConfirmPill))
        await _loaded(visit.app, visit.pilot, "Research papers")
        assert cfg.table_extraction is True
        await visit.leave()
        assert await visit.no_dialog()
    assert chat.run_sync.call_count == rebuilds


async def _apply_research_papers_without_rebuild(visit: _Visit) -> None:
    """Pick the Research papers profile and press plain Apply on its rebuild question."""
    _pick(visit.screen, "Research papers")
    dialog = await _dialog(visit.app, visit.pilot, ApplyProfileDialog)
    await _press(visit.pilot, dialog.query_one("#apply-apply", ConfirmPill))
    await _loaded(visit.app, visit.pilot, "Research papers")


async def test_an_edit_before_a_plain_apply_is_still_offered_on_leaving(index_holds_chunks) -> None:
    """The profile's question names only its own rows, so the earlier edit stays owed."""
    async with _Visit() as visit:
        chat = _rebuild_for_real(visit)
        await visit.save(_REBUILD_KEY, "900")
        await _apply_research_papers_without_rebuild(visit)
        assert cfg.chunk_size == 900
        await visit.leave()
        await visit.answer_offer("confirm-no")
        assert await visit.no_dialog(shown=1)
    chat.run_sync.assert_not_called()


async def test_an_edit_after_a_plain_apply_is_offered_on_leaving(index_holds_chunks) -> None:
    async with _Visit() as visit:
        chat = _rebuild_for_real(visit)
        await _apply_research_papers_without_rebuild(visit)
        await visit.save(_REBUILD_KEY, "900")
        await visit.leave()
        await visit.answer_offer("confirm-no")
        assert await visit.no_dialog(shown=1)
    chat.run_sync.assert_not_called()


@pytest.mark.parametrize(("asked", "offers"), [(_REBUILD_KEY, 0), ("chunk_overlap", 1)])
async def test_an_offer_answered_no_covers_only_its_own_setting(
    index_holds_chunks, asked: str, offers: int
) -> None:
    """A save that asks about one setting leaves an earlier edit to another one owed."""
    async with _Visit() as visit:
        await visit.save(_REBUILD_KEY, "900")
        assert visit.app.set_setting(asked, 901 if asked == _REBUILD_KEY else 99)
        await visit.answer_offer("confirm-no")
        await visit.leave()
        assert await visit.no_dialog(shown=1) is (offers == 0)
        assert len(visit.dialogs) == 1 + offers
    visit.rebuild.assert_not_called()


@pytest.mark.parametrize(("pill_id", "rebuilds"), [("confirm-yes", 1), ("confirm-no", 0)])
async def test_an_embedder_swap_offered_after_its_write_is_asked_once_in_settings(
    services: mock.MagicMock, index_holds_chunks, pill_id: str, rebuilds: int
) -> None:
    """The picker finds no chunks and asks nothing; the offer after the write is the question."""
    services.store.has_chunks.return_value = False
    async with _Visit() as visit:
        chat = _rebuild_for_real(visit)
        visit.screen._on_model_picker_dismissed("embedding_model", _NEW)
        await visit.answer_offer(pill_id)
        assert cfg.embedding_model == _NEW
        await visit.leave()
        assert await visit.no_dialog(shown=1)
    assert chat.run_sync.call_count == rebuilds


@pytest.mark.parametrize(("pill_id", "rebuilds"), [("confirm-yes", 1), ("confirm-no", 0)])
async def test_an_embedder_swap_asked_before_its_write_is_asked_once_in_settings(
    services: mock.MagicMock, index_holds_chunks, pill_id: str, rebuilds: int
) -> None:
    async with _Visit() as visit:
        chat = _rebuild_for_real(visit)
        visit.screen._on_model_picker_dismissed("embedding_model", _NEW)
        await visit.answer_offer(pill_id, msg.EMBED_SWAP_REBUILD_MESSAGE)
        assert await _until(visit.pilot, lambda: chat.run_sync.call_count == rebuilds)
        await visit.app.workers.wait_for_complete()
        assert cfg.embedding_model == (_NEW if rebuilds else TEST_EMBED_REF)
        await visit.leave()
        assert await visit.no_dialog(shown=1)
    assert chat.run_sync.call_count == rebuilds


async def test_a_catalog_round_trip_leaves_no_settings_buried(index_holds_chunks) -> None:
    async with _Visit() as visit:
        visit.app.active_view = "Settings"
        base = visit.app.screen_stack[:-1]
        await _open_catalog(visit)
        await _press_unfocused(visit, "escape")
        assert await _until(visit.pilot, lambda: visit.app.screen is visit.screen)
        assert visit.app.screen_stack == [*base, visit.screen]
        await _open_catalog(visit)
        await _press_unfocused(visit, "c")
        assert await _until(visit.pilot, lambda: visit.screen not in visit.app.screen_stack)
        assert len(visit.app.screen_stack) == len(base) + 1
