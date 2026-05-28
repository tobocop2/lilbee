"""Tests for the left model rail and the shared apply_model_pick helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lilbee.cli.tui.widgets.model_pick import (
    _MODEL_KEY_TO_WORKER_ROLE,
    apply_model_pick,
    config_key_for_scope,
)
from lilbee.providers.worker.transport import WorkerRole
from tests._lilbee_app_test_host import LilbeeAppHost


def test_config_key_for_scope_round_trips_and_rejects_unknown() -> None:
    assert config_key_for_scope("chat") == "chat_model"
    assert config_key_for_scope("rerank") == "reranker_model"
    with pytest.raises(KeyError):
        config_key_for_scope("bogus")  # type: ignore[arg-type]


async def test_apply_model_pick_browse_unknown_key_is_noop() -> None:
    """An unknown key on the browse path logs and returns without navigating."""
    from lilbee.cli.tui.screens.model_picker import BROWSE_CATALOG_REF

    app = LilbeeAppHost()
    async with app.run_test(size=(80, 24)) as pilot:
        before = app.screen
        apply_model_pick(
            app.screen, key="not_a_model_key", ref=BROWSE_CATALOG_REF, on_done=lambda: None
        )
        await pilot.pause()
        assert app.screen is before  # no CatalogScreen pushed


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("chat_model", WorkerRole.CHAT),
        ("embedding_model", WorkerRole.EMBED),
        ("vision_model", WorkerRole.VISION),
        ("reranker_model", WorkerRole.RERANK),
    ],
)
def test_model_key_to_worker_role_covers_all_four(key: str, expected: WorkerRole) -> None:
    assert _MODEL_KEY_TO_WORKER_ROLE[key] is expected


async def test_apply_model_pick_persists_and_reloads_vision() -> None:
    """A vision pick writes the new ref and respawns the vision worker."""
    services_mock = MagicMock()
    services_mock.store.has_chunks.return_value = False
    app = LilbeeAppHost()
    async with app.run_test(size=(80, 24)) as _pilot:
        with (
            patch("lilbee.cli.tui.widgets.model_pick.apply_active_model") as mock_apply,
            patch(
                "lilbee.cli.tui.widgets.model_pick.get_services",
                return_value=services_mock,
            ),
        ):
            apply_model_pick(
                app.screen, key="vision_model", ref="hf:org/vlm-q4", on_done=lambda: None
            )
        mock_apply.assert_called_once()
        assert mock_apply.call_args.args[1:] == ("vision_model", "hf:org/vlm-q4")
        services_mock.reload_role.assert_called_once_with(WorkerRole.VISION)


async def test_apply_model_pick_none_is_cancel() -> None:
    """``ref is None`` is the Esc/cancel path: nothing persists, on_done never runs."""
    app = LilbeeAppHost()
    async with app.run_test(size=(80, 24)) as _pilot:
        done = MagicMock()
        with patch("lilbee.cli.tui.widgets.model_pick.apply_active_model") as mock_apply:
            apply_model_pick(app.screen, key="vision_model", ref=None, on_done=done)
        mock_apply.assert_not_called()
        done.assert_not_called()


async def test_apply_model_pick_empty_clears_nullable_field() -> None:
    """``ref=""`` for a nullable field (vision/rerank) writes the empty string."""
    services_mock = MagicMock()
    services_mock.store.has_chunks.return_value = False
    app = LilbeeAppHost()
    async with app.run_test(size=(80, 24)) as _pilot:
        with (
            patch("lilbee.cli.tui.widgets.model_pick.apply_active_model") as mock_apply,
            patch(
                "lilbee.cli.tui.widgets.model_pick.get_services",
                return_value=services_mock,
            ),
        ):
            apply_model_pick(app.screen, key="reranker_model", ref="", on_done=lambda: None)
        mock_apply.assert_called_once()
        assert mock_apply.call_args.args[1:] == ("reranker_model", "")


async def test_apply_model_pick_empty_ignored_for_non_nullable() -> None:
    """``ref=""`` for a non-nullable field (chat) is treated as an invalid no-op."""
    app = LilbeeAppHost()
    async with app.run_test(size=(80, 24)) as _pilot:
        with patch("lilbee.cli.tui.widgets.model_pick.apply_active_model") as mock_apply:
            apply_model_pick(app.screen, key="chat_model", ref="", on_done=lambda: None)
        mock_apply.assert_not_called()


async def test_picker_always_appends_browse_catalog_row() -> None:
    """Every picker (populated or empty) ends with a 'Browse catalog' row."""
    from lilbee.cli.tui.screens.model_picker import BROWSE_CATALOG_REF, _PickerOptions
    from lilbee.cli.tui.widgets.model_bar import ModelOption

    empty = _PickerOptions(options=[]).to_sections(search="")
    assert len(empty) == 1 and len(empty[0].rows) == 1
    assert empty[0].rows[-1].ref == BROWSE_CATALOG_REF

    populated = _PickerOptions(options=[ModelOption(label="model-a", ref="hf:org/a")]).to_sections(
        search=""
    )
    assert populated[0].rows[-1].ref == BROWSE_CATALOG_REF
    assert populated[0].rows[0].ref == "hf:org/a"


async def test_apply_model_pick_browse_ref_opens_catalog_for_vision() -> None:
    """Selecting the browse-catalog sentinel pushes CatalogScreen on Vision tab."""
    from textual.widgets import TabbedContent

    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import TAB_VISION
    from lilbee.cli.tui.screens.model_picker import BROWSE_CATALOG_REF

    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        done = MagicMock()
        with patch("lilbee.cli.tui.widgets.model_pick.apply_active_model") as mock_apply:
            apply_model_pick(app.screen, key="vision_model", ref=BROWSE_CATALOG_REF, on_done=done)
            await pilot.pause()
            await pilot.pause()
        mock_apply.assert_not_called()
        done.assert_not_called()
        assert isinstance(app.screen, CatalogScreen)
        assert app.screen.query_one("#catalog-tabs", TabbedContent).active == TAB_VISION


async def test_catalog_opens_focused_on_vision_tab() -> None:
    """CatalogScreen(focus_task=TAB_VISION) lands on the Vision tab, not Chat."""
    from textual.widgets import TabbedContent

    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import TAB_VISION

    app = LilbeeAppHost()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(CatalogScreen(focus_task=TAB_VISION))
        await pilot.pause()
        # call_after_refresh schedules the activation; let one more frame settle.
        await pilot.pause()
        assert app.screen.query_one("#catalog-tabs", TabbedContent).active == TAB_VISION


class _RailTestApp(LilbeeAppHost):
    """Minimal host that pushes only the model rail (no full chat screen)."""

    CSS = ""

    def compose(self):
        from textual.widgets import Footer

        from lilbee.cli.tui.widgets.model_rail import ModelRail

        yield ModelRail(id="rail-host")
        yield Footer()


async def test_rail_renders_four_roles_with_optional_off_by_default(monkeypatch) -> None:
    """A fresh rail has Chat/Embed active and Vision/Rerank in the off state."""
    from lilbee.cli.tui.widgets.model_rail import ModelRail, RoleRow
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "chat_model", "fake/chat-model")
    monkeypatch.setattr(cfg, "embedding_model", "fake/embed-model")
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "reranker_model", "")

    app = _RailTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rail = app.screen.query_one(ModelRail)
        rows = {row.scope: row for row in rail.query(RoleRow)}
        assert set(rows) == {"chat", "embed", "vision", "rerank"}
        assert rows["chat"].is_active
        assert rows["embed"].is_active
        assert not rows["vision"].is_active
        assert not rows["rerank"].is_active
        # The off rows carry the muted class and the hollow dot.
        assert rows["vision"].has_class("-off")
        assert rows["rerank"].has_class("-off")


async def test_rail_optional_row_lights_up_when_config_changes(monkeypatch) -> None:
    """Assigning a vision_model via the settings signal flips the row to active."""
    from lilbee.cli.tui.widgets.model_rail import ModelRail, RoleRow
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "vision_model", "")
    app = _RailTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rail = app.screen.query_one(ModelRail)
        vision = next(r for r in rail.query(RoleRow) if r.scope == "vision")
        assert not vision.is_active

        monkeypatch.setattr(cfg, "vision_model", "hf:org/vlm-q4")
        app.settings_changed_signal.publish(("vision_model", "hf:org/vlm-q4"))
        await pilot.pause()
        assert vision.is_active
        assert vision.has_class("-active")


async def test_rail_vision_picker_prepends_disable_row(monkeypatch) -> None:
    """Opening the vision (nullable) picker from the rail prepends the disable row."""
    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.model_picker import ModelPickerModal
    from lilbee.cli.tui.widgets.model_bar import ModelPickerButton
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "vision_model", "")
    app = _RailTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        vision_btn = app.screen.query_one("#rail-pick-vision", ModelPickerButton)
        vision_btn.open_picker()
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ModelPickerModal)
        labels = [o.label for o in modal._options.options]
        assert labels[0] == msg.MODEL_PICKER_DISABLE_LABEL


async def test_rail_shows_cloud_warning_for_cloud_chat_model(monkeypatch) -> None:
    """A cloud-routed chat model lights the rail's cloud-provider warning."""
    from lilbee.cli.tui.widgets.model_rail import _CLOUD_WARNING_ID, ModelRail
    from lilbee.core.config import cfg

    # Pick a real provider-prefixed ref so _cloud_provider_label resolves a label.
    monkeypatch.setattr(cfg, "chat_model", "gemini/gemini-2.0-flash")
    app = _RailTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        from textual.widgets import Static

        warning = app.screen.query_one(ModelRail).query_one(f"#{_CLOUD_WARNING_ID}", Static)
        assert warning.has_class("-visible")


async def test_rail_ignores_non_model_settings_signal() -> None:
    """A signal for a non-model key (e.g. chat_mode) is a no-op for the rail rows."""
    from lilbee.cli.tui.widgets.model_rail import ModelRail, RoleRow

    app = _RailTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        rail = app.screen.query_one(ModelRail)
        # Should not raise and should not change any row state.
        rail._on_settings_changed(("chat_mode", "search"))
        await pilot.pause()
        assert {r.scope for r in rail.query(RoleRow)} == {"chat", "embed", "vision", "rerank"}


class _ChatTestApp(LilbeeAppHost):
    """Test fixture that pushes the chat screen so the rail mounts in context."""

    CSS = ""

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


@pytest.fixture
def _seeded_models(monkeypatch):
    """Pre-populate chat/embedding and skip the SetupWizard pop so the rail is reachable."""
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "chat_model", "fake/chat-model")
    monkeypatch.setattr(cfg, "embedding_model", "fake/embed-model")
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(ChatScreen, "_needs_setup", lambda self: False)


async def test_chat_screen_mounts_with_rail_present(_seeded_models) -> None:
    """ChatScreen.compose places the rail to the left of the chat column."""
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.widgets.model_rail import ModelRail

    app = _ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        chat_screen = next(s for s in app.screen_stack if isinstance(s, ChatScreen))
        rail = chat_screen.query_one("#model-rail", ModelRail)
        assert rail.display is True


async def test_chat_screen_ctrl_b_toggles_rail(_seeded_models) -> None:
    """Ctrl+B hides the rail and sets the user-hidden flag so a later widen won't unhide it."""
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.widgets.model_rail import ModelRail

    app = _ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        chat_screen = next(s for s in app.screen_stack if isinstance(s, ChatScreen))
        rail = chat_screen.query_one("#model-rail", ModelRail)
        assert rail.display is True
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert rail.display is False
        assert chat_screen._rail_user_hidden is True
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert rail.display is True
        assert chat_screen._rail_user_hidden is False


async def test_chat_screen_auto_collapses_rail_on_narrow_terminal(_seeded_models) -> None:
    """on_resize hides the rail below _MIN_WIDTH_FOR_RAIL and restores it above."""
    from textual import events as _events
    from textual.geometry import Size

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.widgets.model_rail import ModelRail

    app = _ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        chat_screen = next(s for s in app.screen_stack if isinstance(s, ChatScreen))
        rail = chat_screen.query_one("#model-rail", ModelRail)
        assert rail.display is True
        chat_screen.on_resize(
            _events.Resize(
                size=Size(60, 40), virtual_size=Size(60, 40), container_size=Size(60, 40)
            )
        )
        assert rail.display is False
        # Widen back; the user did not manually hide, so the rail returns.
        chat_screen.on_resize(
            _events.Resize(
                size=Size(120, 40), virtual_size=Size(120, 40), container_size=Size(120, 40)
            )
        )
        assert rail.display is True
        # Manual hide is sticky against re-widening.
        chat_screen._rail_user_hidden = True
        rail.display = False
        chat_screen.on_resize(
            _events.Resize(
                size=Size(120, 40), virtual_size=Size(120, 40), container_size=Size(120, 40)
            )
        )
        assert rail.display is False


async def test_chat_on_resize_tolerates_missing_rail(_seeded_models) -> None:
    """on_resize is a no-op (no error) if the rail isn't mounted."""
    from textual import events as _events
    from textual.geometry import Size

    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.widgets.model_rail import ModelRail

    app = _ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        chat_screen = next(s for s in app.screen_stack if isinstance(s, ChatScreen))
        chat_screen.query_one("#model-rail", ModelRail).remove()
        await pilot.pause()
        # Must not raise NoMatches.
        chat_screen.on_resize(
            _events.Resize(
                size=Size(60, 40), virtual_size=Size(60, 40), container_size=Size(60, 40)
            )
        )


async def test_apply_model_pick_embed_with_chunks_pushes_confirm() -> None:
    """Embed swap against a populated store pushes ConfirmDialog before writing."""
    from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

    services_mock = MagicMock()
    services_mock.store.has_chunks.return_value = True
    app = LilbeeAppHost()
    async with app.run_test(size=(80, 24)) as pilot:
        with (
            patch("lilbee.cli.tui.widgets.model_pick.apply_active_model") as mock_apply,
            patch(
                "lilbee.cli.tui.widgets.model_pick.get_services",
                return_value=services_mock,
            ),
        ):
            apply_model_pick(
                app.screen,
                key="embedding_model",
                ref="hf:org/new-embed",
                on_done=lambda: None,
            )
            await pilot.pause()
            assert isinstance(app.screen, ConfirmDialog)
            mock_apply.assert_not_called()
