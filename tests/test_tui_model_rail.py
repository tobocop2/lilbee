"""Tests for the left model rail and the shared apply_model_pick helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lilbee.cli.tui.widgets.model_pick import (
    _MODEL_KEY_TO_WORKER_ROLE,
    apply_model_pick,
)
from lilbee.providers.worker.transport import WorkerRole
from tests._lilbee_app_test_host import LilbeeAppHost


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
    from lilbee.core.config import cfg

    from lilbee.cli.tui.widgets.model_rail import ModelRail, RoleRow

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
    from lilbee.core.config import cfg

    from lilbee.cli.tui.widgets.model_rail import ModelRail, RoleRow

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
