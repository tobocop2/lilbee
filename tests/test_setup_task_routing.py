"""Setup wizard: Enter-on-card installs via TaskBarController.

After the Bucket 2 UX redesign there's no Install & Go button: pressing
Enter on a model card (which fires ``GridSelect.Selected``) routes
directly to ``_commit_selection``, which writes settings and submits
the download to the app-level controller.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import ComposeResult
from textual.widgets import Footer

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.setup import SetupWizard
from lilbee.cli.tui.widgets.grid_select import GridSelect
from lilbee.cli.tui.widgets.model_card import ModelCard
from tests._lilbee_app_test_host import LilbeeAppHost


def _patch_setup_scan(chat: list[str] | None = None, embed: list[str] | None = None):
    return patch(
        "lilbee.cli.tui.screens.setup._scan_installed_models",
        return_value=(chat or [], embed or []),
    )


def _patch_setup_ram(ram_gb: float = 16.0):
    return patch("lilbee.modelhub.models.get_system_ram_gb", return_value=ram_gb)


class _PlainApp(LilbeeAppHost):
    """Minimal host so the wizard can mount without LilbeeApp's auto-wizard."""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(SetupWizard())


@pytest.mark.asyncio
async def test_enter_on_non_installed_chat_card_submits_download() -> None:
    """Enter on a non-installed card submits to TaskBarController.start_download."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(10):
                await pilot.pause()
                if isinstance(app.screen, SetupWizard):
                    break
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            chat_cards = [c for c in wizard.query(ModelCard) if c.row.task == "chat"]
            assert chat_cards
            first = chat_cards[0]
            assert not first.row.installed
            mock_grid = GridSelect()
            with (
                patch.object(app.task_bar, "start_download", return_value="tid") as mock_start,
                patch("lilbee.app.settings.persistent_settings.update_values"),
            ):
                wizard._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=first))
            mock_start.assert_called_once()


@pytest.mark.asyncio
async def test_non_installed_card_defers_apply_until_download_finishes() -> None:
    """Picking a not-yet-downloaded card writes cfg only after on_success fires."""
    from lilbee.core.config import cfg

    app = LilbeeApp()
    chat_default = cfg.chat_model
    captured: dict[str, object] = {}
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(10):
                await pilot.pause()
                if isinstance(app.screen, SetupWizard):
                    break
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            chat_cards = [c for c in wizard.query(ModelCard) if c.row.task == "chat"]
            first = chat_cards[0]
            assert not first.row.installed
            mock_grid = GridSelect()

            def _capture(_pending, **kwargs):
                captured["on_success"] = kwargs.get("on_success")
                return "tid"

            try:
                with (
                    patch.object(app.task_bar, "start_download", side_effect=_capture),
                    patch("lilbee.app.settings.persistent_settings.update_values"),
                    patch.object(wizard, "_apply_selection") as mock_apply,
                    patch(
                        "lilbee.cli.tui.screens.setup.call_from_thread",
                        side_effect=lambda _node, fn, *a, **kw: fn(*a, **kw),
                    ),
                ):
                    wizard._on_grid_selected(
                        GridSelect.Selected(grid_select=mock_grid, widget=first)
                    )
                    assert cfg.chat_model == chat_default
                    mock_apply.assert_not_called()
                    on_success = captured["on_success"]
                    assert callable(on_success)
                    on_success()  # simulate the worker firing the post-download hook
                    mock_apply.assert_called_once()
            finally:
                cfg.chat_model = chat_default


@pytest.mark.asyncio
async def test_enter_on_installed_card_does_not_submit_download() -> None:
    """Installed cards save config but skip start_download (nothing to fetch)."""
    from lilbee.catalog import FEATURED_CHAT

    app = LilbeeApp()
    installed_chat = [FEATURED_CHAT[0].ref]
    with _patch_setup_scan(chat=installed_chat), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(10):
                await pilot.pause()
                if isinstance(app.screen, SetupWizard):
                    break
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            installed_cards = [c for c in wizard.query(ModelCard) if c.row.installed]
            assert installed_cards
            chosen = installed_cards[0]
            mock_grid = GridSelect()
            with (
                patch.object(app.task_bar, "start_download") as mock_start,
                patch("lilbee.app.settings.persistent_settings.update_values"),
            ):
                wizard._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=chosen))
            mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_enter_does_not_resubmit_same_model_twice() -> None:
    """Re-selecting the same card doesn't double-enqueue the download."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(10):
                await pilot.pause()
                if isinstance(app.screen, SetupWizard):
                    break
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            chat_cards = [c for c in wizard.query(ModelCard) if c.row.task == "chat"]
            first = chat_cards[0]
            mock_grid = GridSelect()
            with (
                patch.object(app.task_bar, "start_download", return_value="tid") as mock_start,
                patch("lilbee.app.settings.persistent_settings.update_values"),
            ):
                wizard._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=first))
                wizard._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=first))
            assert mock_start.call_count == 1


@pytest.mark.asyncio
async def test_enter_noop_outside_lilbee_app() -> None:
    """Without a TaskBarController, Enter on a card still records the selection."""
    app = _PlainApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            chat_cards = [c for c in wizard.query(ModelCard) if c.row.task == "chat"]
            first = chat_cards[0]
            mock_grid = GridSelect()
            with patch("lilbee.app.settings.persistent_settings.update_values"):
                wizard._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=first))
            assert first.selected is True


@pytest.mark.asyncio
async def test_commit_selection_with_no_ref_returns_early() -> None:
    """Defensive: _commit_selection bails out if _mark_selection left no ref."""
    from lilbee.catalog.types import ModelTask

    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(10):
                await pilot.pause()
                if isinstance(app.screen, SetupWizard):
                    break
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            chat_cards = [c for c in wizard.query(ModelCard) if c.row.task == "chat"]
            first = chat_cards[0]

            # Stub _mark_selection to not populate _selections so ref stays None.
            def _stub(card, task):
                wizard._selections[task] = (None, None)

            with (
                patch.object(wizard, "_mark_selection", side_effect=_stub),
                patch.object(app.task_bar, "start_download") as mock_start,
                patch("lilbee.app.settings.persistent_settings.update_values") as mock_set,
            ):
                wizard._commit_selection(first, ModelTask.CHAT)
            mock_start.assert_not_called()
            mock_set.assert_not_called()


@pytest.mark.asyncio
async def test_escape_without_selection_dismisses_skipped() -> None:
    """Esc with no selections → dismiss('skipped')."""
    app = _PlainApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            from lilbee.catalog.types import ModelTask

            wizard._selections[ModelTask.CHAT] = (None, None)
            wizard._selections[ModelTask.EMBEDDING] = (None, None)
            wizard.action_cancel()
            for _ in range(5):
                await pilot.pause()
                if not isinstance(app.screen, SetupWizard):
                    break
            assert not isinstance(app.screen, SetupWizard)


@pytest.mark.asyncio
async def test_escape_with_selection_dismisses_completed() -> None:
    """Esc after any pick → dismiss('completed') + reset services."""
    app = _PlainApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            # Preselection already filled _selections; action_cancel should
            # treat that as "user committed to something".
            with patch("lilbee.cli.tui.screens.setup.reset_services") as mock_reset:
                wizard.action_cancel()
                for _ in range(5):
                    await pilot.pause()
                    if not isinstance(app.screen, SetupWizard):
                        break
            assert not isinstance(app.screen, SetupWizard)
            mock_reset.assert_called_once()
