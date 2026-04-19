"""Setup wizard: Enter-on-card installs via TaskBarController.

After the Bucket 2 UX redesign there's no Install & Go button — pressing
Enter on a model card (which fires ``GridSelect.Selected``) routes
directly to ``_commit_selection``, which writes settings and submits
the download to the app-level controller.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Footer

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.setup import SetupWizard
from lilbee.cli.tui.widgets.grid_select import GridSelect
from lilbee.cli.tui.widgets.model_card import ModelCard


def _patch_setup_scan(chat: list[str] | None = None, embed: list[str] | None = None):
    return patch(
        "lilbee.cli.tui.screens.setup._scan_installed_models",
        return_value=(chat or [], embed or []),
    )


def _patch_setup_ram(ram_gb: float = 16.0):
    return patch("lilbee.models.get_system_ram_gb", return_value=ram_gb)


class _PlainApp(App[None]):
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
                patch("lilbee.settings.set_value"),
            ):
                wizard._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=first))
            mock_start.assert_called_once()


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
                patch("lilbee.settings.set_value"),
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
                patch("lilbee.settings.set_value"),
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
            with patch("lilbee.settings.set_value"):
                wizard._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=first))
            assert first.selected is True


@pytest.mark.asyncio
async def test_commit_selection_with_no_ref_returns_early() -> None:
    """Defensive: _commit_selection bails out if _mark_selection left no ref."""
    from lilbee.models import ModelTask

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
                patch("lilbee.settings.set_value") as mock_set,
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
            from lilbee.models import ModelTask

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
