"""Setup wizard hands downloads off to TaskBarController and dismisses immediately.

The wizard is a caller, not an owner. It must:
  - resolve non-installed selections,
  - call ``app.task_bar.start_download(model)`` for each,
  - persist config via ``_save_and_dismiss`` and pop itself.

Download progress reporting is tested in ``test_controller_downloads.py``;
here we only cover the wizard-as-caller contract.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Footer

from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.setup import SetupWizard


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
async def test_install_submits_pending_downloads_to_controller() -> None:
    """Install & Go routes non-installed selections to TaskBarController.start_download."""
    app = LilbeeApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            # ChatScreen auto-pushes SetupWizard when no models are installed.
            for _ in range(10):
                await pilot.pause()
                if isinstance(app.screen, SetupWizard):
                    break
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            with (
                patch.object(app.task_bar, "start_download", return_value="tid") as mock_start,
                patch("lilbee.settings.set_value"),
                patch("lilbee.cli.tui.screens.setup.reset_services"),
            ):
                wizard._on_install()
                await pilot.pause()
            # Both recommended models are non-installed; both should submit.
            assert mock_start.call_count == 2


@pytest.mark.asyncio
async def test_install_with_already_installed_selections_submits_nothing() -> None:
    """Selections whose cards have ``installed=True`` bypass start_download."""
    from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = LilbeeApp()
    with (
        _patch_setup_scan(chat=[FEATURED_CHAT[0].ref], embed=[FEATURED_EMBEDDING[0].ref]),
        _patch_setup_ram(),
    ):
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(10):
                await pilot.pause()
                if isinstance(app.screen, SetupWizard):
                    break
            wizard = app.screen
            assert isinstance(wizard, SetupWizard)
            cards = list(wizard.query(ModelCard))
            chat_card = next(
                c for c in cards if c.row.installed and c.row.task == FEATURED_CHAT[0].task
            )
            embed_card = next(
                c for c in cards if c.row.installed and c.row.task == FEATURED_EMBEDDING[0].task
            )
            wizard._select_card(chat_card, chat_card.row.task)
            wizard._select_card(embed_card, embed_card.row.task)
            with (
                patch.object(app.task_bar, "start_download") as mock_start,
                patch("lilbee.settings.set_value"),
                patch("lilbee.cli.tui.screens.setup.reset_services"),
            ):
                wizard._on_install()
                await pilot.pause()
            assert mock_start.call_count == 0


@pytest.mark.asyncio
async def test_install_outside_lilbee_app_saves_and_dismisses_without_downloads() -> None:
    """Without a TaskBarController, the wizard still saves config and pops itself."""
    app = _PlainApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            screen._on_install()  # must not raise without app.task_bar
            for _ in range(5):
                await pilot.pause()
                if not isinstance(app.screen, SetupWizard):
                    break
            assert not isinstance(app.screen, SetupWizard)


@pytest.mark.asyncio
async def test_escape_dismisses_without_affecting_shared_queue() -> None:
    """action_cancel pops the wizard but leaves queued downloads alone."""
    app = _PlainApp()
    with _patch_setup_scan(), _patch_setup_ram():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            screen.action_cancel()
            for _ in range(5):
                await pilot.pause()
                if not isinstance(app.screen, SetupWizard):
                    break
            assert not isinstance(app.screen, SetupWizard)
