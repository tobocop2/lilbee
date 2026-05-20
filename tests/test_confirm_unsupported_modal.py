"""Pilot tests for ConfirmUnsupportedModal."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.cli.tui.screens.confirm_unsupported import ConfirmUnsupportedModal
from tests._lilbee_app_test_host import LilbeeAppHost


class _App(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host", id="host")


async def test_modal_cancel_returns_false() -> None:
    async with _App().run_test(size=(120, 40)) as pilot:
        verdicts: list[bool] = []

        async def _await_modal() -> None:
            verdicts.append(
                await pilot.app.push_screen_wait(ConfirmUnsupportedModal(architecture="kimi_k2"))
            )

        worker = pilot.app.run_worker(_await_modal())
        await pilot.pause()
        await pilot.press("escape")
        await worker.wait()
        assert verdicts == [False]


async def test_modal_pull_anyway_returns_true() -> None:
    async with _App().run_test(size=(120, 40)) as pilot:
        verdicts: list[bool] = []

        async def _await_modal() -> None:
            verdicts.append(
                await pilot.app.push_screen_wait(ConfirmUnsupportedModal(architecture="kimi_k2"))
            )

        worker = pilot.app.run_worker(_await_modal())
        await pilot.pause()
        await pilot.press("y")
        await worker.wait()
        assert verdicts == [True]


async def test_modal_body_includes_architecture() -> None:
    async with _App().run_test(size=(120, 40)) as pilot:

        async def _await_modal() -> None:
            await pilot.app.push_screen_wait(ConfirmUnsupportedModal(architecture="kimi_k2"))

        worker = pilot.app.run_worker(_await_modal())
        await pilot.pause()
        body = pilot.app.screen.query_one("#confirm-unsupported-body", Static)
        text = str(body.content)
        assert "kimi_k2" in text
        await pilot.press("escape")
        await worker.wait()
