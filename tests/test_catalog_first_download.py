"""First download of a role becomes the active model; the welcome banner points there."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import LilbeeAppHost


class _App(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


def _chat_model(repo: str = "acme/llama-GGUF") -> CatalogModel:
    return CatalogModel(
        hf_repo=repo,
        gguf_filename="*.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
        architecture="llama",
        compat=ModelCompat.SUPPORTED,
    )


async def _enqueue_and_capture(pilot, screen, model):
    """Run _enqueue_download against a recording task bar; return its on_success."""
    captured: dict[str, object] = {}

    def _record(m, *, allow_unsupported=False, on_success=None, **_kw):
        captured["on_success"] = on_success
        return "task-id"

    pilot.app.task_bar.start_download = _record  # type: ignore[method-assign]
    screen._enqueue_download(model)
    await pilot.pause()
    return captured["on_success"]


async def test_first_chat_download_becomes_active_and_toasts() -> None:
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    original = cfg.chat_model
    cfg.chat_model = ""
    try:
        async with _App().run_test(size=(120, 40)) as pilot:
            screen = CatalogScreen()
            await pilot.app.push_screen(screen)
            await pilot.pause()
            model = _chat_model()
            on_success = await _enqueue_and_capture(pilot, screen, model)
            assert on_success is not None

            with (
                patch.object(LilbeeApp, "set_active_model") as set_active,
                patch.object(pilot.app, "notify") as notify,
            ):
                # on_success runs on the download worker thread, as in production.
                await asyncio.to_thread(on_success)
                await pilot.pause()
            set_active.assert_called_once_with("chat_model", model.ref)
            assert any(
                msg.CHAT_READY_TOAST in str(c.args[0]) for c in notify.call_args_list if c.args
            )
    finally:
        cfg.chat_model = original or ""


async def test_a_later_download_does_not_steal_the_role() -> None:
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    original = cfg.chat_model
    cfg.chat_model = "owner/existing-GGUF/existing.Q4_K_M.gguf"
    try:
        async with _App().run_test(size=(120, 40)) as pilot:
            screen = CatalogScreen()
            await pilot.app.push_screen(screen)
            await pilot.pause()
            on_success = await _enqueue_and_capture(pilot, screen, _chat_model())

            with patch.object(LilbeeApp, "set_active_model") as set_active:
                await asyncio.to_thread(on_success)
                await pilot.pause()
            set_active.assert_not_called()
    finally:
        cfg.chat_model = original or ""


async def test_welcome_banner_follows_chat_readiness() -> None:
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        banner = screen.query_one("#catalog-welcome", Static)

        pilot.app.chat_is_ready = False
        await pilot.pause()
        assert banner.display

        pilot.app.chat_is_ready = True
        await pilot.pause()
        assert not banner.display
