"""Catalog screen's _enqueue_download branches for UNSUPPORTED rows."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelCompat, ModelTask
from tests._lilbee_app_test_host import LilbeeAppHost


def _unsupported_model() -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/foo-GGUF",
        gguf_filename="*.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
        architecture="kimi_k2",
        compat=ModelCompat.UNSUPPORTED,
    )


def _supported_model() -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/llama-GGUF",
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


async def test_enqueue_unsupported_shows_modal_and_pulls_on_confirm() -> None:
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield Static("host")

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()

        downloads: list[tuple[str, bool]] = []

        def _record(model: CatalogModel, *, allow_unsupported: bool = False, **_kw: object) -> str:
            downloads.append((model.hf_repo, allow_unsupported))
            return "task-id"

        pilot.app.task_bar.start_download = _record  # type: ignore[method-assign]

        screen._enqueue_download(_unsupported_model())
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        assert downloads == [("acme/foo-GGUF", True)]


async def test_enqueue_unsupported_does_nothing_on_cancel() -> None:
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield Static("host")

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()

        downloads: list[tuple[str, bool]] = []

        def _record(model: CatalogModel, *, allow_unsupported: bool = False, **_kw: object) -> str:
            downloads.append((model.hf_repo, allow_unsupported))
            return "task-id"

        pilot.app.task_bar.start_download = _record  # type: ignore[method-assign]

        screen._enqueue_download(_unsupported_model())
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert downloads == []


async def test_enqueue_supported_proceeds_without_modal() -> None:
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    class _App(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield Static("host")

    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()

        downloads: list[tuple[str, bool]] = []

        def _record(model: CatalogModel, *, allow_unsupported: bool = False, **_kw: object) -> str:
            downloads.append((model.hf_repo, allow_unsupported))
            return "task-id"

        pilot.app.task_bar.start_download = _record  # type: ignore[method-assign]

        screen._enqueue_download(_supported_model())
        await pilot.pause()
        assert downloads == [("acme/llama-GGUF", False)]
