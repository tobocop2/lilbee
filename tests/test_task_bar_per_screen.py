"""Regression tests for the per-screen TaskBar mount.

Every lilbee screen must compose its own `TaskBar` so background progress
(downloads, syncs, crawls) stays visible as the user navigates. Before this
fix the TaskBar was mounted once on the app and was hidden behind every
pushed screen.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Footer

from lilbee.catalog import CatalogResult
from lilbee.cli.tui.widgets import task_bar as task_bar_module
from lilbee.cli.tui.widgets.task_bar import TaskBar, TaskBarController
from lilbee.config import cfg
from lilbee.services import set_services

_EMPTY_CATALOG = CatalogResult(total=0, limit=25, offset=0, models=[])


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.chat_model = "test-model:latest"
    cfg.embedding_model = "test-embed:latest"
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def _mock_services():
    from tests.conftest import make_mock_services

    store = MagicMock()
    store.search.return_value = []
    store.bm25_probe.return_value = []
    store.get_sources.return_value = []
    store.add_chunks.side_effect = lambda records: len(records)
    set_services(make_mock_services(store=store))
    yield
    set_services(None)


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    with (
        patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=False,
        ),
        patch(
            "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
            return_value=([], []),
        ),
        patch(
            "lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount",
            return_value=None,
        ),
        patch(
            "lilbee.cli.tui.widgets.model_bar.ModelBar.refresh_models",
            return_value=None,
        ),
        patch(
            "lilbee.cli.tui.screens.catalog.CatalogScreen._fetch_remote_models",
            return_value=None,
        ),
        patch(
            "lilbee.cli.tui.screens.catalog.CatalogScreen._fetch_installed_names",
            return_value=None,
        ),
    ):
        yield


class _ControllerApp(App[None]):
    """Test harness that exposes a real TaskBarController plus a single screen."""

    CSS = ""

    def __init__(self, screen_factory) -> None:
        super().__init__()
        self.task_bar = TaskBarController(self)
        self._screen_factory = screen_factory

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(self._screen_factory())


def _chat_screen():
    """Minimal screen that mirrors ChatScreen's TaskBar without ModelBar."""
    from textual.screen import Screen
    from textual.widgets import Footer

    from lilbee.cli.tui.widgets.task_bar import TaskBar

    class _ChatStub(Screen[None]):
        def compose(self):
            yield TaskBar()
            yield Footer()

    return _ChatStub()


def _catalog_screen():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    return CatalogScreen()


def _settings_screen():
    from lilbee.cli.tui.screens.settings import SettingsScreen

    return SettingsScreen()


def _status_screen():
    from lilbee.cli.tui.screens.status import StatusScreen

    return StatusScreen()


def _task_center_screen():
    from lilbee.cli.tui.screens.task_center import TaskCenter

    return TaskCenter()


def _wiki_screen():
    from lilbee.cli.tui.screens.wiki import WikiScreen

    return WikiScreen()


def _setup_screen():
    from lilbee.cli.tui.screens.setup import SetupWizard

    return SetupWizard()


@pytest.mark.parametrize(
    "factory",
    [
        _chat_screen,
        _catalog_screen,
        _settings_screen,
        _status_screen,
        _task_center_screen,
        _wiki_screen,
        _setup_screen,
    ],
    ids=[
        "chat",
        "catalog",
        "settings",
        "status",
        "task_center",
        "wiki",
        "setup",
    ],
)
async def test_every_screen_mounts_a_task_bar(factory) -> None:
    """Each top-level screen must compose its own TaskBar instance."""
    app = _ControllerApp(factory)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bars = list(app.screen.query(TaskBar))
        assert len(bars) == 1, (
            f"{factory.__name__} should mount exactly one TaskBar, found {len(bars)}"
        )


async def test_task_bar_shows_active_task_on_catalog_screen() -> None:
    """A task added via the controller is rendered by the screen's TaskBar."""
    app = _ControllerApp(_catalog_screen)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bar = app.screen.query_one(TaskBar)
        assert bar.display is False  # idle: hidden
        task_id = app.task_bar.add_task("Download test-model", "download")
        app.task_bar.queue.advance()
        await pilot.pause()
        assert bar.display is True
        assert task_id in bar._panels


async def test_task_bar_state_shared_across_screens() -> None:
    """Switching screens keeps tasks visible because they share one queue."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = _ControllerApp(_chat_screen)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.task_bar.add_task("Background sync", "sync")
        app.task_bar.queue.advance()
        await pilot.pause()

        chat_bar = app.screen.query_one(TaskBar)
        assert chat_bar.display is True

        app.switch_screen(CatalogScreen())
        await pilot.pause()
        assert isinstance(app.screen, CatalogScreen)
        catalog_bar = app.screen.query_one(TaskBar)
        assert catalog_bar is not chat_bar
        assert catalog_bar.display is True
        # Same underlying queue → same active task
        assert catalog_bar.queue is chat_bar.queue
        assert catalog_bar.queue.active_task is not None
        assert catalog_bar.queue.active_task.name == "Background sync"


async def test_task_bar_auto_hides_when_queue_drains() -> None:
    """When all tasks finish, every screen's TaskBar should hide again."""
    app = _ControllerApp(_chat_screen)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        bar = app.screen.query_one(TaskBar)
        task_id = app.task_bar.add_task("Download", "download")
        app.task_bar.queue.advance()
        await pilot.pause()
        assert bar.display is True

        app.task_bar.complete_task(task_id)
        # Wait out the post-completion flash window before the panel is dropped.
        await pilot.pause(delay=task_bar_module._DONE_FLASH_SECONDS + 0.2)
        await pilot.pause()
        assert app.task_bar.queue.is_empty
        assert bar.display is False
