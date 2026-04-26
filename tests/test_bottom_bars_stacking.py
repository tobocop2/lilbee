"""Every row inside BottomBars must land at its own y coordinate.

Sibling dock-bottom widgets collide at the same edge row in Textual,
so the bottom stack has to live inside a single BottomBars container
that lays children out vertically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Footer

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui.widgets.bottom_bars import BottomBars
from lilbee.cli.tui.widgets.status_bar import ViewTabs
from lilbee.cli.tui.widgets.task_bar import TaskBar, TaskBarController
from lilbee.config import cfg
from lilbee.services import set_services


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
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
        patch("lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount"),
    ):
        yield


class _ControllerApp(App[None]):
    """Test harness that installs a real TaskBarController + a single screen."""

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
    from lilbee.cli.tui.screens.chat import ChatScreen

    return ChatScreen()


def _catalog_screen():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    return CatalogScreen()


def _task_center_screen():
    from lilbee.cli.tui.screens.task_center import TaskCenter

    return TaskCenter()


async def test_chat_screen_taskbar_row_is_distinct_from_footer() -> None:
    """TaskBar must render on its own row, not overlapping with Footer.

    Before the BottomBars fix, TaskBar + ViewTabs + Footer were all
    sibling dock-bottom widgets and collided at the same y -- only
    Footer was visible.
    """
    app = _ControllerApp(_chat_screen)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.task_bar.add_task("Sync documents", "sync")
        app.task_bar.queue.advance()
        bar = app.screen.query_one(TaskBar)
        bar._refresh_display()
        await pilot.pause()

        assert bar.display is True
        assert bar.region.height >= 1

        view_tabs = app.screen.query_one(ViewTabs)
        footer = app.screen.query_one(Footer)

        # Each bottom-bar row must occupy a distinct y. If dock-bottom
        # siblings overlap, they share one y and the user sees only
        # the last one composed.
        ys = {bar.region.y, view_tabs.region.y, footer.region.y}
        assert len(ys) == 3, (
            f"TaskBar/ViewTabs/Footer must not overlap; ys={ys}, "
            f"task_bar={bar.region}, view_tabs={view_tabs.region}, footer={footer.region}"
        )
        # Natural stacking order: TaskBar above ViewTabs above Footer.
        assert bar.region.y < view_tabs.region.y < footer.region.y


async def test_chat_screen_taskbar_does_not_overlap_prompt_area() -> None:
    """The TaskBar row must sit below #chat-prompt-area, not inside it.

    Before the fix, #chat-prompt-area docked bottom independently of
    the three-widget stack, overlapping TaskBar+ViewTabs+Footer on the
    same row.
    """
    app = _ControllerApp(_chat_screen)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Force TaskBar visible so its region is populated.
        app.task_bar.add_task("Sync documents", "sync")
        app.task_bar.queue.advance()
        bar = app.screen.query_one(TaskBar)
        bar._refresh_display()
        await pilot.pause()
        prompt_area = app.screen.query_one("#chat-prompt-area")
        # TaskBar row's y must be below prompt area's bottom row.
        prompt_bottom = prompt_area.region.y + prompt_area.region.height - 1
        assert bar.region.y > prompt_bottom, (
            f"TaskBar ({bar.region}) must sit below prompt area ({prompt_area.region})"
        )


@pytest.mark.parametrize(
    "factory",
    [_chat_screen, _catalog_screen, _task_center_screen],
    ids=["chat", "catalog", "task_center"],
)
async def test_bottom_bars_wraps_all_bottom_widgets(factory) -> None:
    """Every screen mounts exactly one BottomBars holding the bottom stack."""
    app = _ControllerApp(factory)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        containers = list(app.screen.query(BottomBars))
        assert len(containers) == 1, (
            f"{factory.__name__} should mount exactly one BottomBars, found {len(containers)}"
        )
        container = containers[0]
        # The TaskBar and Footer must both live inside it.
        assert container.query(TaskBar)
        assert container.query(Footer)
