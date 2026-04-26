"""Screen navigation must fully repaint the bottom-bar row.

Cycling Chat -> Task Center -> Chat -> Task Center should leave no
leftover widgets from the previous screen.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Footer

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui.widgets.bottom_bars import BottomBars
from lilbee.cli.tui.widgets.model_bar import ModelBar
from lilbee.cli.tui.widgets.task_bar import TaskBar
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


async def test_chat_task_center_chat_task_center_cycle_has_no_leftovers() -> None:
    """The sequence chat -> task center -> chat -> task center must leave
    each screen cleanly mounted with no stale widgets from the previous
    screen."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)

        # Cycle: chat -> task center -> chat -> task center
        app.action_open_tasks()
        await pilot.pause()
        assert isinstance(app.screen, TaskCenter)

        app.switch_view("Chat")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)

        app.action_open_tasks()
        await pilot.pause()
        assert isinstance(app.screen, TaskCenter)

        # On the second Task Center render, no ModelBar should exist
        # (that's chat-only), and the bottom stack must have exactly
        # one TaskBar and one Footer.
        assert not app.screen.query(ModelBar)
        assert len(app.screen.query(TaskBar)) == 1
        assert len(app.screen.query(Footer)) == 1
        assert len(app.screen.query(BottomBars)) == 1


async def test_footer_row_is_not_shared_with_other_bottom_widgets() -> None:
    """Footer must have its own y coordinate on the task center too.

    Reproduces the 'm Modelsdone ? Help t Tasks' corruption where two
    dock-bottom widgets at the same y painted on top of each other.
    """
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.action_open_tasks()
        await pilot.pause()
        assert isinstance(app.screen, TaskCenter)

        footer = app.screen.query_one(Footer)
        task_bar = app.screen.query_one(TaskBar)
        # Show the task bar so it has a non-zero region.
        app.task_bar.add_task("Something", "sync")
        app.task_bar.queue.advance()
        task_bar._refresh_display()
        await pilot.pause()

        # Footer and TaskBar must sit on distinct rows now.
        assert footer.region.y != task_bar.region.y, (
            f"Footer ({footer.region}) and TaskBar ({task_bar.region}) must not overlap"
        )
