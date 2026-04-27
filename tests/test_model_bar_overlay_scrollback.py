"""ModelBar Select overlay must not leak borders into terminal scrollback.

The overlay is capped to stay inside the screen, and collapsing it
forces a full screen refresh so the compositor invalidates the region.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui.widgets.model_bar import ModelBar
from lilbee.core.config import cfg


class _ModelBarApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ModelBar(id="model-bar")


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def _mock_classify():
    with patch(
        "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
        return_value=([], []),
    ):
        yield


def test_model_bar_caps_overlay_height_and_constrains_inside() -> None:
    """CSS caps overlay height and constrains it inside the screen.

    Without the cap, Textual's default ``max-height: 12`` + ``constrain:
    none inside`` let the overlay extend below the viewport, causing
    border cells to get pushed into the terminal's scrollback buffer
    when the overlay collapses.
    """
    css = ModelBar.DEFAULT_CSS
    assert "max-height: 8" in css
    assert "constrain: inside inside" in css


async def test_collapsing_select_refreshes_screen() -> None:
    """Collapsing a Select overlay triggers a full screen refresh.

    We expand then collapse the Select and assert that
    ``Screen.refresh`` was called at least once AFTER the collapse.
    Without this refresh on collapse, the overlay's border cells can
    linger in the terminal scrollback.
    """
    app = _ModelBarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        sel = app.query_one("#chat-model-select", Select)

        sel.expanded = True
        await pilot.pause()
        with patch.object(app.screen, "refresh") as mock_refresh:
            sel.expanded = False
            await pilot.pause()
            assert mock_refresh.called, (
                "collapsing the Select must force a screen refresh so "
                "stray overlay cells don't linger in terminal scrollback"
            )


async def test_both_selects_refresh_on_collapse() -> None:
    """Both chat and embed Selects must trigger refresh on collapse."""
    app = _ModelBarApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        chat_sel = app.query_one("#chat-model-select", Select)
        embed_sel = app.query_one("#embed-model-select", Select)

        chat_sel.expanded = True
        await pilot.pause()
        with patch.object(app.screen, "refresh") as mock_refresh:
            chat_sel.expanded = False
            await pilot.pause()
            assert mock_refresh.called, "chat Select collapse must trigger refresh"

        embed_sel.expanded = True
        await pilot.pause()
        with patch.object(app.screen, "refresh") as mock_refresh:
            embed_sel.expanded = False
            await pilot.pause()
            assert mock_refresh.called, "embed Select collapse must trigger refresh"
