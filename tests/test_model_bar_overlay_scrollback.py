"""Scope-select overlay must not leak borders into terminal scrollback.

The overlay is capped to stay inside the screen, and collapsing it forces
a full screen refresh so the compositor invalidates the region.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui.widgets.scope_chip import ScopeChip
from lilbee.core.config import cfg


class _ScopeChipApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ScopeChip(id="scope-chip")


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.chat_mode = "search"
    cfg.wiki = True
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


def test_scope_chip_caps_overlay_height_and_constrains_inside() -> None:
    """CSS caps overlay height and inflects so it stays within the viewport."""
    css = ScopeChip.DEFAULT_CSS
    assert "max-height: 12" in css
    assert "constrain: inflect inflect" in css


async def test_unmount_collapses_open_scope_select() -> None:
    """on_unmount collapses any expanded Select so its overlay does not leak."""
    app = _ScopeChipApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        scope_sel = app.query_one("#scope-select", Select)
        scope_sel.expanded = True
        await pilot.pause()
        chip = app.query_one(ScopeChip)
        chip.on_unmount()
        assert scope_sel.expanded is False
