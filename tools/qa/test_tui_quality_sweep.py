"""T4 TUI: scenarios introduced by the tui-quality-sweep PR.

Covers the new model bar (searchable picker buttons + Search/Chat toggle),
the catalog Local/Frontier sub-tabs, and the lazy Settings tabs. API-key
gated rows are intentionally omitted; flip them on manually when running
the matrix against real provider credentials.
"""

from __future__ import annotations

import re

import pytest
from drivers.tui import TuiSession

from conftest import Lane

_TUI_BOOT_TIMEOUT = 60.0
_TUI_SCREEN_TIMEOUT = 15.0


@pytest.mark.tui
def test_model_bar_shows_picker_buttons_and_search_toggle(tui: TuiSession) -> None:
    """Chat screen renders the chat + embed picker buttons and the
    Search/Chat mode toggle. The search-mode label is the load-bearing
    affordance: a missing toggle should fail the assertion, which the
    previous "chat or chat" OR couldn't detect.
    """
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
    visible = tui.text().lower()
    assert "chat" in visible
    assert "embed" in visible
    assert "search" in visible


@pytest.mark.tui
def test_chat_mode_toggle_flips_with_f3(tui: TuiSession) -> None:
    """F3 flips the Search/Chat toggle and updates the bar label."""
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
    before = tui.text().lower()
    tui.send("\x1b[13~")  # F3 escape sequence
    try:
        tui.wait_for("mode:", timeout=_TUI_SCREEN_TIMEOUT)
    except AssertionError:
        # Fall back to checking the bar label changed (toast is transient).
        tui.send("")
        after = tui.text().lower()
        assert after != before, (
            f"F3 produced no visible change in the model bar. before:\n{before}\nafter:\n{after}"
        )


@pytest.mark.tui
def test_model_picker_modal_opens_on_chat_button(tui: TuiSession) -> None:
    """Pressing Enter on the focused chat picker button opens the search modal.

    The modal title contains "Pick a chat model" and a search input is
    focused; typing should narrow the visible row count.
    """
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
    tui.send("\x1b")  # Escape to normal mode
    tui.send("m")  # focus the model bar (m binding)
    try:
        tui.wait_for("Pick a chat model", timeout=_TUI_SCREEN_TIMEOUT)
    except AssertionError:
        # Some terminals reorder Enter handling; retry with explicit Enter.
        tui.send("\r")
        tui.wait_for("Pick", timeout=_TUI_SCREEN_TIMEOUT)
    # Escape closes; not asserting selection because available models depend
    # on the lane's installed registry.
    tui.send("\x1b")


@pytest.mark.tui
def test_catalog_screen_has_local_tab_visible(tui: TuiSession) -> None:
    """The Catalog screen exposes a Local sub-tab.

    Frontier visibility is API-key dependent and exercised in the manual lane.
    """
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
    tui.send("/models\r")
    tui.wait_for("Local", timeout=_TUI_SCREEN_TIMEOUT)
    visible = tui.text()
    assert "Local" in visible


@pytest.mark.tui
def test_catalog_v_toggles_grid_list_in_local_tab(tui: TuiSession) -> None:
    """`v` swaps grid <-> list view inside the Local tab."""
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
    tui.send("/models\r")
    tui.wait_for("Local", timeout=_TUI_SCREEN_TIMEOUT)
    before = tui.text()
    tui.send("v")
    # The visible state changes (grid renders cards, list renders rows).
    deadline_attempts = 5
    for _ in range(deadline_attempts):
        if tui.text() != before:
            return
    raise AssertionError("`v` did not change catalog Local view state. visible:\n" + tui.text())


@pytest.mark.tui
def test_catalog_renders_fit_chip_for_at_least_one_row(tui: TuiSession) -> None:
    """The Catalog screen renders a hardware-fit chip on at least one row.

    PR #218 added server-side fit + size_variants on /api/models/catalog;
    PR #212's catalog redesign binds those fields to per-row chips with
    the compact labels `fits`, `tight`, `won't run`. This asserts the
    round-trip from the response field through the rendered cell works.
    Uses regex word boundaries so the "fits" label doesn't match unrelated
    substrings like "benefits". Won't run is unambiguous as a phrase.
    """
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
    tui.send("/models\r")
    tui.wait_for("Local", timeout=_TUI_SCREEN_TIMEOUT)
    # Compact chip labels rendered by _fit_pill_compact in
    # src/lilbee/cli/tui/widgets/model_grid.py.
    chip_pattern = re.compile(r"\bfits\b|\btight\b|won't run", re.IGNORECASE)
    deadline_attempts = 8
    for _ in range(deadline_attempts):
        if chip_pattern.search(tui.text()):
            return
    raise AssertionError(
        "catalog rendered without any fits/tight/won't run chip; visible:\n" + tui.text()
    )


@pytest.mark.tui
def test_settings_screen_omits_unavailable_tabs(tui: TuiSession, lane: Lane) -> None:
    """Settings tabs that depend on optional extras are hidden, not greyed out.

    The plain wheel install ships without [crawler] or [litellm] and leaves
    ``cfg.wiki=False``, so the API-Keys / Crawling / Wiki tabs are absent.
    The release binary bundles those extras, so the tabs render there and
    this invariant doesn't apply.
    """
    if lane.is_binary:
        pytest.skip("the release binary bundles the extras these tabs gate on")
    tui.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
    tui.send("/settings\r")
    tui.wait_for("Settings", timeout=_TUI_SCREEN_TIMEOUT)
    visible = tui.text()
    assert "API-Keys" not in visible, "API-Keys tab should be hidden without litellm"
    assert "Crawling" not in visible, "Crawling tab should be hidden without crawler extra"
    assert "Wiki" not in visible, "Wiki tab should be hidden when cfg.wiki is false"
