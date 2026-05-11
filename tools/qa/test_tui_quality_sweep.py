"""T4 TUI: scenarios introduced by the tui-quality-sweep PR.

Covers the new model bar (searchable picker buttons + Search/Chat toggle),
the catalog Local/Frontier sub-tabs, and the lazy Settings tabs. API-key
gated rows are intentionally omitted; flip them on manually when running
the matrix against real provider credentials.
"""

from __future__ import annotations

import re
import time

import pytest
from drivers.tui import TuiSession

from conftest import TUI_BOOT_TIMEOUT, TUI_SCREEN_TIMEOUT, Lane

_TUI_REDRAW_POLL = 0.25


@pytest.fixture
def tui(tui_with_models: TuiSession) -> TuiSession:
    """Every scenario here drives a post-wizard screen (model bar, catalog,
    settings), so this module's ``tui`` needs models pre-assigned. Override
    the bare conftest fixture for the whole file."""
    return tui_with_models


def _open_model_catalog(tui: TuiSession) -> None:
    """Navigate to the model catalog screen and wait for its 'Local' tab.

    ``/models`` opens the catalog screen; CatalogScreen does an HF fetch on
    mount, so the tab strip can take a beat to render on a slow runner.
    Wait patiently (don't poke keys at a half-mounted screen); retry the
    submit once if the first Enter didn't take.
    """
    wait = TUI_SCREEN_TIMEOUT * 2  # 30s: covers a slow HF fetch on mount
    tui.send("/models")
    time.sleep(_TUI_REDRAW_POLL * 2)  # let the completion dropdown settle
    tui.send("\r")
    try:
        tui.wait_for("Local", timeout=wait)
        return
    except AssertionError:
        pass
    # If "/models" is still in the input the first Enter didn't submit;
    # nudge it once more. Don't retype (key-spam can corrupt input state).
    if "/models" in tui.text():
        tui.send("\r")
    tui.wait_for("Local", timeout=wait)


@pytest.mark.tui
def test_model_bar_shows_picker_buttons_and_search_toggle(tui: TuiSession) -> None:
    """Chat screen renders the chat + embed picker buttons and the Search/Chat mode toggle."""
    tui.wait_for("lilbee", timeout=TUI_BOOT_TIMEOUT)
    visible = tui.text().lower()
    assert "chat" in visible
    assert "embed" in visible
    assert "search" in visible


@pytest.mark.tui
@pytest.mark.flaky(reruns=2)
def test_chat_mode_toggle_flips_with_f3(tui: TuiSession) -> None:
    """F3 flips the Search/Chat toggle and shows a `Mode: ...` toast.

    The active pill is a CSS class, not text, so the screen capture can't see
    the flip directly; the toast is the observable signal. F3 is a no-op
    (no toast) only when Search mode is disabled because no embedding model
    is ready, which doesn't apply here. PTY key delivery for F3 is timing
    sensitive on the frozen-binary lanes, so rerun on a miss.
    """
    tui.wait_for("lilbee", timeout=TUI_BOOT_TIMEOUT)
    tui.send("\x1b[13~")  # F3 escape sequence
    tui.wait_for("Mode:", timeout=TUI_SCREEN_TIMEOUT)


@pytest.mark.tui
@pytest.mark.flaky(reruns=2)
def test_model_picker_modal_opens_on_chat_button(tui: TuiSession) -> None:
    """Esc -> NORMAL mode, ``m`` focuses the chat-model picker button, then
    Space activates it and the "Pick a chat model" modal opens.

    Esc and ``m`` go out with a gap so the terminal doesn't fold them into a
    single Alt+m event, and the whole sequence retries because a fast Esc
    can land mid-redraw and not take. Space (not Enter) activates the button
    so a stray keypress in INSERT mode can't submit a chat message.
    """
    tui.wait_for("lilbee", timeout=TUI_BOOT_TIMEOUT)
    deadline = time.monotonic() + TUI_SCREEN_TIMEOUT * 3
    while time.monotonic() < deadline:
        if "Pick" in tui.text():
            tui.send("\x1b")  # close the modal so teardown is clean
            return
        tui.send("\x1b")  # -> NORMAL mode
        time.sleep(_TUI_REDRAW_POLL)
        tui.send("m")  # focus the chat-model picker button
        time.sleep(_TUI_REDRAW_POLL)
        tui.send(" ")  # Space activates the focused button
        time.sleep(_TUI_REDRAW_POLL * 4)
    raise AssertionError(
        "model picker modal ('Pick a chat model') never opened; visible:\n" + tui.text()
    )


@pytest.mark.tui
@pytest.mark.timeout(180)
@pytest.mark.flaky(reruns=2)
def test_catalog_screen_has_local_tab_visible(tui: TuiSession) -> None:
    """The Catalog screen exposes a Local sub-tab.

    Frontier visibility is API-key dependent and exercised in the manual lane.
    """
    tui.wait_for("lilbee", timeout=TUI_BOOT_TIMEOUT)
    _open_model_catalog(tui)
    assert "Local" in tui.text()


@pytest.mark.tui
@pytest.mark.timeout(180)
@pytest.mark.flaky(reruns=2)
def test_catalog_v_toggles_grid_list_in_local_tab(tui: TuiSession) -> None:
    """`v` swaps grid <-> list view inside the Local tab."""
    tui.wait_for("lilbee", timeout=TUI_BOOT_TIMEOUT)
    _open_model_catalog(tui)
    before = tui.text()
    tui.send("v")
    # The visible state changes (grid renders cards, list renders rows).
    deadline = time.monotonic() + TUI_SCREEN_TIMEOUT
    while time.monotonic() < deadline:
        if tui.text() != before:
            return
        time.sleep(_TUI_REDRAW_POLL)
    raise AssertionError("`v` did not change catalog Local view state. visible:\n" + tui.text())


@pytest.mark.tui
@pytest.mark.timeout(180)
@pytest.mark.flaky(reruns=2)
def test_catalog_renders_fit_chip_for_at_least_one_row(tui: TuiSession) -> None:
    """The Catalog screen renders a hardware-fit chip on at least one row.

    The catalog response carries server-computed fit + size_variants;
    the catalog screen binds those fields to per-row chips with the
    compact labels `fits`, `tight`, `won't run`. This asserts the
    round-trip from the response field through the rendered cell works.
    Uses regex word boundaries so the "fits" label doesn't match unrelated
    substrings like "benefits". "Won't run" is unambiguous as a phrase.
    """
    tui.wait_for("lilbee", timeout=TUI_BOOT_TIMEOUT)
    _open_model_catalog(tui)
    chip_pattern = re.compile(r"\bfits\b|\btight\b|won't run", re.IGNORECASE)
    deadline = time.monotonic() + TUI_SCREEN_TIMEOUT
    while time.monotonic() < deadline:
        if chip_pattern.search(tui.text()):
            return
        time.sleep(_TUI_REDRAW_POLL)
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
    tui.wait_for("lilbee", timeout=TUI_BOOT_TIMEOUT)
    tui.send("/settings\r")
    tui.wait_for("Settings", timeout=TUI_SCREEN_TIMEOUT)
    visible = tui.text()
    assert "API-Keys" not in visible, "API-Keys tab should be hidden without litellm"
    assert "Crawling" not in visible, "Crawling tab should be hidden without crawler extra"
    assert "Wiki" not in visible, "Wiki tab should be hidden when cfg.wiki is false"
