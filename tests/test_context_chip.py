"""The context chip: how full the model's memory is, and when it is condensing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.services import set_services
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.widgets.context_chip import ContextChip
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import await_chat
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def _services():
    store = MagicMock()
    store.get_sources.return_value = []
    set_services(make_mock_services(store=store))
    yield
    set_services(None)


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    with (
        patch("lilbee.cli.tui.app.models_ready", return_value=True),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
        patch("lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount"),
    ):
        yield


def test_stays_quiet_while_the_window_is_half_empty() -> None:
    """A chip that always shouts is one the eye stops reading."""
    chip = ContextChip()
    chip.usage = 0.2
    assert str(chip.render()) == ""


def test_shows_the_percentage_once_the_window_is_filling() -> None:
    chip = ContextChip()
    chip.usage = 0.62
    assert "62" in str(chip.render())


def test_reports_full_rather_than_over_full() -> None:
    """Usage can exceed 1.0 before the next turn trims; 137% would read as a bug."""
    chip = ContextChip()
    chip.usage = 1.37
    assert "100" in str(chip.render())


def test_warns_that_turns_will_drop_only_when_compaction_is_off() -> None:
    """The nudge is worth showing exactly when the user can still act on it.

    Compaction off and the window nearly full: turns are about to leave the
    model's view and flipping the setting (or asking the important thing now)
    would prevent it. With compaction on it resolves itself, so saying anything
    would be noise.
    """
    cfg.chat_compaction = False
    chip = ContextChip()
    chip.usage = 0.85
    assert "drop" in str(chip.render()), "the user is about to lose turns and can act"

    cfg.chat_compaction = True
    assert "drop" not in str(chip.render()), "nothing to decide; the number is enough"
    assert "85" in str(chip.render())


def test_does_not_warn_before_there_is_anything_to_warn_about() -> None:
    """Advice that fires while nothing is wrong is advice people learn to ignore."""
    cfg.chat_compaction = False
    chip = ContextChip()
    chip.usage = 0.6  # filling, but nothing is dropping yet
    rendered = str(chip.render())
    assert "60" in rendered
    assert "drop" not in rendered


def test_says_it_is_condensing_while_the_model_call_blocks() -> None:
    """The whole point: a 20-50s pause on CPU must not look like a hang."""
    chip = ContextChip()
    chip.usage = 0.9
    chip.compacting = True
    rendered = str(chip.render())
    assert rendered == msg.CONTEXT_CHIP_COMPACTING
    assert "%" not in rendered, "the pause is the news, not the percentage"


async def test_the_chip_tracks_a_real_conversation_filling_the_window() -> None:
    """Driven from the screen: usage rises with history, and falls after a trim."""
    cfg.chat_n_ctx_target = 2048  # budget 1024
    cfg.chat_compaction = False
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        chip = screen.query_one("#context-chip", ContextChip)
        screen._refresh_context_usage()
        assert chip.usage == pytest.approx(0.0), "an empty chat shows an empty window"

        screen._history = [{"role": "user", "content": "x" * 2000}]  # ~500 tokens
        screen._refresh_context_usage()
        filling = chip.usage
        assert filling > 0.4, "the chip sees the window filling"

        screen._history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 2000} for i in range(6)
        ]
        screen._compact_history()  # compaction off: trims, no model call
        await pilot.pause()
        assert chip.usage < filling * 6, "the chip falls back after turns leave the prompt"


async def test_the_chip_is_actually_visible_on_screen() -> None:
    """It must occupy columns, not merely return text from render().

    The chip's width is `auto`, so it is measured from its own text. A reactive
    that only repaints redraws inside the box the widget already has: mounted
    empty at 0% it measures zero columns and stays invisible no matter what it
    renders later. Every other test here calls render() directly and would pass
    against a chip nobody can see, which is exactly how this shipped once.
    """
    cfg.chat_compaction = False
    app = LilbeeApp()
    async with app.run_test(size=(120, 30)) as pilot:
        screen = await await_chat(app, pilot)
        chip = screen.query_one("#context-chip", ContextChip)
        assert chip.region.width == 0, "quiet at 0%: nothing to say, no space taken"

        chip.usage = 0.85
        await pilot.pause()
        assert chip.region.width > 0, "the chip must take space once it has something to say"
        assert chip.region.width >= len(str(chip.render())), "and enough space to be read"


async def test_the_chip_shrinks_back_when_it_goes_quiet() -> None:
    """The reverse of the same layout bug: it must give the space back."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 30)) as pilot:
        screen = await await_chat(app, pilot)
        chip = screen.query_one("#context-chip", ContextChip)
        chip.usage = 0.85
        await pilot.pause()
        assert chip.region.width > 0
        chip.usage = 0.1  # a new chat: nothing to report
        await pilot.pause()
        assert chip.region.width == 0, "a quiet chip must not hold columns hostage"


async def test_the_screen_flips_the_chip_while_compacting() -> None:
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        chip = screen.query_one("#context-chip", ContextChip)
        assert chip.compacting is False
        screen._set_compacting(True)
        assert chip.compacting is True
        screen._set_compacting(False)
        assert chip.compacting is False
