"""Chat says "just ask", so the first keys a user types must reach the prompt.

Each test opens Chat the way a user does and types a question at once, with no
key to change mode first. A key that a mode binding consumes (w, h, a, i, o) would
drop out of the question.
"""

from __future__ import annotations

from typing import Any

import pytest

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.startup_gate import StartupGate
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.core.config import cfg
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import await_chat, send_key_burst
from tests._lilbee_app_test_host import ready_services as _ready_services

QUESTION = "what table is in the paper"
KEYS = ["space" if char == " " else char for char in QUESTION]


@pytest.fixture(autouse=True)
def _gate_releases_at_once():
    with _ready_services():
        yield


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path) -> Any:
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.wiki = False
    for path in (cfg.data_dir, cfg.documents_dir, cfg.models_dir, cfg.lancedb_dir):
        path.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


async def test_chat_opens_in_insert_mode_with_the_prompt_focused() -> None:
    """The welcome hint says "just ask", so Chat opens ready to type."""
    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        chat = await await_chat(app, pilot)
        assert isinstance(chat, ChatScreen)
        assert msg.CHAT_WELCOME_HINT.endswith("just ask.")
        assert chat._insert_mode is True
        assert app.focused is chat.query_one("#chat-input", ChatInput)


async def test_a_question_typed_at_once_reaches_the_prompt() -> None:
    """Type the whole question the moment Chat shows; none of it may act as a key."""
    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        chat = await await_chat(app, pilot)
        inp = chat.query_one("#chat-input", ChatInput)
        await pilot.press(*KEYS)
        assert await wait_until(pilot, lambda: inp.value == QUESTION, timeout=5.0), inp.value
        assert chat._insert_mode is True


async def test_a_question_typed_while_chat_starts_reaches_the_prompt() -> None:
    """Keys pressed before Chat shows are kept and land in the prompt in order."""
    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.press(*KEYS)
        chat = await await_chat(app, pilot)
        inp = chat.query_one("#chat-input", ChatInput)
        assert await wait_until(pilot, lambda: inp.value == QUESTION, timeout=5.0), inp.value


async def test_keys_typed_as_the_gate_hands_over_reach_the_prompt(monkeypatch) -> None:
    """Keys that arrive between the gate's handover and Chat showing still type.

    The handover switches Chat in at once, but Chat composes and focuses its
    prompt a few steps later. The rest of the first word lands in that gap, as
    typing does when the handover falls in the middle of a word.
    """
    monkeypatch.setattr(StartupGate, "start_boot", lambda self: None)
    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        assert await wait_until(
            pilot,
            lambda: isinstance(app.screen, StartupGate) and app.chat_screen() is not None,
            timeout=10.0,
        )
        gate = app.screen
        assert isinstance(gate, StartupGate)
        await pilot.press(*KEYS[:1])
        app.chat_is_ready = True
        gate._release()
        send_key_burst(app, *KEYS[1:4])
        chat = await await_chat(app, pilot)
        inp = chat.query_one("#chat-input", ChatInput)
        await pilot.press(*KEYS[4:])
        assert await wait_until(pilot, lambda: inp.value == QUESTION, timeout=5.0), inp.value


async def test_keys_typed_before_a_catalog_landing_never_reach_the_prompt(monkeypatch) -> None:
    """With no chat model the gate lands on the Catalog, and the keys typed at the gate go.

    A later visit to Chat opens an empty prompt, not the old keys.
    """
    monkeypatch.setattr(StartupGate, "start_boot", lambda self: None)
    app = LilbeeApp()
    async with app.run_test(size=(160, 48)) as pilot:
        assert await wait_until(
            pilot,
            lambda: isinstance(app.screen, StartupGate) and app.chat_screen() is not None,
            timeout=10.0,
        )
        gate = app.screen
        assert isinstance(gate, StartupGate)
        await pilot.press(*KEYS[:4])
        app.chat_is_ready = False
        gate._release()
        # A switch made while the catalog switch still runs is dropped, so wait for it to end.
        assert await wait_until(
            pilot,
            lambda: app.active_view == msg.CATALOG_VIEW and not app._switching,
            timeout=10.0,
        )
        app.switch_view(msg.DEFAULT_VIEW)
        chat = await await_chat(app, pilot)
        assert chat.query_one("#chat-input", ChatInput).value == ""


def _app_held_at_the_gate(monkeypatch, sent: list[str]) -> LilbeeApp:
    """An app whose gate waits for the test to release it; a sent prompt lands in *sent*."""
    monkeypatch.setattr(StartupGate, "start_boot", lambda self: None)
    monkeypatch.setattr(ChatScreen, "_send_message", lambda self, text: sent.append(text))
    return LilbeeApp()


async def test_backspace_at_the_gate_edits_the_kept_text(monkeypatch) -> None:
    sent: list[str] = []
    app = _app_held_at_the_gate(monkeypatch, sent)
    async with app.run_test(size=(160, 48)) as pilot:
        gate = await _held_gate(app, pilot)
        await pilot.press("a", "b", "backspace", "c")
        app.chat_is_ready = True
        gate._release()
        chat = await await_chat(app, pilot)
        inp = chat.query_one("#chat-input", ChatInput)
        assert await wait_until(pilot, lambda: inp.value == "ac", timeout=5.0), inp.value


async def test_backspace_removes_one_character_of_a_paste(monkeypatch) -> None:
    from textual import events

    sent: list[str] = []
    app = _app_held_at_the_gate(monkeypatch, sent)
    async with app.run_test(size=(160, 48)) as pilot:
        gate = await _held_gate(app, pilot)
        app.post_message(events.Paste("abc"))
        await pilot.press("backspace")
        app.chat_is_ready = True
        gate._release()
        chat = await await_chat(app, pilot)
        inp = chat.query_one("#chat-input", ChatInput)
        assert await wait_until(pilot, lambda: inp.value == "ab", timeout=5.0), inp.value


async def test_enter_at_the_gate_sends_the_kept_text(monkeypatch) -> None:
    sent: list[str] = []
    app = _app_held_at_the_gate(monkeypatch, sent)
    async with app.run_test(size=(160, 48)) as pilot:
        gate = await _held_gate(app, pilot)
        await pilot.press("h", "i", "enter")
        app.chat_is_ready = True
        gate._release()
        chat = await await_chat(app, pilot)
        assert await wait_until(pilot, lambda: sent == ["hi"], timeout=5.0), sent
        assert chat.query_one("#chat-input", ChatInput).value == ""


async def test_enter_with_nothing_kept_sends_nothing(monkeypatch) -> None:
    sent: list[str] = []
    app = _app_held_at_the_gate(monkeypatch, sent)
    async with app.run_test(size=(160, 48)) as pilot:
        gate = await _held_gate(app, pilot)
        await pilot.press("enter")
        app.chat_is_ready = True
        gate._release()
        chat = await await_chat(app, pilot)
        await pilot.pause()
        assert sent == []
        assert chat.query_one("#chat-input", ChatInput).value == ""


async def _held_gate(app, pilot) -> StartupGate:
    assert await wait_until(
        pilot,
        lambda: isinstance(app.screen, StartupGate) and app.chat_screen() is not None,
        timeout=10.0,
    )
    gate = app.screen
    assert isinstance(gate, StartupGate)
    return gate
