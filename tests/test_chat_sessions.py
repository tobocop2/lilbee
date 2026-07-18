"""Chat session lifecycle: auto-save, new chat, and resume."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage
from lilbee.cli.tui.widgets.thinking_header import ThinkingHeader
from lilbee.core.config import Config, cfg
from lilbee.retrieval.query.compaction import CompactionResult
from lilbee.sessions import MessageRole, SessionMessage, TitleSource
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
        patch("lilbee.cli.tui.screens.chat.needs_setup", return_value=False),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
        patch("lilbee.cli.tui.widgets.model_bar.ModelBar.on_mount"),
    ):
        yield


@pytest.fixture
def sessions():
    return get_services().session_store


def _bubble_text(bubble: AssistantMessage) -> str:
    """The answer text a reader would actually see in a bubble.

    Reads the mounted content widget rather than the object's internals: the bug
    this guards was precisely that the internals held the text while the widget
    on screen was empty.
    """
    from textual.widgets import Markdown, Static

    for child in bubble.query(Markdown):
        return child.source
    for child in bubble.query(Static):
        if "response-md" in child.classes:
            return str(child.renderable)
    return ""


async def test_first_turn_creates_titled_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("Torque specs in the manual")
        metas = sessions.list()
        assert len(metas) == 1
        assert metas[0].title == "Torque specs in the manual"
        assert metas[0].message_count == 1
        assert screen._session_id == metas[0].id


async def test_turns_append_to_the_same_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("first question")
        session_id = screen._session_id
        screen._persist_assistant_turn("the answer", ["manual.pdf"])
        screen._persist_user_turn("second question")
        assert screen._session_id == session_id
        assert len(sessions.list()) == 1
        session = sessions.get(session_id)
        assert session.meta.message_count == 3
        assert session.messages[1].role == MessageRole.ASSISTANT
        assert session.messages[1].sources == ("manual.pdf",)


async def test_user_turn_recovers_if_active_session_deleted(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("first")
        first_id = screen._session_id
        sessions.delete(first_id)  # deleted from the drawer / another surface
        screen._persist_user_turn("second")  # must not raise
        assert screen._session_id != first_id
        assert len(sessions.list()) == 1


async def test_assistant_turn_swallows_deleted_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("q")
        sessions.delete(screen._session_id)  # gone mid-stream
        screen._persist_assistant_turn("a", [])  # must not raise
        assert sessions.list() == []


async def test_assistant_persist_is_noop_without_a_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_assistant_turn("orphan answer", [])
        assert sessions.list() == []


async def test_reset_conversation_opens_a_fresh_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("first")
        first_id = screen._session_id
        screen._reset_conversation()
        assert screen._session_id is None
        screen._persist_user_turn("second")
        assert screen._session_id != first_id
        assert len(sessions.list()) == 2


async def test_resume_loads_history_and_renders(sessions):
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    sessions.set_title(session_id, "Seeded", TitleSource.AUTO)
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="the question"))
    sessions.add_message(
        session_id,
        SessionMessage(role=MessageRole.ASSISTANT, content="the answer", sources=("s.pdf",)),
    )
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(session_id)
        await pilot.pause()
        assert screen._session_id == session_id
        assert screen._history == [
            {"role": "user", "content": "the question"},
            {"role": "assistant", "content": "the answer"},
        ]
        assert len(screen.query(UserMessage)) == 1
        assert len(screen.query(AssistantMessage)) == 1
        # Counting the bubbles is not the same as reading them. Resume once
        # rendered an empty bubble with a stuck spinner -- mount() is async, so
        # appending content straight after it silently no-ops -- and this test
        # passed the whole time because it only counted widgets.
        bubble = screen.query(AssistantMessage).first()
        assert "the answer" in _bubble_text(bubble), "the answer must actually be on screen"
        assert not bubble.query(ThinkingHeader), "a restored turn must not spin"


async def test_resume_keeps_the_whole_transcript_for_compaction(sessions):
    """Resume must not window: what does not fit is summarized on the next turn.

    Windowing at resume drops the turns between the stored summary and the window
    with nothing standing in for them, which is the silent memory loss compaction
    exists to prevent. _compact_history folds them in instead, off the UI thread.
    """
    cfg.chat_n_ctx_target = 512  # budget = 256 tokens after the 0.5 fraction
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    long_text = "x" * 1200  # ~300 tokens each, over the whole budget on its own
    for i in range(6):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        sessions.add_message(session_id, SessionMessage(role=role, content=long_text))
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(session_id)
        await pilot.pause()
        rendered = len(screen.query(UserMessage)) + len(screen.query(AssistantMessage))
        assert rendered == 6
        assert len(screen._history) == 6, "the whole transcript is kept for compaction"


async def test_resume_carries_the_stored_summary(sessions):
    """The notes from the conversation's earlier compactions come back with it."""
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="q"))
    sessions.set_summary(session_id, "earlier: they compared torque specs")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(session_id)
        await pilot.pause()
        assert screen._summary == "earlier: they compared torque specs"


async def test_new_conversation_drops_the_previous_summary(sessions):
    """A fresh chat must not leak the last conversation's notes into its prompt."""
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="q"))
    sessions.set_summary(session_id, "earlier: secrets")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(session_id)
        await pilot.pause()
        screen.start_new_conversation()
        await pilot.pause()
        assert screen._summary == ""


async def test_compaction_summarizes_the_overflow_instead_of_dropping_it(sessions):
    """The turns that no longer fit become notes, and the notes are persisted."""
    cfg.chat_n_ctx_target = 512
    cfg.chat_compaction = True
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    # Persist the same turns the chat holds, so "the transcript is untouched"
    # asserts against a real transcript instead of trivially against an empty one.
    for i in range(6):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        sessions.add_message(session_id, SessionMessage(role=role, content="x" * 1200))
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._session_id = session_id
        screen._history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 1200} for i in range(6)
        ]
        with patch.object(
            get_services().searcher,
            "summarize_history",
            return_value=CompactionResult(summary="NOTES", condensed=4, stranded=0),
        ) as summarize:
            screen._compact_history()
        summarize.assert_called_once()
        dropped = summarize.call_args.args[0]
        assert dropped, "the overflow is what gets summarized"
        assert screen._summary == "NOTES"
        assert len(screen._history) == 6 - len(dropped)
        session = sessions.get(session_id)
        assert session.summary == "NOTES", "notes survive a restart"
        assert session.meta.message_count == 6, "compaction condenses the prompt, not the log"
        assert [m.content for m in session.messages] == ["x" * 1200] * 6


async def test_compaction_is_off_by_default(sessions):
    """The default must be free: nobody's chat pays a model call they didn't ask for."""
    # Read the default off the class: the instance attribute is deprecated in
    # pydantic 2.11, and cfg's live value is whatever this test session set.
    assert Config.model_fields["chat_compaction"].default is False


async def test_default_path_drops_the_tail_with_zero_model_calls(sessions):
    """Compaction off: the oldest turns leave the model's view, costing nothing.

    A blocking summarize call is seconds on a GPU and tens of seconds on a CPU,
    so the out-of-the-box path must not make one. The turns still leave the
    prompt; the marker is what stops that being silent.
    """
    cfg.chat_compaction = False
    cfg.chat_n_ctx_target = 512
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 1200} for i in range(6)
        ]
        with patch.object(get_services().searcher, "summarize_history") as summarize:
            screen._compact_history()
        summarize.assert_not_called(), "the default path must not call the model"
        assert len(screen._history) < 6, "the oldest turns still leave the prompt"
        assert screen._summary == "", "nothing is summarized when compaction is off"


async def test_trimming_marks_the_log_and_points_at_the_setting(sessions):
    """Turns leaving the model's memory must be visible, and fixable."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._on_history_trimmed(6)
        await pilot.pause()
        rendered = str(screen.query(".compaction-marker").first().render())
        assert "6" in rendered
        assert "summary" not in rendered, "nothing was summarized; do not imply it was"


async def test_compaction_never_summarizes_the_turn_being_answered(sessions):
    """One huge exchange can fill the budget alone. It must not be condensed.

    Folding it would replace the question the user just asked with a paraphrase
    and answer that instead. There is nothing older to fold here, so compaction
    stands down and lets the prompt window handle it.
    """
    cfg.chat_compaction = True
    cfg.chat_n_ctx_target = 512
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        # a single exchange, far over the budget, with no older turns behind it
        screen._history = [
            {"role": "user", "content": "x" * 4000},
            {"role": "assistant", "content": "y" * 4000},
        ]
        with patch.object(get_services().searcher, "summarize_history") as summarize:
            screen._compact_history()
        summarize.assert_not_called(), "the live exchange must never be summarized"
        assert len(screen._history) == 2, "and it stays intact for the prompt to window"


async def test_compaction_is_skipped_when_everything_still_fits(sessions):
    """No overflow means no summarizing model call on every single turn."""
    cfg.chat_compaction = True
    cfg.chat_n_ctx_target = 32768
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._history = [{"role": "user", "content": "short"}]
        with patch.object(get_services().searcher, "summarize_history") as summarize:
            screen._compact_history()
        summarize.assert_not_called()
        assert screen._summary == ""


async def test_compaction_survives_the_session_being_deleted_mid_chat(sessions):
    """Deleting the active session must not crash the turn that is compacting."""
    cfg.chat_compaction = True
    cfg.chat_n_ctx_target = 512
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._session_id = session_id
        screen._history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 1200} for i in range(6)
        ]
        sessions.delete(session_id)
        with patch.object(
            get_services().searcher,
            "summarize_history",
            return_value=CompactionResult(summary="NOTES", condensed=4, stranded=0),
        ):
            screen._compact_history()  # must not raise
        assert screen._summary == "NOTES"


async def test_compaction_marks_the_log_where_the_model_s_memory_becomes_a_summary(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._on_history_compacted(4, 0)
        await pilot.pause()
        markers = screen.query(".compaction-marker")
        assert len(markers) == 1
        rendered = str(markers.first().render())
        assert "4" in rendered
        assert "dropped" not in rendered, "nothing was stranded, so say nothing about dropping"


async def test_a_boundary_lands_above_the_turn_being_answered(sessions):
    """The rule must not appear under the question it did not drop.

    Compaction runs after _send_message has already mounted the question and its
    pending answer, so mounting the rule at the end of the log puts it below them
    -- reading as though the question just asked had itself fallen out of context.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        log = screen._chat_log
        log.remove_children()
        # an older exchange, then the live turn, exactly as _send_message leaves it
        log.mount(UserMessage("older question"))
        log.mount(AssistantMessage(content="older answer"))
        question = UserMessage("the live question")
        log.mount(question)
        screen._active_question = question
        log.mount(AssistantMessage())
        await pilot.pause()

        screen._on_history_trimmed(10)
        await pilot.pause()

        kinds = [type(w).__name__ for w in log.children]
        rule_at = next(i for i, w in enumerate(log.children) if "compaction-marker" in w.classes)
        live_at = list(log.children).index(question)
        assert rule_at < live_at, f"the rule landed below the live question: {kinds}"
        assert rule_at == 2, "and directly above it, not further up the log"


async def test_a_boundary_racing_the_question_mount_appends(sessions):
    """The anchor can exist before it is in the log; mounting before it would raise."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        log = screen._chat_log
        log.remove_children()
        await pilot.pause()
        screen._active_question = UserMessage("not in the log yet")
        screen._on_history_trimmed(4)
        await pilot.pause()
        assert "compaction-marker" in list(log.children)[-1].classes


async def test_a_boundary_without_a_live_turn_appends(sessions):
    """No turn in flight (e.g. after a resume): the rule simply ends the log."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        log = screen._chat_log
        log.remove_children()
        log.mount(UserMessage("q"))
        screen._active_question = None
        await pilot.pause()
        screen._on_history_trimmed(4)
        await pilot.pause()
        assert "compaction-marker" in list(log.children)[-1].classes


async def test_stranded_turns_are_reported_not_hidden(sessions):
    """A model too small to hold the conversation must say so, not seem forgetful."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._on_history_compacted(8, 120)
        await pilot.pause()
        # Two rules: what survived as notes, and what did not. Reading only the
        # first would pass while the loss went unreported.
        rules = [str(w.render()) for w in screen.query(".compaction-marker")]
        assert len(rules) == 2, "the stranded turns get their own rule"
        assert "8" in rules[0], "what was condensed"
        assert "120" in rules[1], "what was dropped outright"


async def test_resume_restores_a_different_model_when_installed(sessions):
    session_id = sessions.create(model_ref="other/model:latest", scope="both")
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="q"))
    get_services().registry.is_installed.return_value = True
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with patch.object(app, "set_active_model") as mock_set:
            screen.resume_session(session_id)
        mock_set.assert_called_once_with("chat_model", "other/model:latest")


async def test_resume_keeps_current_model_when_session_model_is_gone(sessions):
    session_id = sessions.create(model_ref="deleted/model:latest", scope="both")
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="q"))
    get_services().registry.is_installed.return_value = False  # model was deleted
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with patch.object(app, "set_active_model") as mock_set:
            screen.resume_session(session_id)
        mock_set.assert_not_called()  # never point chat at a missing model
        assert screen.session_id == session_id  # the conversation still loaded
        assert screen._history[0] == {"role": "user", "content": "q"}


# --- turning sessions off mid-conversation --------------------------------


async def test_assistant_turn_is_not_saved_after_sessions_go_off(sessions, monkeypatch):
    """Toggling sessions off mid-chat stops the auto-save.

    ``_persist_user_turn`` returns early, but the session is already open by
    then, so ``_session_id`` stays set and the assistant turn would otherwise
    keep appending to a conversation the user has switched off.
    """
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("first question")
        session_id = screen._session_id
        before = sessions.get(session_id).meta.message_count

        monkeypatch.setattr(cfg, "sessions_enabled", False)
        screen._persist_assistant_turn("the answer", [])

        assert sessions.get(session_id).meta.message_count == before


async def test_compaction_summary_is_not_saved_after_sessions_go_off(sessions, monkeypatch):
    """The compaction summary write honors the toggle too."""
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_user_turn("first question")
        session_id = screen._session_id

        monkeypatch.setattr(cfg, "sessions_enabled", False)
        with patch.object(
            get_services().searcher,
            "summarize_history",
            return_value=CompactionResult(summary="folded", condensed=2, stranded=0),
        ) as summarize:
            screen._history = [{"role": "user", "content": "x" * 1200} for _ in range(6)]
            cfg.chat_n_ctx_target = 512
            cfg.chat_compaction = True
            screen._compact_history()

        assert summarize.called, "the test must drive a real fold, or it asserts nothing"
        assert screen._summary == "folded", "the fold landed in memory"
        assert sessions.get(session_id).summary == "", "but nothing reached the disk"
