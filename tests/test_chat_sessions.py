"""Chat session lifecycle: auto-save, new chat, and resume."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage, _answer_markdown_parser
from lilbee.cli.tui.widgets.thinking_header import ThinkingHeader
from lilbee.core.config import Config, cfg
from lilbee.retrieval.query.compaction import CompactionResult
from lilbee.retrieval.query.formatting import SourceRef
from lilbee.retrieval.reasoning import StreamToken
from lilbee.sessions import MessageRole, SessionMessage, StoredSourceRef, TitleSource
from tests._lilbee_app_test_host import await_chat, pump_until
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
        patch("lilbee.cli.tui.app.chat_ready", return_value=True),
        patch("lilbee.cli.tui.app.embedding_ready", return_value=True),
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
        screen._persist_assistant_turn(
            session_id, "the answer [2]", [SourceRef("manual.pdf", "manual.pdf", 2)]
        )
        screen._persist_user_turn("second question")
        assert screen._session_id == session_id
        assert len(sessions.list()) == 1
        session = sessions.get(session_id)
        assert session.meta.message_count == 3
        assert session.messages[1].role == MessageRole.ASSISTANT
        assert session.messages[1].sources == ("manual.pdf",)
        assert session.messages[1].source_refs == (StoredSourceRef("manual.pdf", 2),)


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
        screen._persist_assistant_turn(screen._session_id, "a", [])  # must not raise
        assert sessions.list() == []


async def test_assistant_persist_is_noop_without_a_session(sessions):
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen._persist_assistant_turn(None, "orphan answer", [])
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
            screen._compact_history(screen._session_id, screen._conversation_generation)
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
            screen._compact_history(screen._session_id, screen._conversation_generation)
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
            screen._compact_history(screen._session_id, screen._conversation_generation)
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
            screen._compact_history(screen._session_id, screen._conversation_generation)
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
            # Must not raise.
            screen._compact_history(screen._session_id, screen._conversation_generation)
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
        screen._persist_assistant_turn(session_id, "the answer", [])

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
            screen._compact_history(screen._session_id, screen._conversation_generation)

        assert summarize.called, "the test must drive a real fold, or it asserts nothing"
        assert screen._summary == "folded", "the fold landed in memory"
        assert sessions.get(session_id).summary == "", "but nothing reached the disk"


def _seed_fold_source(store) -> str:
    """A saved conversation long enough that its next turn folds."""
    session_id = store.create(model_ref=cfg.chat_model, scope="both")
    for i in range(6):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        store.add_message(session_id, SessionMessage(role=role, content="x" * 1200))
    return session_id


def _seed_other(store) -> str:
    """A short conversation with its own notes, the one the user switches to."""
    session_id = store.create(model_ref=cfg.chat_model, scope="both")
    store.add_message(session_id, SessionMessage(role=MessageRole.USER, content="other q"))
    store.add_message(session_id, SessionMessage(role=MessageRole.ASSISTANT, content="other a"))
    store.set_summary(session_id, "OTHER NOTES")
    return session_id


async def _fold_then(app, pilot, switch) -> None:
    """Send a turn whose fold blocks in the model call, run *switch*, then release it."""
    folding, release = threading.Event(), threading.Event()

    def slow_summarize(*_args, **_kwargs):
        folding.set()
        release.wait()
        return CompactionResult(summary="OLD NOTES", condensed=4, stranded=0)

    screen = app.screen
    cfg.chat_n_ctx_target = 512
    cfg.chat_compaction = True
    with (
        patch.object(ChatScreen, "_await_chat_engine", return_value=True),
        patch.object(get_services().searcher, "ask_stream", return_value=iter(())),
        patch.object(get_services().searcher, "summarize_history", side_effect=slow_summarize),
    ):
        try:
            screen._send_message("new question")
            assert await pump_until(pilot, folding.is_set), "the turn must reach the fold"
            switch()
        finally:
            release.set()
        assert await pump_until(pilot, lambda: not screen.streaming)


async def test_resuming_during_a_fold_leaves_the_resumed_conversation_alone(sessions):
    """The old turn's fold must not trim, re-summarize or save into the session resumed over it."""
    source = _seed_fold_source(sessions)
    other = _seed_other(sessions)
    other_before = sessions.get(other)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        await _fold_then(app, pilot, lambda: app.resume_session(other))
        other_after = sessions.get(other)
        assert (screen._session_id, screen._history, screen._summary, other_after.summary) == (
            other,
            [{"role": "user", "content": "other q"}, {"role": "assistant", "content": "other a"}],
            "OTHER NOTES",
            "OTHER NOTES",
        )
        assert other_after.messages == other_before.messages


async def test_clearing_during_a_fold_starts_from_an_empty_conversation(sessions):
    """/clear mid-fold: the new conversation inherits none of the old one's notes."""
    source = _seed_fold_source(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        await _fold_then(app, pilot, lambda: screen._cmd_clear(""))
        assert screen._session_id is None
        assert screen._history == []
        assert screen._summary == ""
        assert [meta.id for meta in sessions.list()] == [source]


async def test_a_trim_for_a_replaced_conversation_is_dropped(sessions):
    """Compaction off: a trim computed for an earlier conversation leaves the current one whole."""
    cfg.chat_compaction = False
    cfg.chat_n_ctx_target = 512
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        stale = screen._conversation_generation
        screen._reset_conversation()
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 1200} for i in range(6)
        ]
        screen._history = list(history)
        screen._compact_history(None, stale)
        assert screen._history == history


async def test_a_fold_in_the_same_conversation_lands_through_the_send_path(sessions):
    """With no resume or /clear, the turn's fold trims the history and saves its summary."""
    source = _seed_fold_source(sessions)
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(source)
        await pilot.pause()
        await _fold_then(app, pilot, lambda: None)
        assert screen._summary == "OLD NOTES"
        assert sessions.get(source).summary == "OLD NOTES"
        assert len(screen._history) < 7, "the folded turns leave the history"
        assert screen._history[-1] == {"role": "user", "content": "new question"}


async def test_resuming_mid_answer_keeps_the_old_answer_out_of_the_resumed_history(sessions):
    """The cancelled answer is saved to its own session, never added to the resumed one."""
    other = _seed_other(sessions)
    first_token, release = threading.Event(), threading.Event()

    def stream(*_args, **_kwargs):
        yield StreamToken(content="PARTIAL", is_reasoning=False)
        first_token.set()
        release.wait()
        yield StreamToken(content=" more", is_reasoning=False)

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with (
            patch.object(ChatScreen, "_await_chat_engine", return_value=True),
            patch.object(get_services().searcher, "ask_stream", side_effect=stream),
        ):
            try:
                screen._send_message("old question")
                source = screen._session_id
                assert await pump_until(pilot, first_token.is_set), "the answer must have started"
                app.resume_session(other)
            finally:
                release.set()
            assert await pump_until(pilot, lambda: not screen.streaming)
        assert screen._history == [
            {"role": "user", "content": "other q"},
            {"role": "assistant", "content": "other a"},
        ]
        assert [m.content for m in sessions.get(source).messages] == ["old question", "PARTIAL"]


_RESTORED_BODY = "# Title\n\n- a\n- b\n\n```py\nx = 1\n```\n"


async def test_resume_shows_message_bodies_line_for_line_without_escapes(sessions):
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    sessions.set_title(session_id, "Seeded", TitleSource.AUTO)
    sessions.add_message(
        session_id,
        SessionMessage(role=MessageRole.USER, content=_RESTORED_BODY + "\x1b[31mred\x1b[0m"),
    )
    sessions.add_message(
        session_id,
        SessionMessage(role=MessageRole.ASSISTANT, content=_RESTORED_BODY + "\x1b]0;x\x07end"),
    )
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(session_id)
        await pilot.pause()
        assert _bubble_text(screen.query(AssistantMessage).first()) == _RESTORED_BODY + "end"
        user_text = str(
            screen.query(UserMessage).first().query(".message-content").first().render()
        )
        assert user_text.startswith(_RESTORED_BODY)
        assert "\x1b" not in user_text


def _source_chunk(
    source: str, content_type: str = "text", pages: int = 0, lines: tuple[int, int] = (0, 0)
):
    from lilbee.data.store import SearchChunk

    return SearchChunk(
        source=source,
        content_type=content_type,
        chunk="text",
        page_start=pages,
        page_end=pages,
        line_start=lines[0],
        line_end=lines[1],
        chunk_index=0,
        vector=[0.1],
    )


def _shown_sources(source: str) -> str:
    """The Sources list a bubble's markdown *source* shows, one line per hard break,
    as the widget's parser reads it: no ordered list, so no renumbering."""
    from lilbee.cli.tui.widgets.message import _answer_markdown_parser

    tokens = _answer_markdown_parser().parse(source.rpartition("\n\nSources:\n\n")[2])
    assert "ordered_list_open" not in {t.type for t in tokens}
    inline = next(t for t in tokens if t.type == "inline")
    return "".join(
        "\n" if c.type == "hardbreak" else c.content
        for c in inline.children or []
        if c.content or c.type == "hardbreak"
    )


async def test_a_streamed_turn_saves_its_sources_beside_the_answer(sessions):
    """The answer is saved without its Sources list; the list is saved as refs, and
    a resumed session shows the same list as the live turn did."""
    from lilbee.retrieval.query.formatting import AnswerSources

    def stream(*_args, **_kwargs):
        yield StreamToken(content="Oil [2].", is_reasoning=False)
        yield AnswerSources([_source_chunk("a.md"), _source_chunk("b.md")], {})

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with (
            patch.object(ChatScreen, "_await_chat_engine", return_value=True),
            patch.object(get_services().searcher, "ask_stream", side_effect=stream),
        ):
            screen._send_message("oil?")
            assert await pump_until(pilot, lambda: not screen.streaming)
            await pilot.pause()
        session_id = screen._session_id
        live = _bubble_text(screen.query(AssistantMessage).last())
        saved = sessions.get(session_id).messages[-1]
        assert saved.content == "Oil [2]."
        assert saved.source_refs == (StoredSourceRef("a.md", 1), StoredSourceRef("b.md", 2))
        assert "1. [a.md](file://" in live and "2. [b.md](file://" in live
        screen.resume_session(session_id)
        await pilot.pause()
        assert _bubble_text(screen.query(AssistantMessage).last()) == live


async def test_a_live_answer_that_keeps_a_sources_heading_shows_its_sources_list(sessions):
    """An answer whose text keeps a "Sources:" heading before prose still gets lilbee's
    list from its refs, and the resumed turn shows the live one whole."""
    from lilbee.retrieval.query.formatting import AnswerSources

    prose = "Answer [1].\n\nSources:\nI used the documents provided."

    def stream(*_args, **_kwargs):
        yield StreamToken(content=prose, is_reasoning=False)
        yield AnswerSources([_source_chunk("a.md")], {})

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with (
            patch.object(ChatScreen, "_await_chat_engine", return_value=True),
            patch.object(get_services().searcher, "ask_stream", side_effect=stream),
        ):
            screen._send_message("q?")
            assert await pump_until(pilot, lambda: not screen.streaming)
            await pilot.pause()
        live = _bubble_text(screen.query(AssistantMessage).last())
        assert live.startswith(prose)
        assert "1. [a.md](file://" in live
        screen.resume_session(screen._session_id)
        await pilot.pause()
        assert _bubble_text(screen.query(AssistantMessage).last()) == live


async def test_a_restored_legacy_turn_keeps_the_prose_after_its_own_sources_heading(sessions):
    """A turn saved with no refs, whose text has a model-written "Sources:" heading and
    no lilbee list after it, restores whole."""
    prose = "Answer [1].\n\nSources:\nI used the documents provided."
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="q"))
    sessions.add_message(session_id, SessionMessage(role=MessageRole.ASSISTANT, content=prose))
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(session_id)
        await pilot.pause()
        assert _bubble_text(screen.query(AssistantMessage).first()) == prose


async def test_a_restored_cited_subset_shows_its_true_numbers(sessions):
    """A turn a client saved with the cited subset [1], [3] shows 1. and 3., not 1. and 2."""
    session_id = sessions.create(model_ref=cfg.chat_model, scope="both")
    sessions.add_message(session_id, SessionMessage(role=MessageRole.USER, content="q"))
    sessions.add_message(
        session_id,
        SessionMessage(
            role=MessageRole.ASSISTANT,
            content="See [1] and [3].",
            sources=("first.md", "third.md"),
            source_refs=(StoredSourceRef("first.md", 1), StoredSourceRef("third.md", 3)),
        ),
    )
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        screen.resume_session(session_id)
        await pilot.pause()
        shown = _shown_sources(_bubble_text(screen.query(AssistantMessage).first()))
        assert shown == "1. first.md\n3. third.md"


async def test_a_resumed_turn_keeps_the_page_and_line_locators_it_showed_live(sessions):
    """A PDF page and a code line span shown in the live Sources list come back on resume."""
    from lilbee.retrieval.query.formatting import AnswerSources

    def stream(*_args, **_kwargs):
        yield StreamToken(content="See [1] and [2].", is_reasoning=False)
        yield AnswerSources(
            [
                _source_chunk("manual.pdf", "pdf", pages=3),
                _source_chunk("tool.py", "code", lines=(10, 20)),
            ],
            {},
        )

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        with (
            patch.object(ChatScreen, "_await_chat_engine", return_value=True),
            patch.object(get_services().searcher, "ask_stream", side_effect=stream),
        ):
            screen._send_message("q?")
            assert await pump_until(pilot, lambda: not screen.streaming)
            await pilot.pause()
        session_id = screen._session_id
        live = _bubble_text(screen.query(AssistantMessage).last()).rpartition("Sources:")[2]
        assert ", page 3\n" in live and live.endswith(", lines 10-20")
        screen.resume_session(session_id)
        await pilot.pause()
        restored = _bubble_text(screen.query(AssistantMessage).last()).rpartition("Sources:")[2]
        assert restored == live
        assert sessions.get(session_id).messages[-1].source_refs == (
            StoredSourceRef("manual.pdf", 1, "page 3"),
            StoredSourceRef("tool.py", 2, "lines 10-20"),
        )


# Locators a client may post: a link, a link target, inline markup, a block after a
# line break, an autolink.
_ATTACK_LOCATORS = [
    "[click](https://evil.example/x)",
    "x](file:///etc/passwd)",
    "page 3](file:///etc/passwd)",
    "page 3 **bold** ![px](https://evil.example/p.png)",
    "page 3\n\n# Heading",
    "<https://evil.example/auto>",
]
# Sources a client may post whose label reads as a link elsewhere: one with no file
# (a NUL byte), one with markup, ones whose label holds a ``](file://`` target.
_ATTACK_SOURCES = [
    ("a.md", "page 3"),
    ("[b.md](file:///etc/passwd)\x00", ""),
    ("b.md](file:///etc/passwd)", "pages 2-4"),
    ("[b.md](file:///etc/passwd)", ""),
    ("memory:[click](https://evil.example/m) **m**", "line 7"),
    ("[c.md](file:///etc/passwd)\x1b", "lines 1-2"),
]
_MARKUP_TOKENS = {"image", "strong_open", "em_open", "heading_open", "html_inline", "html_block"}


def _markup_in(markdown: str) -> list[str]:
    """Each web link, image, emphasis, heading or html token the answer widget's parser
    makes. A bare URL the parser links as written (as in any answer text) is not markup."""
    found = []
    for token in _answer_markdown_parser().parse(markdown):
        tokens = [token, *(token.children or [])]
        for t, after in zip(tokens, [*tokens[1:], None], strict=True):
            href = str(t.attrGet("href"))
            as_written = t.markup == "linkify" and after is not None and after.content == href
            if t.type == "link_open" and not href.startswith("file://") and not as_written:
                found.append(f"link:{href}")
            elif t.type in _MARKUP_TOKENS:
                found.append(t.type)
    return found


def _list_lines(markdown: str) -> list[str]:
    """The shown text of each item of the Sources list *markdown* ends in."""
    lines = []
    for t in _answer_markdown_parser().parse(markdown.rpartition("Sources:")[2]):
        if t.type == "inline" and t.level == 3:
            lines.append("".join(c.content for c in t.children or [] if c.type == "text"))
    return lines


def _line_links(markdown: str) -> list[list[str]]:
    """The target of each link on each item of the Sources list *markdown* ends in. A
    bare URL the parser links as written shows its own target, so it is not counted."""
    lines = []
    for t in _answer_markdown_parser().parse(markdown.rpartition("Sources:")[2]):
        if t.type == "inline" and t.level == 3:
            kids = t.children or []
            hrefs = []
            for c, after in zip(kids, [*kids[1:], None], strict=True):
                href = str(c.attrGet("href"))
                as_written = c.markup == "linkify" and after is not None and after.content == href
                if c.type == "link_open" and not as_written:
                    hrefs.append(href)
            lines.append(hrefs)
    return lines


def _own_file_url(source: str) -> str | None:
    """The ``file://`` URL of *source*'s own file, or None when it has none."""
    from lilbee.data.store import is_memory_source
    from lilbee.retrieval.query.formatting import _source_file_url

    return None if is_memory_source(source) else _source_file_url(source)


def _post_both(sessions, refs: list[dict]) -> tuple[int, dict, str, str]:
    """*refs* posted over HTTP and over MCP: the HTTP status, the MCP result and both ids."""
    from litestar.testing import TestClient

    import lilbee.server.auth as auth_mod
    from lilbee.mcp_server import session_add_message, session_create
    from lilbee.server.app import create_app

    content = "See " + " ".join(f"[{r['citation_number']}]" for r in refs) + "."
    auth_mod.session_manager.disable()
    try:
        client = TestClient(create_app())
        with patch.object(cfg, "mcp_sessions_enabled", True):
            http_id = sessions.create(model_ref="m", scope="both")
            response = client.post(
                f"/api/sessions/{http_id}/messages",
                json={"role": "assistant", "content": content, "source_refs": refs},
            )
            mcp_id = session_create("m")["id"]
            result = session_add_message(mcp_id, "assistant", content, source_refs=refs)
    finally:
        auth_mod.session_manager.cleanup()
    return response.status_code, result, http_id, mcp_id


@pytest.mark.parametrize("locator", _ATTACK_LOCATORS)
def test_a_locator_lilbee_never_writes_is_refused_over_http_and_mcp(sessions, locator):
    """HTTP answers 422 and MCP an error naming the locator forms; nothing is saved."""
    refs = [{"source": "a.md", "citation_number": 1, "locator": locator}]
    status, result, http_id, mcp_id = _post_both(sessions, refs)
    assert status == 422
    assert "page N" in result["error"] and repr(locator) in result["error"]
    assert not sessions.get(http_id).messages and not sessions.get(mcp_id).messages


async def test_every_link_in_the_resumed_tui_list_targets_its_own_source(sessions):
    """Sources posted over HTTP and over MCP resume in the TUI with each line's link,
    file:// included, on the file of that line's own source, or no link when it has
    none; no web link, image, emphasis, heading or html; each locator as written."""
    refs = [
        {"source": source, "citation_number": n, "locator": locator}
        for n, (source, locator) in enumerate(_ATTACK_SOURCES, 1)
    ]
    status, result, http_id, mcp_id = _post_both(sessions, refs)
    assert status == 201 and "error" not in result
    shown = {}
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await await_chat(app, pilot)
        for surface, session_id in (("http", http_id), ("mcp", mcp_id)):
            screen.resume_session(session_id)
            await pilot.pause()
            shown[surface] = _bubble_text(screen.query(AssistantMessage).last())
    expected = [[url] if (url := _own_file_url(s)) else [] for s, _ in _ATTACK_SOURCES]
    assert expected[0] and not expected[1], "the instrument sees a linked and an unlinked line"
    assert _markup_in("x [y](https://evil.example/c) **b** # c"), "the instrument finds markup"
    for surface, markdown in shown.items():
        assert _line_links(markdown) == expected, surface
        assert _markup_in(markdown) == [], surface
        for line, (_, locator) in zip(_list_lines(markdown), _ATTACK_SOURCES, strict=True):
            assert line.endswith(f", {locator}" if locator else ""), (surface, line)
