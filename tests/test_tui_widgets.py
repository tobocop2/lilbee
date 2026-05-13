"""Tests for Textual TUI widgets: 100 % coverage target."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Static

from conftest import (
    TEST_EMBED_REF,
    TEST_LOCAL_REF,
)
from conftest import (
    make_test_catalog_model as _make_model,
)
from lilbee.cli.tui.screens.catalog_utils import (
    CatalogRow,
    FrontierCatalogRow,
    KeyStatus,
    LocalCatalogRow,
)
from lilbee.cli.tui.widgets.model_bar import ModelOption
from lilbee.core.config import cfg
from tests._lilbee_app_test_host import LilbeeAppHost


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class _MsgApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage

        yield UserMessage("hello")
        self._am = AssistantMessage()
        yield self._am


class TestUserMessage:
    def test_renders_text(self) -> None:
        from lilbee.cli.tui.widgets.message import UserMessage

        msg = UserMessage("hi")
        assert "user-message" in msg.classes

    def test_has_speaker_label(self) -> None:
        from lilbee.cli.tui.widgets.message import UserMessage

        msg = UserMessage("hi")
        children = list(msg.compose())
        assert len(children) == 2  # speaker label + content


class TestAssistantMessageAsync:
    async def test_compose_yields_speaker_content_citation(self) -> None:
        """Compose yields three children: speaker label, content, citation. No
        Collapsible up front; the reasoning fold is mounted lazily inside
        ``on_mount``/``append_reasoning``.
        """
        from lilbee.cli.tui.widgets.message import AssistantMessage

        am = AssistantMessage()
        children = list(am.compose())
        assert len(children) == 3
        # First child carries the lilbee speaker label markup.
        assert "lilbee" in str(children[0].render())
        assert am._reasoning_widget is None
        assert am._content_widget is not None
        assert am._citation_widget is not None

    async def test_on_mount_attaches_thinking_header(self) -> None:
        """Mounting the message inserts a sibling ``ThinkingHeader`` above content."""
        from lilbee.cli.tui.widgets.thinking_header import ThinkingHeader

        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            assert am._thinking_header is not None
            assert isinstance(am._thinking_header, ThinkingHeader)
            assert am._thinking_header.is_mounted

    async def test_first_reasoning_token_mounts_streaming_collapsible(self) -> None:
        """The Collapsible appears only when the first reasoning token arrives,
        carrying the ``-streaming`` modifier so the toggle row is hidden by CSS.
        """
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            assert am._reasoning_widget is None
            am.append_reasoning("step 1")
            await pilot.pause()
            assert am._reasoning_widget is not None
            assert am._reasoning_widget.collapsed is False
            assert "reasoning-block" in am._reasoning_widget.classes
            assert "-streaming" in am._reasoning_widget.classes

    async def test_append_reasoning_debounces_static_updates(self) -> None:
        """Reasoning bursts collapse to one ``Static.update``; ``finish`` flushes the tail."""
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.append_reasoning("a ")
            assert am._reasoning_static is not None
            with mock.patch.object(am._reasoning_static, "update") as mock_update:
                am.append_reasoning("b ")
                am.append_reasoning("c ")
                # The first append fired an update on its own; subsequent
                # bursts inside the 0.1 s debounce window do not.
                assert mock_update.call_count == 0
                am.finish(sources=None)
                # finish() flushes the buffered tail.
                assert mock_update.call_count == 1
                last_call_text = mock_update.call_args_list[-1].args[0]
                assert last_call_text == "a b c "

    async def test_append_content_updates_markdown(self) -> None:
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.append_content("token1")
            am.append_content("token2")
            assert am._content_parts == ["token1", "token2"]

    async def test_first_content_token_dismisses_thinking_header(self) -> None:
        """Without any reasoning, the first content token retires the header."""
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            assert am._thinking_header is not None
            am.append_content("hi")
            await pilot.pause()
            assert am._thinking_header is None

    async def test_first_content_token_after_reasoning_keeps_header(self) -> None:
        """When reasoning has already fired, the header stays until ``finish``."""
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.append_reasoning("step 1")
            am.append_content("answer")
            await pilot.pause()
            assert am._thinking_header is not None

    async def test_reasoning_after_content_mounts_collapsible_before_content(self) -> None:
        """Late reasoning (after content already dismissed the header) mounts the
        Collapsible relative to the existing content widget, not the missing header.
        """
        from textual.widgets import Collapsible

        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.append_content("answer first")
            await pilot.pause()
            assert am._thinking_header is None, "content should have dismissed the header"
            am.append_reasoning("late thought")
            await pilot.pause()
            assert isinstance(am._reasoning_widget, Collapsible)
            assert am._reasoning_widget.is_mounted

    async def test_finish_with_sources_shows_citations(self) -> None:
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.append_reasoning("think")
            am.append_content("answer")
            am.finish(sources=["doc.pdf:1"])
            assert am._finished is True
            assert am._reasoning_widget is not None
            assert "reasoning" in am._reasoning_widget.title
            assert "token" in am._reasoning_widget.title
            assert am._reasoning_widget.collapsed is True
            assert "-streaming" not in am._reasoning_widget.classes
            assert am._thinking_header is None

    async def test_finish_without_reasoning_leaves_widget_unmounted(self) -> None:
        """No reasoning emitted => no Collapsible was ever mounted."""
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.append_content("hi")
            am.finish(sources=None)
            assert am._finished is True
            assert am._reasoning_widget is None
            assert am._thinking_header is None

    async def test_finish_without_sources_hides_citation(self) -> None:
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.finish(sources=None)
            assert am._citation_widget is not None
            assert am._citation_widget.display is False

    async def test_finish_with_empty_sources_hides_citation(self) -> None:
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.finish(sources=[])
            assert am._citation_widget is not None
            assert am._citation_widget.display is False

    async def test_markdown_rendering_true_uses_markdown_widget(self) -> None:
        from textual.widgets import Markdown

        cfg.markdown_rendering = True
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            assert am.use_markdown is True
            assert isinstance(am._content_widget, Markdown)

    async def test_markdown_rendering_false_uses_static_widget(self) -> None:
        from textual.widgets import Markdown

        cfg.markdown_rendering = False
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            assert am.use_markdown is False
            assert not isinstance(am._content_widget, Markdown)
            assert isinstance(am._content_widget, Static)

    async def test_rebuild_content_widget_toggles_type(self) -> None:
        from textual.widgets import Markdown

        cfg.markdown_rendering = True
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            assert isinstance(am._content_widget, Markdown)
            am.append_content("hello")
            await am.rebuild_content_widget(use_markdown=False)
            assert isinstance(am._content_widget, Static)
            assert not isinstance(am._content_widget, Markdown)
            assert am.use_markdown is False

    async def test_rebuild_content_widget_noop_when_no_widget(self) -> None:
        from lilbee.cli.tui.widgets.message import AssistantMessage

        app = _MsgApp()
        async with app.run_test():
            am = AssistantMessage()
            am._content_widget = None
            await am.rebuild_content_widget(use_markdown=False)
            assert am._content_widget is None

    async def test_on_mount_noop_when_content_widget_missing(self) -> None:
        """on_mount returns early if compose was bypassed (defensive guard)."""
        from lilbee.cli.tui.widgets.message import AssistantMessage

        app = _MsgApp()
        async with app.run_test():
            am = AssistantMessage()
            am._content_widget = None
            am.on_mount()
            assert am._thinking_header is None


class _ThinkingHeaderApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.thinking_header import ThinkingHeader

        self._header = ThinkingHeader()
        yield self._header


class TestThinkingHeader:
    """The thinking header animates a Knight-Rider bouncing block until stopped."""

    def test_frame_content_shifts_filled_block_with_frame(self) -> None:
        """Frame N renders exactly one filled block; consecutive frames differ."""
        from lilbee.cli.tui.widgets.thinking_header import (
            _BLOCK_EMPTY,
            _BLOCK_FILLED,
            _TRACK_CELLS,
            _frame_content,
        )

        rendered_frame_0 = str(_frame_content(0))
        rendered_frame_1 = str(_frame_content(1))
        # Exactly one cell is lit per frame; the rest are dim.
        assert rendered_frame_0.count(_BLOCK_FILLED) == 1
        assert rendered_frame_1.count(_BLOCK_FILLED) == 1
        assert rendered_frame_0.count(_BLOCK_EMPTY) == _TRACK_CELLS - 1
        assert rendered_frame_0 != rendered_frame_1

    def test_frame_content_bounces_back_at_track_end(self) -> None:
        """At the right edge the lit block reverses direction (Knight Rider)."""
        from lilbee.cli.tui.widgets.thinking_header import _TRACK_CELLS, _bounce_position

        # Forward sweep occupies frames 0..cells-1.
        for f in range(_TRACK_CELLS):
            assert _bounce_position(f) == f
        # Backward sweep retraces cells-2..1 (skips both endpoints to
        # avoid stalling for two ticks at the turnaround).
        for offset in range(1, _TRACK_CELLS - 1):
            f = _TRACK_CELLS - 1 + offset
            assert _bounce_position(f) == _TRACK_CELLS - 1 - offset
        # And the cycle restarts at the left edge.
        cycle = 2 * (_TRACK_CELLS - 1)
        assert _bounce_position(cycle) == 0

    async def test_tick_advances_frame_and_repaints(self) -> None:
        """Calling _tick increments the internal frame counter and repaints."""
        app = _ThinkingHeaderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            header = app._header
            start = header._frame
            with mock.patch.object(header, "update") as update_mock:
                header._tick()
                assert header._frame == start + 1
                update_mock.assert_called_once()

    async def test_redirect_to_routes_frames_to_target(self) -> None:
        """``redirect_to`` swaps the render target so callers can intercept frames."""
        app = _ThinkingHeaderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            header = app._header
            sink = mock.Mock()
            header.redirect_to(sink)
            with mock.patch.object(header, "update") as update_mock:
                header._tick()
                update_mock.assert_not_called()
                sink.assert_called_once()

    async def test_on_unmount_stops_timer(self) -> None:
        """Unmounting the header cancels the running interval."""
        app = _ThinkingHeaderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            header = app._header
            assert header._timer is not None
            header.on_unmount()
            assert header._timer is None

    async def test_stop_is_idempotent(self) -> None:
        """``stop`` can be called repeatedly without raising."""
        app = _ThinkingHeaderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            header = app._header
            header.stop()
            header.stop()
            assert header._timer is None


class _HelpApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("bg")

    def key_f1(self) -> None:
        self.action_show_help_panel()


class TestHelpPanel:
    async def test_show_help_panel(self) -> None:
        app = _HelpApp()
        async with app.run_test() as pilot:
            app.action_show_help_panel()
            await pilot.pause()
            assert app.screen.query("HelpPanel")

    async def test_hide_help_panel(self) -> None:
        app = _HelpApp()
        async with app.run_test() as pilot:
            app.action_show_help_panel()
            await pilot.pause()
            assert app.screen.query("HelpPanel")
            app.action_hide_help_panel()
            await pilot.pause()
            assert not app.screen.query("HelpPanel")


class _TaskBarApp(LilbeeAppHost):
    def __init__(self) -> None:
        super().__init__()
        from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        yield TaskBar(id="task-bar")


class TestTaskBar:
    async def test_spawning_workers_template_singular(self) -> None:
        """Single role -> 'Starting <role> worker...'."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            assert bar._spawning_workers_template(["chat"]) == "Starting chat worker..."

    async def test_spawning_workers_template_plural(self) -> None:
        """Multiple roles -> 'Starting <a>, <b> workers...'."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            out = bar._spawning_workers_template(["chat", "embed"])
            assert out == "Starting chat, embed workers..."

    async def test_renders_spawning_summary_when_idle_with_spawn(self) -> None:
        """When idle except for in-flight worker spawns, the bar surfaces the
        'Starting <role> worker...' summary instead of hiding."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            app.task_bar.mark_role_spawning("chat")
            bar._refresh_display()
            await pilot.pause()
            assert bar.display is True

    async def test_hidden_when_empty(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            assert bar.display is False

    async def test_shows_active_task(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            bar.add_task("Sync docs", "sync")
            bar.queue.advance()
            bar._refresh_display()
            await pilot.pause()
            assert bar.display is True

    async def test_shows_multiple_queued(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            bar.add_task("Download A", "download")
            bar.queue.advance()
            bar.add_task("Sync", "sync")
            bar.add_task("Crawl", "crawl")
            bar._refresh_display()
            await pilot.pause()
            assert bar.display is True
            assert len(bar.queue.queued_tasks) == 2

    async def test_complete_removes_after_flash(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            task_id = bar.add_task("Sync", "sync")
            bar.queue.advance()
            bar.complete_task(task_id)
            await pilot.pause()
            # After flash timer fires, task is removed
            await pilot.pause(delay=2.5)
            assert bar.queue.is_empty

    async def test_unmount_cancels_poll_interval(self) -> None:
        """bb-3uzp: on_unmount must stop the 10 Hz poll interval.

        Without this, a detached TaskBar's interval keeps firing after a
        screen push/pop cycle and can set ``display=False`` on the new
        TaskBar mid-render, making the bar vanish from chat.
        """
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            assert getattr(bar, "_interval", None) is not None
            await bar.remove()
            await pilot.pause()
            assert getattr(bar, "_interval", None) is None

    async def test_queue_advances_on_complete(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            t1 = bar.add_task("Download A", "download")
            bar.queue.advance()
            bar.add_task("Sync B", "sync")
            bar.complete_task(t1)
            # After flash, next task should advance
            await pilot.pause(delay=2.5)
            active = bar.queue.active_task
            assert active is not None
            assert active.name == "Sync B"

    async def test_cancel_removes_immediately(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            task_id = bar.add_task("Sync", "sync")
            bar.queue.advance()
            bar.cancel_task(task_id)
            await pilot.pause()
            assert bar.queue.is_empty

    async def test_update_task_changes_progress(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            task_id = bar.add_task("Download", "download")
            bar.queue.advance()
            bar.update_task(task_id, 42, "21/50 MB")
            await pilot.pause()
            assert bar.queue.active_task is not None
            assert bar.queue.active_task.progress == 42

    async def test_fail_task_shows_then_removes(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            task_id = bar.add_task("Download", "download")
            bar.queue.advance()
            bar.fail_task(task_id, "Network error")
            await pilot.pause(delay=2.5)
            assert bar.queue.is_empty

    async def test_app_task_bar_ref(self) -> None:
        """TaskBarController is accessible via app.task_bar from other screens."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar
        from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            assert isinstance(app.task_bar, TaskBarController)
            assert bar.queue is app.task_bar.queue


class _ModelBarApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        yield ModelBar()


@pytest.mark.usefixtures("wiki_enabled")
class TestModelBar:
    """ModelBar tests assert the scope picker, which only renders when wiki is on."""

    @pytest.fixture(autouse=True)
    def mock_classify(self):
        empty = ([], [])
        with mock.patch(
            "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
            return_value=empty,
        ):
            yield

    async def test_renders_picker_buttons_and_no_scope_select(self) -> None:
        """ModelBar mounts only the two model pickers; scope lives in ScopeChip."""
        from textual.widgets import Select

        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            buttons = list(app.query(ModelPickerButton))
            assert len(buttons) == 2
            assert list(app.query(Select)) == []

    async def test_button_ids_are_present(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#chat-model-button", ModelPickerButton) is not None
            assert app.query_one("#embed-model-button", ModelPickerButton) is not None

    async def test_labels_rendered(self) -> None:
        """Chat/Embed labels render as pills (label inside the styled-space padding)."""
        from textual.widgets import Static

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            pills = [str(s.render()) for s in app.query(Static) if "model-bar-pill" in s.classes]
            assert any(" Chat " in p for p in pills)
            assert any(" Embed " in p for p in pills)

    async def test_chat_mode_toggle_renders_search_when_embedding_ready(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "search"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=True):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                assert "-disabled" not in toggle.classes
                search_pill = toggle.query_one("#chat-mode-search", Static)
                assert "-active" in search_pill.classes

    async def test_chat_mode_toggle_disabled_without_embedding(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "search"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=False):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                assert "-disabled" in toggle.classes
                chat_pill = toggle.query_one("#chat-mode-chat", Static)
                assert "-active" in chat_pill.classes

    async def test_chat_mode_toggle_flips_on_click(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "search"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=True):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                assert toggle.toggle() is True
                assert cfg.chat_mode == "chat"
                assert toggle.toggle() is True
                assert cfg.chat_mode == "search"

    async def test_chat_mode_toggle_no_op_when_disabled(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "chat"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=False):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                assert toggle.toggle() is False
                assert cfg.chat_mode == "chat"

    async def test_chat_mode_toggle_action_flip_mode_invokes_toggle(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "search"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=True):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                toggle.action_flip_mode()
                assert cfg.chat_mode == "chat"

    async def test_chat_mode_toggle_repaints_when_embedding_set(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle, ModelBar

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "chat"
        with mock.patch(
            "lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=False
        ) as patched:
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                assert "-disabled" in toggle.classes
                patched.return_value = True
                app.query_one(ModelBar)._refresh_chat_mode_toggle()
                await pilot.pause()
                assert "-disabled" not in toggle.classes

    async def test_chat_mode_toggle_left_selects_search(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "chat"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=True):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                toggle.action_select_search()
                assert cfg.chat_mode == "search"

    async def test_chat_mode_toggle_right_selects_chat(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "search"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=True):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                toggle.action_select_chat()
                assert cfg.chat_mode == "chat"

    async def test_chat_mode_toggle_select_chat_when_already_chat_is_noop(self) -> None:
        """Selecting the already-active half is a no-op; cfg is not rewritten."""
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "chat"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=True):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                # already in chat mode; selecting chat returns False
                assert toggle._set_mode("chat") is False
                assert cfg.chat_mode == "chat"

    async def test_chat_mode_toggle_select_search_no_op_when_disabled(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "chat"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=False):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                toggle.action_select_search()
                assert cfg.chat_mode == "chat"

    async def test_chat_mode_toggle_renders_two_pill_children(self) -> None:
        """Active half wears ``-active``; the search half disables when embedding is missing."""
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "search"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=True):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                search = toggle.query_one("#chat-mode-search", Static)
                chat = toggle.query_one("#chat-mode-chat", Static)
                assert "Search" in str(search.render())
                assert "Chat" in str(chat.render())
                assert "-active" in search.classes
                assert "-active" not in chat.classes
                assert "-disabled" not in search.classes
                assert "chat-mode-pill" in search.classes
                assert "chat-mode-pill" in chat.classes

    async def test_chat_mode_toggle_disabled_class_lives_on_search_pill(self) -> None:
        """When embedding is missing, the search pill alone gets the disabled mark."""
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "chat"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=False):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                search = toggle.query_one("#chat-mode-search", Static)
                chat = toggle.query_one("#chat-mode-chat", Static)
                assert "-disabled" in search.classes
                assert "-disabled" not in chat.classes
                assert "-active" in chat.classes

    async def test_picker_buttons_have_tooltips(self) -> None:
        """Chat and Embed pickers expose hover tooltips like the scope chip does."""
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat_btn = app.query_one("#chat-model-button", ModelPickerButton)
            embed_btn = app.query_one("#embed-model-button", ModelPickerButton)
            assert chat_btn.tooltip == msg.MODEL_PICKER_CHAT_TOOLTIP
            assert embed_btn.tooltip == msg.MODEL_PICKER_EMBED_TOOLTIP

    async def test_cloud_warning_hidden_for_local_model(self) -> None:
        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            warning = app.query_one("#cloud-provider-warning", Static)
            assert "-visible" not in warning.classes

    async def test_cloud_warning_hidden_for_ollama_model(self) -> None:
        cfg.chat_model = "ollama/qwen3:8b"
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            warning = app.query_one("#cloud-provider-warning", Static)
            assert "-visible" not in warning.classes

    async def test_cloud_warning_visible_and_names_provider(self) -> None:
        cfg.chat_model = "openai/gpt-4o"
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            warning = app.query_one("#cloud-provider-warning", Static)
            assert "-visible" in warning.classes
            rendered = str(warning.render())
            assert "OpenAI" in rendered
            assert "sensitive" in rendered.lower()


class TestCloudProviderLabel:
    def test_empty_chat_model_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _cloud_provider_label

        assert _cloud_provider_label("") is None


@pytest.mark.usefixtures("wiki_enabled")
class TestModelPickerButton:
    """ModelPickerButton labels mirror cfg and route picks via apply_active_model."""

    @pytest.fixture(autouse=True)
    def mock_classify(self):
        with mock.patch(
            "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
            return_value=([], []),
        ):
            yield

    async def test_button_label_reflects_cfg_chat_model(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#chat-model-button", ModelPickerButton)
            rendered = str(btn.render())
            assert TEST_LOCAL_REF in rendered or "Test" in rendered

    async def test_populate_repaints_buttons_when_options_arrive(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelBar, ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar._populate(
                [ModelOption("Test Chat", TEST_LOCAL_REF)],
                [ModelOption("Test Embed", TEST_EMBED_REF)],
            )
            await pilot.pause()
            chat_btn = app.query_one("#chat-model-button", ModelPickerButton)
            assert chat_btn._options[0].ref == TEST_LOCAL_REF

    async def test_picker_dismiss_with_new_ref_writes_cfg(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#chat-model-button", ModelPickerButton)
            new_ref = "ollama/new-chat:latest"
            with (
                mock.patch("lilbee.core.settings.set_value"),
                mock.patch("lilbee.cli.tui.widgets.model_bar.reset_services"),
            ):
                btn._on_picker_dismissed(new_ref)
                await pilot.pause()
            assert cfg.chat_model == new_ref

    async def test_picker_dismiss_same_ref_is_noop(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#chat-model-button", ModelPickerButton)
            write_tracker = mock.Mock()
            with mock.patch("lilbee.core.settings.set_value", write_tracker):
                btn._on_picker_dismissed(TEST_LOCAL_REF)
                await pilot.pause()
            write_tracker.assert_not_called()

    async def test_picker_dismiss_none_is_noop(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#chat-model-button", ModelPickerButton)
            write_tracker = mock.Mock()
            with mock.patch("lilbee.core.settings.set_value", write_tracker):
                btn._on_picker_dismissed(None)
                await pilot.pause()
            write_tracker.assert_not_called()
            assert cfg.chat_model == TEST_LOCAL_REF

    async def test_embed_picker_dismiss_writes_cfg_after_legacy_pin(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#embed-model-button", ModelPickerButton)
            new_ref = "ollama/new-embed:latest"
            store_mock = mock.MagicMock()
            with (
                mock.patch("lilbee.core.settings.set_value"),
                mock.patch("lilbee.cli.tui.widgets.model_bar.reset_services"),
                mock.patch(
                    "lilbee.cli.tui.widgets.model_bar.get_services",
                    return_value=mock.MagicMock(store=store_mock),
                ),
            ):
                btn._on_picker_dismissed(new_ref)
                await pilot.pause()
            store_mock.initialize_meta_if_legacy.assert_called_once()
            assert cfg.embedding_model == new_ref

    async def test_embed_picker_dismiss_same_ref_is_noop(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#embed-model-button", ModelPickerButton)
            write_tracker = mock.Mock()
            with mock.patch("lilbee.core.settings.set_value", write_tracker):
                btn._on_picker_dismissed(TEST_EMBED_REF)
                await pilot.pause()
            write_tracker.assert_not_called()

    async def test_embed_picker_dismiss_reloads_only_embed_role(self) -> None:
        """Embed swap respawns just the embed worker. Chat stream stays untouched.

        Regression for the mid-stream embed-swap hang: the old code called
        ``reset_services`` on every model change, which races the chat
        worker if a stream is in flight. The fix scopes the reset to the
        role that actually changed.
        """
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton
        from lilbee.providers.worker.transport import WorkerRole

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#embed-model-button", ModelPickerButton)
            new_ref = "ollama/new-embed:latest"
            services_mock = mock.MagicMock()
            with (
                mock.patch("lilbee.core.settings.set_value"),
                mock.patch("lilbee.cli.tui.widgets.model_bar.reset_services") as mock_reset,
                mock.patch(
                    "lilbee.cli.tui.widgets.model_bar.get_services",
                    return_value=services_mock,
                ),
            ):
                btn._on_picker_dismissed(new_ref)
                await pilot.pause()
            services_mock.reload_role.assert_called_once_with(WorkerRole.EMBED)
            mock_reset.assert_not_called()

    async def test_chat_picker_dismiss_does_not_reload_role(self) -> None:
        """Chat scope still routes through ``apply_model_change`` (cancel + reset).

        Chat-model swap is the legitimate cancel-and-restart UX. The new
        per-role reload path is only for siblings whose change should not
        disturb the in-flight chat stream.
        """
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#chat-model-button", ModelPickerButton)
            services_mock = mock.MagicMock()
            with (
                mock.patch("lilbee.core.settings.set_value"),
                mock.patch("lilbee.cli.tui.widgets.model_bar.reset_services"),
                mock.patch(
                    "lilbee.cli.tui.widgets.model_bar.get_services",
                    return_value=services_mock,
                ),
            ):
                btn._on_picker_dismissed("ollama/new-chat:latest")
                await pilot.pause()
            services_mock.reload_role.assert_not_called()

    async def test_picker_button_click_pushes_modal(self) -> None:
        from lilbee.cli.tui.screens.model_picker import ModelPickerModal
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#chat-model-button", ModelPickerButton)
            await pilot.click(btn)
            await pilot.pause()
            assert isinstance(app.screen, ModelPickerModal)

    async def test_chat_mode_toggle_click_flips_mode(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "search"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=True):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                await pilot.click(toggle)
                await pilot.pause()
                assert cfg.chat_mode == "chat"

    async def test_chat_mode_pill_click_routes_per_id(self) -> None:
        """Click on the search/chat pill calls ``_set_mode`` for that side only."""
        from textual import events

        from lilbee.cli.tui.widgets.model_bar import ChatModeToggle

        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        cfg.chat_mode = "chat"
        with mock.patch("lilbee.cli.tui.widgets.model_bar.is_model_available", return_value=True):
            app = _ModelBarApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                toggle = app.query_one(ChatModeToggle)
                with mock.patch.object(toggle, "_set_mode") as set_mode:
                    search = toggle.query_one("#chat-mode-search", Static)
                    click = mock.MagicMock(spec=events.Click)
                    click.widget = search
                    toggle.on_click(click)
                    set_mode.assert_called_once_with("search")
                with mock.patch.object(toggle, "_set_mode") as set_mode:
                    chat = toggle.query_one("#chat-mode-chat", Static)
                    click = mock.MagicMock(spec=events.Click)
                    click.widget = chat
                    toggle.on_click(click)
                    set_mode.assert_called_once_with("chat")


class _ScopeChipApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        yield ScopeChip(id="scope-chip")


@pytest.mark.usefixtures("wiki_enabled")
class TestScopeChip:
    """ScopeChip is the search-only filter; only visible when wiki is on AND chat_mode == search."""

    async def test_visible_when_search_mode_and_wiki_on(self) -> None:
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            assert "-hidden" not in chip.classes

    async def test_hidden_in_chat_mode(self) -> None:
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        cfg.chat_mode = "chat"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            assert "-hidden" in chip.classes

    async def test_hidden_when_wiki_disabled(self) -> None:
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        cfg.chat_mode = "search"
        cfg.wiki = False
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            assert "-hidden" in chip.classes

    async def test_scope_property_defaults_to_both(self) -> None:
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip
        from lilbee.data.store import SearchScope

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            assert chip.scope is SearchScope.BOTH

    async def test_active_pill_tracks_scope(self) -> None:
        """At rest, the BOTH pill carries -active; the others do not."""
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            both = chip.query_one("#scope-pill-both", Static)
            wiki = chip.query_one("#scope-pill-wiki", Static)
            raw = chip.query_one("#scope-pill-raw", Static)
            assert "-active" in both.classes
            assert "-active" not in wiki.classes
            assert "-active" not in raw.classes
            assert "scope-pill" in both.classes
            assert "scope-pill" in wiki.classes
            assert "scope-pill" in raw.classes

    async def test_cycle_walks_both_wiki_raw_and_back(self) -> None:
        """cycle_scope() advances Both -> Wiki -> Raw -> Both and repaints each step."""
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip
        from lilbee.data.store import SearchScope

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            wiki_pill = chip.query_one("#scope-pill-wiki", Static)
            raw_pill = chip.query_one("#scope-pill-raw", Static)
            both_pill = chip.query_one("#scope-pill-both", Static)
            assert chip.scope is SearchScope.BOTH
            assert chip.cycle_scope() is SearchScope.WIKI
            assert "-active" in wiki_pill.classes
            assert "-active" not in both_pill.classes
            assert chip.cycle_scope() is SearchScope.RAW
            assert "-active" in raw_pill.classes
            assert "-active" not in wiki_pill.classes
            assert chip.cycle_scope() is SearchScope.BOTH
            assert "-active" in both_pill.classes
            assert "-active" not in raw_pill.classes

    async def test_pill_click_routes_to_matching_scope(self) -> None:
        """A click on a child pill sets the scope to the value that pill represents."""
        from textual import events

        from lilbee.cli.tui.widgets.scope_chip import ScopeChip
        from lilbee.data.store import SearchScope

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            wiki_pill = chip.query_one("#scope-pill-wiki", Static)
            click = mock.MagicMock(spec=events.Click)
            click.widget = wiki_pill
            chip.on_click(click)
            click.stop.assert_called_once()
            assert chip.scope is SearchScope.WIKI
            raw_pill = chip.query_one("#scope-pill-raw", Static)
            click2 = mock.MagicMock(spec=events.Click)
            click2.widget = raw_pill
            chip.on_click(click2)
            assert chip.scope is SearchScope.RAW

    async def test_pill_click_on_unknown_widget_is_a_noop(self) -> None:
        """Clicks routed through an unknown id leave the scope untouched."""
        from textual import events

        from lilbee.cli.tui.widgets.scope_chip import ScopeChip
        from lilbee.data.store import SearchScope

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            stranger = mock.Mock()
            stranger.id = "not-a-pill"
            click = mock.MagicMock(spec=events.Click)
            click.widget = stranger
            chip.on_click(click)
            click.stop.assert_not_called()
            assert chip.scope is SearchScope.BOTH

    async def test_pill_click_with_no_widget_is_a_noop(self) -> None:
        """A click without a widget reference is dropped without raising."""
        from textual import events

        from lilbee.cli.tui.widgets.scope_chip import ScopeChip
        from lilbee.data.store import SearchScope

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            click = mock.MagicMock(spec=events.Click)
            click.widget = None
            chip.on_click(click)
            click.stop.assert_not_called()
            assert chip.scope is SearchScope.BOTH

    async def test_set_scope_to_current_is_a_noop(self) -> None:
        """Re-clicking the active pill keeps the scope and avoids redundant repaints."""
        from textual import events

        from lilbee.cli.tui.widgets.scope_chip import ScopeChip
        from lilbee.data.store import SearchScope

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            both_pill = chip.query_one("#scope-pill-both", Static)
            click = mock.MagicMock(spec=events.Click)
            click.widget = both_pill
            chip.on_click(click)
            assert chip.scope is SearchScope.BOTH

    async def test_on_settings_changed_chat_mode_recomputes_visibility(self) -> None:
        """A chat_mode flip in the signal payload toggles the chip visibility."""
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            assert "-hidden" not in chip.classes
            cfg.chat_mode = "chat"
            chip._on_settings_changed(("chat_mode", "chat"))
            assert "-hidden" in chip.classes

    async def test_on_settings_changed_unrelated_key_is_a_noop(self) -> None:
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            chip._on_settings_changed(("temperature", 0.5))
            assert "-hidden" not in chip.classes

    async def test_pills_render_label_constants(self) -> None:
        """Each child pill renders the label constant from messages.py."""
        from lilbee.cli.tui import messages as msg_mod
        from lilbee.cli.tui.widgets.scope_chip import ScopeChip

        cfg.chat_mode = "search"
        cfg.wiki = True
        app = _ScopeChipApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chip = app.query_one(ScopeChip)
            assert (
                str(chip.query_one("#scope-pill-both", Static).render()) == msg_mod.SCOPE_PILL_BOTH
            )
            assert (
                str(chip.query_one("#scope-pill-wiki", Static).render()) == msg_mod.SCOPE_PILL_WIKI
            )
            assert str(chip.query_one("#scope-pill-raw", Static).render()) == msg_mod.SCOPE_PILL_RAW


def _make_local_row(name: str = "Local Model", installed: bool = False) -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task="chat",
        params="8B",
        size="4.0 GB",
        quant="Q4_0",
        downloads="1K",
        featured=False,
        installed=installed,
        sort_downloads=1000,
        sort_size=4.0,
        ref=name.lower().replace(" ", "/"),
    )


def _make_frontier_row(
    name: str = "gpt-test", provider: str = "OpenAI", ready: bool = True
) -> FrontierCatalogRow:
    return FrontierCatalogRow(
        name=name,
        ref=f"openai/{name}",
        task="chat",
        provider=provider,
        provider_id=provider.lower(),
        key_status=KeyStatus.READY if ready else KeyStatus.MISSING_KEY,
    )


class _ModelListApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.model_list import ModelList

        yield ModelList(id="model-list")


class TestModelPickerModal:
    """ModelPickerModal filters options by typed search and dismisses with selected ref."""

    async def test_modal_dismisses_with_selected_ref(self) -> None:
        from textual.widgets import Button

        from lilbee.cli.tui.screens.model_picker import ModelPickerModal
        from lilbee.cli.tui.widgets.model_list import ModelList

        opts = [
            ModelOption("Qwen3 0.6B", "qwen3-0.6b"),
            ModelOption("Llama 8B", "llama-8b"),
        ]

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield Button("anchor")

        app = _App()
        results: list[str | None] = []
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            def _capture(value: str | None) -> None:
                results.append(value)

            app.push_screen(ModelPickerModal(scope="chat", options=opts), _capture)
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ModelPickerModal)
            ml = modal.query_one("#picker-list", ModelList)
            ml.highlighted = 0
            ml.action_select()
            await pilot.pause()
            assert "qwen3-0.6b" in results

    async def test_modal_consecutive_keystrokes_stop_prior_debounce_timer(self) -> None:
        from textual.widgets import Button, Input

        from lilbee.cli.tui.screens.model_picker import ModelPickerModal
        from lilbee.cli.tui.widgets.model_list import ModelList

        opts = [ModelOption("Qwen3 0.6B", "qwen3-0.6b"), ModelOption("Llama 8B", "llama-8b")]

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield Button("anchor")

        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ModelPickerModal(scope="chat", options=opts))
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ModelPickerModal)
            inp = modal.query_one("#picker-search", Input)
            inp.value = "q"
            await pilot.pause(0.02)
            inp.value = "qw"
            await pilot.pause(0.15)
            ml = modal.query_one("#picker-list", ModelList)
            assert ml.option_count == 1

    async def test_modal_filters_options_by_search(self) -> None:
        from textual.widgets import Button, Input

        from lilbee.cli.tui.screens.model_picker import ModelPickerModal
        from lilbee.cli.tui.widgets.model_list import ModelList

        opts = [
            ModelOption("Qwen3 0.6B", "qwen3-0.6b"),
            ModelOption("Llama 8B", "llama-8b"),
        ]

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield Button("anchor")

        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ModelPickerModal(scope="chat", options=opts))
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ModelPickerModal)
            inp = modal.query_one("#picker-search", Input)
            inp.value = "llama"
            await pilot.pause(0.15)  # let the debounce timer fire
            ml = modal.query_one("#picker-list", ModelList)
            assert ml.option_count == 1

    async def test_modal_escape_dismisses_with_none(self) -> None:
        from textual.widgets import Button

        from lilbee.cli.tui.screens.model_picker import ModelPickerModal

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield Button("anchor")

        results: list[str | None] = []

        def _capture(value: str | None) -> None:
            results.append(value)

        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(
                ModelPickerModal(scope="embed", options=[ModelOption("X", "x")]),
                _capture,
            )
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert results == [None]

    async def test_modal_enter_in_search_picks_first_match(self) -> None:
        from textual.widgets import Button, Input

        from lilbee.cli.tui.screens.model_picker import ModelPickerModal

        opts = [ModelOption("Qwen3 0.6B", "qwen3-0.6b")]

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield Button("anchor")

        app = _App()
        results: list[str | None] = []
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(
                ModelPickerModal(scope="chat", options=opts),
                lambda v: results.append(v),
            )
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ModelPickerModal)
            modal.query_one("#picker-list").highlighted = 0
            inp = modal.query_one("#picker-search", Input)
            await pilot.click(inp)
            await pilot.press("enter")
            await pilot.pause()
            assert results == ["qwen3-0.6b"]

    async def test_modal_enter_with_empty_list_is_noop(self) -> None:
        from textual.widgets import Button, Input

        from lilbee.cli.tui.screens.model_picker import ModelPickerModal

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield Button("anchor")

        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ModelPickerModal(scope="chat", options=[]))
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ModelPickerModal)
            inp = modal.query_one("#picker-search", Input)
            await pilot.click(inp)
            await pilot.press("enter")
            await pilot.pause()
            # Modal stays open with no selection.
            assert isinstance(app.screen, ModelPickerModal)

    async def test_modal_action_focus_search_focuses_input(self) -> None:
        from textual.widgets import Button, Input

        from lilbee.cli.tui.screens.model_picker import ModelPickerModal

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield Button("anchor")

        app = _App()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen(ModelPickerModal(scope="embed", options=[ModelOption("X", "x")]))
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ModelPickerModal)
            modal.query_one("#picker-list").focus()
            await pilot.pause()
            modal.action_focus_search()
            await pilot.pause()
            assert isinstance(app.focused, Input)


class TestModelList:
    async def test_set_rows_with_headings(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.set_rows(
                [
                    ModelListSection(
                        heading="OpenAI", rows=[_make_frontier_row("gpt-4", "OpenAI")]
                    ),
                    ModelListSection(
                        heading="Anthropic",
                        rows=[_make_frontier_row("claude-x", "Anthropic", ready=False)],
                    ),
                ]
            )
            await pilot.pause()
            assert ml.option_count == 4  # 2 headings + 2 rows

    async def test_selecting_row_posts_selected(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

        captured: list[CatalogRow] = []
        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            target = _make_frontier_row("gpt-x")
            ml.set_rows([ModelListSection(heading=None, rows=[target])])

            def on_selected(message: ModelList.Selected) -> None:
                captured.append(message.row)

            app.screen._on_model_list_selected = on_selected  # type: ignore[attr-defined]
            ml.action_select()
            await pilot.pause()
            await pilot.pause()
            assert len(captured) == 0  # message bubbles past _ModelListApp screen
            ml.post_message(ModelList.Selected(target))
            await pilot.pause()

    async def test_local_row_rendering_includes_installed_pill(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.set_rows(
                [
                    ModelListSection(
                        heading="Installed",
                        rows=[_make_local_row("Already Here", installed=True)],
                    )
                ]
            )
            await pilot.pause()
            opt = ml.get_option_at_index(1)
            rendered = str(opt.prompt)
            assert "Already Here" in rendered
            assert "installed" in rendered

    async def test_local_row_renders_featured_star_and_meta_strip(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

        row = LocalCatalogRow(
            name="Featured Model",
            task="chat",
            params="8B",
            size="4 GB",
            quant="Q4_0",
            downloads="--",
            featured=True,
            installed=False,
            sort_downloads=0,
            sort_size=4.0,
            ref="featured/ref",
            backend="native",
        )
        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.set_rows([ModelListSection(heading=None, rows=[row])])
            await pilot.pause()
            rendered = str(ml.get_option_at_index(0).prompt)
            assert "Featured Model" in rendered
            assert "chat" in rendered
            assert "8B" in rendered
            # native is the implicit default; we drop it from the meta strip.
            assert "native" not in rendered

    async def test_selected_event_with_unknown_option_id_is_dropped(self) -> None:
        from textual.widgets import OptionList
        from textual.widgets.option_list import Option

        from lilbee.cli.tui.widgets.model_list import ModelList

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            evt = OptionList.OptionSelected(option_list=ml, option=Option("x", id="ghost"), index=0)
            ml._on_option_selected(evt)
            await pilot.pause()

    async def test_selected_event_with_no_option_id_is_dropped(self) -> None:
        from textual.widgets import OptionList
        from textual.widgets.option_list import Option

        from lilbee.cli.tui.widgets.model_list import ModelList

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            evt = OptionList.OptionSelected(option_list=ml, option=Option("x"), index=0)
            ml._on_option_selected(evt)
            await pilot.pause()

    async def test_set_rows_clears_previous_population(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.set_rows([ModelListSection(heading="A", rows=[_make_frontier_row("first", "X")])])
            ml.set_rows([ModelListSection(heading="B", rows=[_make_frontier_row("second", "Y")])])
            await pilot.pause()
            assert ml.option_count == 2
            opt = ml.get_option_at_index(1)
            assert "second" in str(opt.prompt)

    async def test_empty_sections_yield_no_options(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.set_rows([])
            await pilot.pause()
            assert ml.option_count == 0

    async def test_append_rows_extends_existing_population(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.set_rows([ModelListSection(heading=None, rows=[_make_frontier_row("a", "X")])])
            ml.append_rows([_make_frontier_row("b", "X"), _make_frontier_row("c", "X")])
            await pilot.pause()
            assert ml.row_count == 3
            assert "c" in str(ml.get_option_at_index(2).prompt)

    async def test_append_rows_with_empty_list_is_noop(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.set_rows([ModelListSection(heading=None, rows=[_make_frontier_row("a", "X")])])
            ml.append_rows([])
            await pilot.pause()
            assert ml.row_count == 1

    async def test_highlighted_row_returns_none_when_no_selection(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.highlighted = None
            assert ml.highlighted_row() is None

    async def test_highlighted_row_returns_none_on_index_error(self) -> None:
        from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.set_rows([ModelListSection(heading=None, rows=[_make_frontier_row("a", "X")])])
            ml.highlighted = 0
            with mock.patch.object(ml, "get_option_at_index", side_effect=IndexError):
                assert ml.highlighted_row() is None

    async def test_highlighted_row_returns_none_for_option_without_id(self) -> None:
        from textual.widgets.option_list import Option

        from lilbee.cli.tui.widgets.model_list import ModelList, ModelListSection

        app = _ModelListApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            ml = app.query_one(ModelList)
            ml.set_rows([ModelListSection(heading=None, rows=[_make_frontier_row("a", "X")])])
            ml.highlighted = 0
            with mock.patch.object(ml, "get_option_at_index", return_value=Option("ghost")):
                assert ml.highlighted_row() is None


class TestIsMmproj:
    def test_mmproj_detected(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _is_mmproj

        assert _is_mmproj("llava-mmproj-f16.gguf") is True

    def test_mmproj_case_insensitive(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _is_mmproj

        assert _is_mmproj("model-MMPROJ-q4.gguf") is True

    def test_normal_model_not_mmproj(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _is_mmproj

        assert _is_mmproj("qwen3:8b") is False


@pytest.mark.real_model_classify
class TestClassifyInstalledModels:
    def test_native_models_classified_by_task(self, tmp_path) -> None:
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.modelhub.registry import ModelManifest

        chat_manifest = ModelManifest(
            hf_repo="Qwen/Qwen3-8B-GGUF",
            gguf_filename="Qwen3-8B-Q4_K_M.gguf",
            size_bytes=100,
            task="chat",
            downloaded_at="",
        )
        embed_manifest = ModelManifest(
            hf_repo="nomic-ai/nomic-embed-text-v1.5-GGUF",
            gguf_filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
            size_bytes=100,
            task="embedding",
            downloaded_at="",
        )
        vision_manifest = ModelManifest(
            hf_repo="org/Llava-GGUF",
            gguf_filename="llava-Q4_K_M.gguf",
            size_bytes=100,
            task="vision",
            downloaded_at="",
        )
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()

        with (
            mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.modelhub.model_manager.classify_remote_models",
                return_value=[],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = [
                chat_manifest,
                embed_manifest,
                vision_manifest,
            ]
            chat, embed = _classify_installed_models()

        chat_refs = [ref for _, ref in chat]
        embed_refs = [ref for _, ref in embed]
        assert chat_manifest.ref in chat_refs
        assert embed_manifest.ref in embed_refs

    def test_mmproj_filtered_from_all_sources(self, tmp_path) -> None:
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.modelhub.model_manager import RemoteModel
        from lilbee.modelhub.registry import ModelManifest

        # Manifest whose filename contains "mmproj" must be filtered out;
        # the picker only surfaces the main model files.
        mmproj_manifest = ModelManifest(
            hf_repo="org/Llava-GGUF",
            gguf_filename="llava-mmproj-f16.gguf",
            size_bytes=100,
            task="vision",
            downloaded_at="",
        )
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()

        remote_mmproj = RemoteModel(
            name="mmproj-model:latest",
            task="vision",
            family="clip",
            parameter_size="",
        )
        with (
            mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.modelhub.model_manager.classify_remote_models",
                return_value=[remote_mmproj],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = [mmproj_manifest]
            chat, embed = _classify_installed_models()

        all_refs = [ref for _, ref in chat + embed]
        assert not any("mmproj" in r.lower() for r in all_refs)

    def test_remote_ollama_models_stored_with_prefix(self, tmp_path) -> None:
        """Ollama-backed options carry the ollama/ prefix in their ref.

        Routing uses the prefix as the single source of truth, so the
        origin must survive in config. The human label still shows the
        tag without the prefix so the dropdown stays readable.
        """
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.modelhub.model_manager import RemoteModel

        remote_chat = RemoteModel(
            name="llama3:8b",
            task="chat",
            family="llama",
            parameter_size="8B",
            provider="Ollama",
        )
        remote_embed = RemoteModel(
            name="nomic-embed-text:latest",
            task="embedding",
            family="nomic-bert",
            parameter_size="137M",
            provider="Ollama",
        )
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        cfg.remote_base_url = "http://localhost:11434"

        with (
            mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.modelhub.model_manager.classify_remote_models",
                return_value=[remote_chat, remote_embed],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = []
            chat, embed = _classify_installed_models()

        chat_refs = [ref for _, ref in chat]
        embed_refs = [ref for _, ref in embed]
        chat_labels = [label for label, _ in chat]
        assert "ollama/llama3:8b" in chat_refs
        assert "ollama/nomic-embed-text:latest" in embed_refs
        assert any("llama3:8b" in lbl and "Ollama" in lbl for lbl in chat_labels)

    def test_remote_ollama_coexists_with_native_repo(self, tmp_path) -> None:
        """A native HF manifest and an Ollama listing for the same family coexist.

        Refs are distinct (``<repo>/<file>.gguf`` vs ``ollama/<name>``),
        so both rows survive the dedup pass and appear in the picker.
        """
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.modelhub.model_manager import RemoteModel
        from lilbee.modelhub.registry import ModelManifest

        native = ModelManifest(
            hf_repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            gguf_filename="Mistral-7B-Q4_K_M.gguf",
            size_bytes=100,
            task="chat",
            downloaded_at="",
        )
        remote = RemoteModel(
            name="mistral:latest",
            task="chat",
            family="mistral",
            parameter_size="7B",
            provider="Ollama",
        )
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        cfg.remote_base_url = "http://localhost:11434"

        with (
            mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.modelhub.model_manager.classify_remote_models",
                return_value=[remote],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = [native]
            chat, _ = _classify_installed_models()

        chat_refs = {ref for _, ref in chat}
        assert native.ref in chat_refs
        assert "ollama/mistral:latest" in chat_refs

    def test_remote_blank_name_dropped(self, tmp_path) -> None:
        """Remote entries with an empty name are skipped before reaching the picker."""
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.modelhub.model_manager import RemoteModel

        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        cfg.remote_base_url = "http://localhost:11434"

        blank = RemoteModel(
            name="",
            task="chat",
            family="",
            parameter_size="",
            provider="Ollama",
        )
        good = RemoteModel(
            name="qwen3:8b",
            task="chat",
            family="qwen3",
            parameter_size="8B",
            provider="Ollama",
        )
        with (
            mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.modelhub.model_manager.classify_remote_models",
                return_value=[blank, good],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = []
            chat, _ = _classify_installed_models()

        chat_labels = [label for label, _ in chat]
        chat_refs = [ref for _, ref in chat]
        assert chat_refs == ["ollama/qwen3:8b"]
        assert all(lbl != " (Ollama)" and lbl.strip() != "(Ollama)" for lbl in chat_labels)

    def test_multi_quant_same_repo_disambiguates_label(self, tmp_path) -> None:
        """Two quants from the same repo render with quant suffixes."""
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.modelhub.registry import ModelManifest

        repo = "Qwen/Qwen3-0.6B-GGUF"
        m_q4 = ModelManifest(
            hf_repo=repo,
            gguf_filename="Qwen3-0.6B-Q4_K_M.gguf",
            size_bytes=100,
            task="chat",
            downloaded_at="",
        )
        m_q8 = ModelManifest(
            hf_repo=repo,
            gguf_filename="Qwen3-0.6B-Q8_0.gguf",
            size_bytes=100,
            task="chat",
            downloaded_at="",
        )
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()

        with (
            mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.modelhub.model_manager.classify_remote_models",
                return_value=[],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = [m_q4, m_q8]
            chat, _ = _classify_installed_models()

        labels = sorted(label for label, _ in chat)
        assert labels == ["Qwen3 0.6B (Q4_K_M)", "Qwen3 0.6B (Q8_0)"]
        refs = sorted(ref for _, ref in chat)
        assert refs == [m_q4.ref, m_q8.ref]

    def test_no_models_returns_empty(self, tmp_path) -> None:
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models

        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()

        with (
            mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.modelhub.model_manager.classify_remote_models",
                return_value=[],
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.discover_api_models",
                return_value={},
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = []
            chat, embed = _classify_installed_models()

        assert chat == []
        assert embed == []

    def test_unknown_task_manifest_dropped(self, tmp_path) -> None:
        """Manifests with a task outside the known taxonomy are silently dropped.

        Protects against forward-compat manifests: a future task slug the
        current build doesn't know about must not accidentally land in the
        chat bucket (that would let an unrelated model get picked as a
        chat model via the TUI).
        """
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.modelhub.registry import ModelManifest

        bogus = ModelManifest(
            hf_repo="org/Mystery-GGUF",
            gguf_filename="mystery-Q4_K_M.gguf",
            size_bytes=100,
            task="unknown",  # untyped on purpose: task is a plain str on manifests
            downloaded_at="",
        )
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()

        with (
            mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.modelhub.model_manager.classify_remote_models",
                return_value=[],
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.discover_api_models",
                return_value={},
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = [bogus]
            chat, embed = _classify_installed_models()

        chat_refs = [ref for _, ref in chat]
        embed_refs = [ref for _, ref in embed]
        assert bogus.ref not in chat_refs
        assert bogus.ref not in embed_refs


class TestSlashSuggester:
    async def test_empty_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        assert await s.get_suggestion("") is None

    async def test_slash_prefix_suggests_command(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        r = await s.get_suggestion("/he")
        assert r == "/help"

    async def test_exact_command_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        assert await s.get_suggestion("/help") is None

    async def test_plain_text_with_space_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        # Has space but doesn't start with /: hits _suggest_argument which returns None
        assert await s.get_suggestion("hello world") is None

    async def test_plain_text_no_space_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        # No space, doesn't start with /: hits line 43 return None
        assert await s.get_suggestion("hello") is None

    async def test_no_match_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        assert await s.get_suggestion("/zzzz") is None

    @mock.patch("lilbee.cli.tui.widgets.suggester.SlashSuggester._get_model_names")
    async def test_suggest_model_arg(self, mock_names: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        mock_names.return_value = ["qwen3:8b", "mistral:7b"]
        s = SlashSuggester(use_cache=False)
        r = await s.get_suggestion("/model qw")
        assert r is not None
        assert "qwen3:8b" in r

    async def test_suggest_set_arg(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        r = await s.get_suggestion("/set chat")
        assert r is not None
        assert "chat_model" in r

    @mock.patch("lilbee.cli.tui.widgets.suggester.SlashSuggester._get_document_names")
    async def test_suggest_delete_arg(self, mock_names: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        mock_names.return_value = ["readme.md", "notes.txt"]
        s = SlashSuggester(use_cache=False)
        r = await s.get_suggestion("/delete rea")
        assert r is not None
        assert "readme.md" in r

    async def test_suggest_theme_arg(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        r = await s.get_suggestion("/theme dra")
        assert r is not None
        assert "dracula" in r

    async def test_unknown_command_with_space_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        assert await s.get_suggestion("/foobar xyz") is None

    async def test_suggest_from_list_no_match(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        r = s._suggest_from_list("/model zzz", "zzz", ["alpha", "beta"])
        assert r is None

    async def test_suggest_from_list_exact_match_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        r = s._suggest_from_list("/model alpha", "alpha", ["alpha"])
        assert r is None

    def test_get_model_names_error(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        with mock.patch(
            "lilbee.cli.tui.widgets.suggester.SlashSuggester._get_model_names",
            side_effect=Exception("fail"),
        ):
            # Calling through suggest_argument won't crash
            pass
        # Direct call with mock
        with mock.patch(
            "lilbee.modelhub.models.list_installed_models", side_effect=Exception("err")
        ):
            assert s._get_model_names() == []

    def test_get_document_names_error(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        with mock.patch(
            "lilbee.cli.tui.widgets.suggester.get_services", side_effect=Exception("err")
        ):
            assert s._get_document_names() == []


class TestGetCompletions:
    def test_non_slash_returns_empty(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import get_completions

        assert get_completions("hello") == []

    def test_slash_prefix_returns_commands(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import get_completions

        r = get_completions("/he")
        assert "/help" in r

    def test_exact_command_returns_empty(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import get_completions

        r = get_completions("/help")
        assert r == []

    @mock.patch(
        "lilbee.modelhub.models.list_installed_models", return_value=["qwen3:8b", "mistral:7b"]
    )
    def test_model_arg_completions(self, _mock: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.autocomplete import get_completions

        r = get_completions("/model qw")
        assert "qwen3:8b" in r

    @mock.patch("lilbee.modelhub.models.list_installed_models", return_value=["qwen3:8b"])
    def test_model_arg_no_partial(self, _mock: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.autocomplete import get_completions

        r = get_completions("/model ")
        assert "qwen3:8b" in r

    def test_slash_prefix_includes_aliases(self) -> None:
        """/cat expands via the /catalog alias for /models."""
        from lilbee.cli.tui.widgets.autocomplete import get_completions

        r = get_completions("/cat")
        assert "/catalog" in r

    def test_unknown_command_arg_returns_empty(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import get_completions

        r = get_completions("/foobar something")
        assert r == []

    def test_add_arg_completions(self, tmp_path: object) -> None:
        from pathlib import Path as P

        from lilbee.cli.tui.widgets.autocomplete import get_completions

        d = P(str(tmp_path))
        (d / "testfile.txt").touch()
        r = get_completions(f"/add {d}/")
        assert any("testfile.txt" in x for x in r)


class TestModelOptions:
    def test_returns_models(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _model_options

        with mock.patch("lilbee.modelhub.models.list_installed_models", return_value=["a", "b"]):
            assert _model_options() == ["a", "b"]

    def test_returns_empty_on_error(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _model_options

        with mock.patch(
            "lilbee.modelhub.models.list_installed_models", side_effect=Exception("err")
        ):
            assert _model_options() == []


class TestSettingOptions:
    def test_returns_keys(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _setting_options

        r = _setting_options()
        assert "chat_model" in r


class TestDocumentOptions:
    def test_returns_filenames(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _document_options

        mock_svc = mock.MagicMock()
        mock_svc.store.get_sources.return_value = [{"filename": "a.txt", "source": "a.txt"}]
        with mock.patch("lilbee.cli.tui.widgets.autocomplete.get_services", return_value=mock_svc):
            assert _document_options() == ["a.txt"]

    def test_returns_empty_on_error(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _document_options

        with mock.patch(
            "lilbee.cli.tui.widgets.autocomplete.get_services", side_effect=Exception("err")
        ):
            assert _document_options() == []

    def test_falls_back_to_source_key(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _document_options

        mock_svc = mock.MagicMock()
        mock_svc.store.get_sources.return_value = [{"source": "b.pdf"}]
        with mock.patch("lilbee.cli.tui.widgets.autocomplete.get_services", return_value=mock_svc):
            assert _document_options() == ["b.pdf"]


class TestThemeOptions:
    def test_returns_themes(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _theme_options

        r = _theme_options()
        assert "dracula" in r


class TestPathOptions:
    def test_returns_paths_no_partial(self, tmp_path: object) -> None:
        from pathlib import Path as P

        from lilbee.cli.tui.widgets.autocomplete import _path_options

        d = P(str(tmp_path))
        (d / "file.txt").touch()
        (d / "subdir").mkdir()
        r = _path_options(str(d) + "/")
        assert any("file.txt" in x for x in r)
        assert any("subdir/" in x for x in r)

    def test_returns_empty_on_error(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _path_options

        r = _path_options("/nonexistent_xyzzy_path/abc")
        assert r == []

    def test_excludes_dotfiles(self, tmp_path: object) -> None:
        from pathlib import Path as P

        from lilbee.cli.tui.widgets.autocomplete import _path_options

        d = P(str(tmp_path))
        (d / ".hidden").touch()
        (d / "visible").touch()
        r = _path_options(str(d) + "/")
        assert all(".hidden" not in x for x in r)
        assert any("visible" in x for x in r)

    def test_partial_path_filters(self, tmp_path: object) -> None:
        from pathlib import Path as P

        from lilbee.cli.tui.widgets.autocomplete import _path_options

        d = P(str(tmp_path))
        (d / "abc.txt").touch()
        (d / "xyz.txt").touch()
        r = _path_options(str(d / "ab"))
        assert any("abc.txt" in x for x in r)
        assert all("xyz.txt" not in x for x in r)

    def test_directory_trailing_slash(self, tmp_path: object) -> None:
        from pathlib import Path as P

        from lilbee.cli.tui.widgets.autocomplete import _path_options

        d = P(str(tmp_path))
        (d / "mydir").mkdir()
        r = _path_options(str(d) + "/")
        assert any(x.endswith("/") for x in r)

    def test_nonexistent_parent_returns_empty(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _path_options

        r = _path_options("/nonexistent/path/abc")
        assert r == []

    def test_tilde_expansion(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _path_options

        r = _path_options("~")
        assert isinstance(r, list)

    def test_empty_partial_uses_cwd(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _path_options

        r = _path_options("")
        assert isinstance(r, list)

    def test_exception_returns_empty(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _path_options

        with mock.patch("lilbee.cli.tui.widgets.autocomplete.Path") as MockPath:
            MockPath.side_effect = RuntimeError("boom")
            r = _path_options("something")
        assert r == []

    def test_limits_results_to_20(self, tmp_path):
        from lilbee.cli.tui.widgets.autocomplete import _path_options

        for i in range(25):
            (tmp_path / f"file_{i:02d}.txt").touch()
        r = _path_options(str(tmp_path) + "/")
        assert len(r) == 20


class _OverlayApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        yield CompletionOverlay()


class TestCompletionOverlay:
    async def test_show_completions_populates(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            overlay.show_completions(["/help", "/model", "/set"])
            await pilot.pause()
            assert overlay.is_visible
            assert overlay.get_current() == "/help"

    async def test_show_empty_hides(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            overlay.show_completions([])
            assert not overlay.is_visible

    async def test_cycle_next(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            overlay.show_completions(["/help", "/model", "/set"])
            r = overlay.cycle_next()
            assert r == "/model"
            r = overlay.cycle_next()
            assert r == "/set"
            r = overlay.cycle_next()
            assert r == "/help"  # wraps

    async def test_cycle_next_empty(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            assert overlay.cycle_next() is None

    async def test_get_current_empty(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            assert overlay.get_current() is None

    async def test_hide(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            overlay.show_completions(["/help"])
            overlay.hide()
            assert not overlay.is_visible

    async def test_action_dismiss(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            overlay.show_completions(["/help"])
            overlay.action_dismiss_overlay()
            assert not overlay.is_visible

    async def test_max_visible_truncates(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _MAX_VISIBLE, CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            many = [f"/opt{i}" for i in range(20)]
            overlay.show_completions(many)
            assert len(overlay._options) == _MAX_VISIBLE


class TestTaskQueue:
    def test_enqueue_and_advance(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        q.enqueue(lambda: None, "Sync", "sync")
        assert q.is_empty is False
        task = q.advance()
        assert task is not None
        assert task.status == TaskStatus.ACTIVE
        assert q.active_task is task

    def test_complete_clears_active(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "Sync", "sync")
        q.advance()
        q.complete_task(tid)
        q.remove_task(tid)
        assert q.is_empty

    def test_cancel_queued(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        q.enqueue(lambda: None, "A", "download")
        q.advance()
        queued_id = q.enqueue(lambda: None, "B", "sync")
        assert q.cancel(queued_id) is True
        assert len(q.queued_tasks) == 0

    def test_cancel_active(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "A", "download")
        q.advance()
        assert q.cancel(tid) is True
        assert q.active_task is None

    def test_advance_returns_none_when_same_type_active(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        q.enqueue(lambda: None, "A", "download")
        q.advance("download")
        q.enqueue(lambda: None, "B", "download")
        assert q.advance("download") is None  # same type already active

    def test_advance_different_types_concurrent(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        q.enqueue(lambda: None, "A", "download")
        q.advance("download")
        q.enqueue(lambda: None, "B", "sync")
        task = q.advance("sync")
        assert task is not None  # different type can advance
        assert task.name == "B"
        assert len(q.active_tasks) == 2

    def test_get_task_returns_task(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "A", "download")
        task = q.get_task(tid)
        assert task is not None
        assert task.name == "A"

    def test_get_task_returns_none_for_unknown(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        assert q.get_task("nonexistent") is None

    def test_fail_task(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "A", "download")
        q.advance()
        q.fail_task(tid, "oops")
        task = q.get_task(tid)
        assert task is not None
        assert task.status == TaskStatus.FAILED

    def test_on_change_callback(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        calls: list[bool] = []
        q = TaskQueue(on_change=lambda: calls.append(True))
        q.enqueue(lambda: None, "A", "sync")
        assert len(calls) >= 1

    def test_complete_task_adds_to_history(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "Sync", "sync")
        q.advance()
        q.complete_task(tid)
        assert len(q.history) == 1
        assert q.history[0].status == TaskStatus.DONE

    def test_fail_task_adds_to_history(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "Sync", "sync")
        q.advance()
        q.fail_task(tid, "oops")
        assert len(q.history) == 1
        assert q.history[0].status == TaskStatus.FAILED

    def test_history_accumulates(self) -> None:
        """Completed + failed tasks sit in history until remove_task prunes them."""
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        t1 = q.enqueue(lambda: None, "A", "sync")
        q.advance()
        q.complete_task(t1)
        t2 = q.enqueue(lambda: None, "B", "sync")
        q.advance()
        q.fail_task(t2, "err")
        # Both completions sit in history together; remove_task would prune.
        assert len(q.history) == 2

    def test_remove_task_prunes_history(self) -> None:
        """remove_task drops the entry from history so TaskCenter rows unmount."""
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        t1 = q.enqueue(lambda: None, "A", "sync")
        q.advance()
        q.complete_task(t1)
        assert any(t.task_id == t1 for t in q.history)
        q.remove_task(t1)
        assert not any(t.task_id == t1 for t in q.history)

    def test_clear_history_drops_all_finished_tasks(self) -> None:
        """clear_history prunes DONE/FAILED/CANCELLED in one shot."""
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        a = q.enqueue(lambda: None, "A", "sync")
        q.advance()
        q.complete_task(a)
        b = q.enqueue(lambda: None, "B", "sync")
        q.advance()
        q.fail_task(b, "err")
        c = q.enqueue(lambda: None, "C", "sync")
        q.advance()
        # C stays ACTIVE; cleared list should still leave it behind.
        assert len(q.history) == 2
        cleared = q.clear_history()
        assert cleared == 2
        assert q.history == []
        active_task = q.get_task(c)
        assert active_task is not None and active_task.status == TaskStatus.ACTIVE

    def test_history_empty_initially(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        assert q.history == []

    def test_cancel_nonexistent_returns_false(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        assert q.cancel("nonexistent") is False

    def test_cancel_done_task_is_noop(self) -> None:
        """terminal rows are immutable. Cancel on DONE returns False
        and leaves status + completed_at frozen."""
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "A", "download")
        q.advance()
        q.complete_task(tid)
        task = q.get_task(tid)
        assert task is not None
        frozen_completed_at = task.completed_at
        assert q.cancel(tid) is False
        task_after = q.get_task(tid)
        assert task_after is not None
        assert task_after.status == TaskStatus.DONE
        assert task_after.completed_at == frozen_completed_at

    def test_cancel_failed_task_is_noop(self) -> None:
        """cancel on a FAILED row must not flip it to CANCELLED."""
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "A", "download")
        q.advance()
        q.fail_task(tid, "boom")
        assert q.cancel(tid) is False
        task_after = q.get_task(tid)
        assert task_after is not None
        assert task_after.status == TaskStatus.FAILED

    def test_cancel_already_cancelled_is_noop(self) -> None:
        """cancelling an already-cancelled row does not re-append
        it to history or reset its completed_at."""
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "A", "download")
        q.advance()
        assert q.cancel(tid) is True
        task = q.get_task(tid)
        assert task is not None
        first_completed_at = task.completed_at
        history_len = len(q.history)
        assert q.cancel(tid) is False
        assert len(q.history) == history_len
        task_after = q.get_task(tid)
        assert task_after is not None
        assert task_after.status == TaskStatus.CANCELLED
        assert task_after.completed_at == first_completed_at

    def test_remove_task_nonexistent_is_noop(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        q.remove_task("nonexistent")
        assert q.is_empty

    def test_update_task_nonexistent_is_noop(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        q.update_task("nonexistent", 50, "detail")
        assert q.is_empty

    def test_advance_empty_returns_none(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        assert q.advance() is None

    def test_remove_active_task_clears_active_id(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "A", "sync")
        q.advance()
        assert q.active_task is not None
        q.remove_task(tid)
        assert q.active_task is None

    def test_active_tasks_returns_all(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        q.enqueue(lambda: None, "DL", "download")
        q.enqueue(lambda: None, "Sync", "sync")
        q.advance("download")
        q.advance("sync")
        assert len(q.active_tasks) == 2

    def test_active_tasks_empty_initially(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        assert q.active_tasks == []

    def test_is_empty_with_multiple_types(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        t1 = q.enqueue(lambda: None, "DL", "download")
        q.advance("download")
        assert not q.is_empty
        q.complete_task(t1)
        q.remove_task(t1)
        assert q.is_empty

    def test_advance_with_task_type_arg(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        q.enqueue(lambda: None, "DL", "download")
        q.enqueue(lambda: None, "Sync", "sync")
        task = q.advance("sync")
        assert task is not None
        assert task.name == "Sync"
        assert task.status == TaskStatus.ACTIVE
        # download not yet advanced
        assert len(q.active_tasks) == 1

    def test_complete_frees_type_slot(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        t1 = q.enqueue(lambda: None, "DL-A", "download")
        q.enqueue(lambda: None, "DL-B", "download")
        q.advance("download")
        q.complete_task(t1)
        q.remove_task(t1)
        t2 = q.advance("download")
        assert t2 is not None
        assert t2.name == "DL-B"

    def test_queued_tasks_across_types(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        q.enqueue(lambda: None, "DL", "download")
        q.enqueue(lambda: None, "Sync", "sync")
        assert len(q.queued_tasks) == 2
        q.advance("download")
        # DL is now active, Sync still queued
        assert len(q.queued_tasks) == 1

    def test_cancel_concurrent_task(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        t1 = q.enqueue(lambda: None, "DL", "download")
        t2 = q.enqueue(lambda: None, "Sync", "sync")
        q.advance("download")
        q.advance("sync")
        assert len(q.active_tasks) == 2
        q.cancel(t1)
        assert len(q.active_tasks) == 1
        assert q.active_tasks[0].task_id == t2


class TestCompletionOverlayCyclePrev:
    async def test_cycle_prev_wraps(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            overlay.show_completions(["a", "b", "c"])
            result = overlay.cycle_prev()
            assert result == "c"  # wraps from 0 to 2

    async def test_cycle_prev_returns_none_when_empty(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        app = _OverlayApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            overlay = app.query_one(CompletionOverlay)
            assert overlay.cycle_prev() is None


class _SetupApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("bg")


class TestSetupWizard:
    def test_creates(self) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        wizard = SetupWizard()
        assert wizard._selected_chat is None
        assert wizard._selected_embed is None

    async def test_compose_mounts(self) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = _SetupApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupWizard())
            await pilot.pause()
            assert len(app.screen_stack) == 2

    async def test_action_cancel_dismisses_skipped_when_no_selection(self) -> None:
        """action_cancel returns 'skipped' only when the user picked nothing."""
        from lilbee.catalog.types import ModelTask
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = _SetupApp()
        results: list[object] = []
        async with app.run_test() as pilot:
            app.push_screen(SetupWizard(), callback=lambda r: results.append(r))
            await pilot.pause()
            # Clear the RAM-based preselection so action_cancel treats it as empty.
            app.screen._selections[ModelTask.CHAT] = (None, None)
            app.screen._selections[ModelTask.EMBEDDING] = (None, None)
            app.screen.action_cancel()
            await pilot.pause()
        assert "skipped" in results

    async def test_action_cancel_dismisses_completed_when_any_selection(self) -> None:
        """action_cancel returns 'completed' if any model was picked."""
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = _SetupApp()
        results: list[object] = []
        async with app.run_test() as pilot:
            app.push_screen(SetupWizard(), callback=lambda r: results.append(r))
            await pilot.pause()
            # Preselected chat+embed survive; action_cancel should return completed.
            with mock.patch("lilbee.app.services.reset_services"):
                app.screen.action_cancel()
            await pilot.pause()
        assert "completed" in results

    def test_scan_installed_models_empty_dir(self, tmp_path) -> None:
        from lilbee.cli.tui.screens.setup import _scan_installed_models

        cfg.models_dir = tmp_path / "nonexistent"
        chat, embed = _scan_installed_models()
        assert chat == []
        assert embed == []

    def test_scan_installed_models_uses_registry(self, tmp_path) -> None:
        from lilbee.cli.tui.screens.setup import _scan_installed_models

        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        with mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry:
            MockRegistry.return_value.list_installed.return_value = []
            chat, embed = _scan_installed_models()
        assert chat == []
        assert embed == []

    def test_installed_name_to_row(self) -> None:
        from lilbee.cli.tui.screens.setup import _installed_name_to_row

        row = _installed_name_to_row("test-model:latest", "chat")
        assert row.name == "test-model:latest"
        assert row.task == "chat"
        assert row.installed is True
        assert row.featured is False
        assert row.backend == ""

    def test_installed_name_to_row_cleans_native_gguf_ref(self) -> None:
        """Native GGUF refs should render as a clean label, not the raw filename."""
        from lilbee.cli.tui.screens.setup import _installed_name_to_row

        ref = "unsloth/embeddinggemma-300M-qat-GGUF/embeddinggemma-300M-qat-Q8_0.gguf"
        row = _installed_name_to_row(ref, "embedding")
        assert row.name == "embeddinggemma 300M"
        assert row.quant == "Q8_0"
        assert row.ref == ref

    def test_model_card_from_table_row(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import catalog_to_row
        from lilbee.cli.tui.widgets.model_card import ModelCard

        model = _make_model("Test 8B", task="chat", featured=True)
        row = catalog_to_row(model, installed=False)
        card = ModelCard(row)
        assert card.row is row
        assert card.row.featured is True
        assert card.row.task == "chat"
        assert card.row.backend == "native"

    def test_pick_recommended_picks_largest_fitting_not_first(self) -> None:
        """Default pick is the biggest-size-gb model whose min_ram_gb fits."""
        from lilbee.cli.tui.screens import setup as setup_mod

        tiny = _make_model("Tiny", size_gb=0.4, min_ram_gb=0.5, featured=True)
        small = _make_model("Small", size_gb=2.5, min_ram_gb=4, featured=True)
        medium = _make_model("Medium", size_gb=5.0, min_ram_gb=8, featured=True)
        large = _make_model("Large", size_gb=18.0, min_ram_gb=16, featured=True)
        embed = _make_model("Embed", task="embedding", size_gb=0.3, min_ram_gb=1)
        with (
            mock.patch.object(setup_mod, "FEATURED_CHAT", (tiny, small, medium, large)),
            mock.patch.object(setup_mod, "FEATURED_EMBEDDING", (embed,)),
        ):
            # 64 GB: everything fits, largest wins.
            chat, picked_embed = setup_mod._pick_recommended(64.0)
            assert chat is large
            assert picked_embed is embed
            # 16 GB: large just fits, still wins.
            assert setup_mod._pick_recommended(16.0)[0] is large
            # 8 GB: medium is the largest that fits.
            assert setup_mod._pick_recommended(8.0)[0] is medium
            # 4 GB: small is the largest that fits.
            assert setup_mod._pick_recommended(4.0)[0] is small
            # 1 GB: only tiny fits.
            assert setup_mod._pick_recommended(1.0)[0] is tiny

    def test_pick_recommended_falls_back_when_nothing_fits(self) -> None:
        """If no featured model fits, fall back to the first entry."""
        from lilbee.cli.tui.screens import setup as setup_mod

        big = _make_model("BigOnly", size_gb=40.0, min_ram_gb=64, featured=True)
        embed = _make_model("Embed", task="embedding", size_gb=0.3, min_ram_gb=1)
        with (
            mock.patch.object(setup_mod, "FEATURED_CHAT", (big,)),
            mock.patch.object(setup_mod, "FEATURED_EMBEDDING", (embed,)),
        ):
            assert setup_mod._pick_recommended(4.0)[0] is big

    def test_build_section_marks_installed_catalog_cards(self) -> None:
        """Catalog cards whose hf_repo is already installed come back with
        ``installed=True`` so the Enter-to-install hint stays hidden."""
        from lilbee.cli.tui.screens.setup import SetupWizard

        a = _make_model(
            "Qwen3 0.6B",
            featured=True,
            size_gb=0.6,
            hf_repo="Qwen/Qwen3-0.6B-GGUF",
        )
        b = _make_model(
            "Qwen3 4B",
            featured=True,
            size_gb=2.5,
            hf_repo="Qwen/Qwen3-4B-GGUF",
        )
        wizard = SetupWizard.__new__(SetupWizard)
        widgets: list = []
        cards = SetupWizard._build_section(
            wizard, "Chat", (a, b), {"Qwen/Qwen3-0.6B-GGUF"}, widgets
        )
        assert cards[0].row.installed is True
        assert cards[1].row.installed is False

    def test_scan_installed_feeds_build_grid_installed_refs(self, tmp_path) -> None:
        """_scan_installed_models output must be usable as installed refs for the
        catalog grid so the same model never appears with a phantom download."""
        from lilbee.catalog.types import ModelTask
        from lilbee.cli.tui.screens.setup import _scan_installed_models

        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        chat_ref = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
        embed_ref = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
        fake_chat = mock.Mock(ref=chat_ref, task=ModelTask.CHAT)
        fake_embed = mock.Mock(ref=embed_ref, task=ModelTask.EMBEDDING)
        with mock.patch("lilbee.modelhub.registry.ModelRegistry") as MockRegistry:
            MockRegistry.return_value.list_installed.return_value = [fake_chat, fake_embed]
            chat, embed = _scan_installed_models()
        assert chat_ref in chat
        assert embed_ref in embed


class TestAllTasksFetched:
    def test_all_tasks_constant(self) -> None:
        from lilbee.cli.tui.screens.catalog import _ALL_TASKS

        assert "chat" in _ALL_TASKS
        assert "embedding" in _ALL_TASKS
        assert "vision" in _ALL_TASKS


class TestMatchesSearchWidget:
    def test_matches_name(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import catalog_to_row, matches_search

        m = _make_model("Qwen3 8B", task="chat")
        row = catalog_to_row(m, installed=False)
        assert matches_search(row, "qwen") is True

    def test_no_match(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import catalog_to_row, matches_search

        m = _make_model("Qwen3 8B", task="chat")
        row = catalog_to_row(m, installed=False)
        assert matches_search(row, "mistral") is False


class TestLoginCommandRegistered:
    def test_login_in_registry(self) -> None:
        from lilbee.cli.tui.command_registry import COMMANDS, build_dispatch_dict

        names = [c.name for c in COMMANDS]
        assert "/login" in names
        dispatch = build_dispatch_dict()
        assert dispatch["/login"] == "_cmd_login"


class TestRunTuiKeyboardInterrupt:
    def test_keyboard_interrupt_does_not_raise(self) -> None:
        with mock.patch("lilbee.cli.tui.app.LilbeeApp") as MockApp:
            MockApp.return_value.run.side_effect = KeyboardInterrupt
            with (
                mock.patch("lilbee.cli.sync.shutdown_executor"),
                mock.patch("lilbee.cli.tui.reset_services"),
            ):
                from lilbee.cli.tui import run_tui

                run_tui()
            MockApp.return_value.run.assert_called_once()

    def test_cleanup_called_on_interrupt(self) -> None:
        with mock.patch("lilbee.cli.tui.app.LilbeeApp") as MockApp:
            MockApp.return_value.run.side_effect = KeyboardInterrupt
            with (
                mock.patch("lilbee.cli.sync.shutdown_executor") as mock_shutdown,
                mock.patch("lilbee.cli.tui.reset_services") as mock_reset,
            ):
                from lilbee.cli.tui import run_tui

                run_tui()
                mock_shutdown.assert_called_once()
                mock_reset.assert_called_once()


class _ViewTabsApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        yield ViewTabs()


class TestViewTabsWikiVisibility:
    """ViewTabs hides Wiki tab live when cfg.wiki toggles."""

    async def test_wiki_visibility_follows_cfg_signal(self) -> None:
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.status_bar import ViewTab

        cfg.wiki = True
        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            wiki_tab = app.screen.query_one("#view-tab-wiki", ViewTab)
            assert wiki_tab.display is True

            cfg.wiki = False
            app.settings_changed_signal.publish(("wiki", False))
            for _ in range(3):
                await pilot.pause()
            assert wiki_tab.display is False

            cfg.wiki = True
            app.settings_changed_signal.publish(("wiki", True))
            for _ in range(3):
                await pilot.pause()
            assert wiki_tab.display is True

    async def test_wiki_tab_hidden_at_mount_when_disabled(self) -> None:
        """When cfg.wiki=False at startup, the Wiki tab must be hidden by the
        time the first paint settles -- not just after the user toggles the
        setting at runtime.
        """
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.status_bar import ViewTab

        cfg.wiki = False
        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            wiki_tab = app.screen.query_one("#view-tab-wiki", ViewTab)
            wiki_sep = app.screen.query_one("#view-tab-sep-wiki")
            assert wiki_tab.display is False
            assert wiki_sep.display is False

    async def test_apply_wiki_visibility_noop_when_unmounted(self) -> None:
        """The settings signal can fire after the widget unmounts (its
        subscription persists). The apply helper must short-circuit on the
        unmounted widget instead of crashing inside ``query()``.
        """
        from unittest.mock import PropertyMock, patch

        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        async with _ViewTabsApp().run_test() as pilot:
            await pilot.pause()
            bar = pilot.app.query_one(ViewTabs)
            with (
                patch.object(
                    type(bar),
                    "is_mounted",
                    new_callable=PropertyMock,
                    return_value=False,
                ),
                patch.object(bar, "query") as query,
            ):
                bar._apply_wiki_visibility()
                assert not query.called


class TestViewTabs:
    async def test_compose_yields_static(self) -> None:
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = _ViewTabsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ViewTabs)
            assert bar is not None

    async def test_view_tab_on_click_invokes_switch_view(self) -> None:
        """Clicking a ViewTab dispatches to the host's switch_view."""
        from unittest.mock import patch

        from lilbee.cli.tui.widgets.status_bar import ViewTab

        app = _ViewTabsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            tab = next(iter(app.query(ViewTab)))
            with patch.object(app, "switch_view") as switch:
                tab.on_click()
                switch.assert_called_once_with(tab.view_name)

    async def test_view_tab_action_activate_invokes_switch_view(self) -> None:
        """The keyboard binding (Enter / Space) lands on the same dispatch."""
        from unittest.mock import patch

        from lilbee.cli.tui.widgets.status_bar import ViewTab

        app = _ViewTabsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            tab = next(iter(app.query(ViewTab)))
            with patch.object(app, "switch_view") as switch:
                tab.action_activate()
                switch.assert_called_once_with(tab.view_name)

    async def test_default_active_view_is_chat(self) -> None:
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = _ViewTabsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ViewTabs)
            assert bar.active_view == "Chat"

    async def test_watch_active_view_updates_display(self) -> None:
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = _ViewTabsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ViewTabs)
            bar.active_view = "Catalog"
            await pilot.pause()
            assert bar.active_view == "Catalog"

    async def test_set_active_view_to_status(self) -> None:
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = _ViewTabsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ViewTabs)
            bar.active_view = "Status"
            await pilot.pause()
            assert bar.active_view == "Status"

    async def test_mode_text_updates(self) -> None:
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = _ViewTabsApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ViewTabs)
            bar.mode_text = msg.MODE_NORMAL
            await pilot.pause()
            assert bar.mode_text == msg.MODE_NORMAL

    async def test_bottom_bars_container_docks_bottom(self) -> None:
        """BottomBars owns the dock; ViewTabs/TaskBar must not dock themselves.

        Sibling dock-bottom widgets overlap at the same edge row in Textual
        (see BottomBars docstring). Keep the dock on the single container.
        """
        from lilbee.cli.tui.widgets.bottom_bars import BottomBars
        from lilbee.cli.tui.widgets.status_bar import ViewTabs
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        assert "dock: bottom" in BottomBars.DEFAULT_CSS
        assert "dock: bottom" not in ViewTabs.DEFAULT_CSS
        assert "dock: bottom" not in TaskBar.DEFAULT_CSS

    async def test_clicking_tab_calls_switch_view(self) -> None:
        """Clicking a ViewTab routes to app.switch_view with the tab's name.

        Drives a real LilbeeApp because ViewTab now type-checks against
        ``LilbeeApp`` to avoid the ``getattr(self.app, "switch_view", None)``
        smell flagged in AGENTS.md.
        """
        from unittest.mock import patch

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.status_bar import ViewTab

        app = LilbeeApp()
        switch_calls: list[str] = []
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = list(app.screen.query(ViewTab))
            assert len(tabs) >= 2
            target = tabs[1]

            def _record(_self, name: str) -> None:
                switch_calls.append(name)

            with patch.object(LilbeeApp, "switch_view", _record):
                target.on_click()
                await pilot.pause()
            assert switch_calls == [target.view_name]

    async def test_nav_views_contains_all_screens(self) -> None:
        from lilbee.cli.tui.messages import get_nav_views

        views = get_nav_views()
        for name in ("Chat", "Catalog", "Status", "Settings", "Tasks"):
            assert name in views

    async def test_default_view_is_first(self) -> None:
        from lilbee.cli.tui import messages as msg

        assert msg.get_nav_views()[0] == msg.DEFAULT_VIEW


class LilbeeAppHostSettingWriter:
    """LilbeeApp.set_setting + apply_setting cover the non-model write boundary."""

    async def test_set_setting_writes_cfg_settings_and_publishes(self) -> None:
        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            with (
                mock.patch("lilbee.core.settings.set_value") as mock_set,
                mock.patch.object(app.settings_changed_signal, "publish") as mock_publish,
            ):
                app.set_setting("chat_mode", "chat")
            assert cfg.chat_mode == "chat"
            assert mock_set.called
            mock_publish.assert_called_once_with(("chat_mode", "chat"))
            cfg.chat_mode = "search"

    async def test_apply_setting_routes_through_lilbee_app(self) -> None:
        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        from lilbee.cli.tui.app import LilbeeApp, apply_setting

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            with mock.patch.object(app, "set_setting") as mock_set_setting:
                apply_setting(app, "chat_mode", "chat")
                mock_set_setting.assert_called_once_with("chat_mode", "chat")


class LilbeeAppHostViewTabs:
    async def test_screen_composes_status_bar(self) -> None:
        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            while not isinstance(app.screen, ChatScreen):
                app.pop_screen()
                await pilot.pause()
            bar = app.screen.query_one(ViewTabs)
            assert bar is not None

    async def test_status_bar_default_is_chat(self) -> None:
        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            while not isinstance(app.screen, ChatScreen):
                app.pop_screen()
                await pilot.pause()
            bar = app.screen.query_one(ViewTabs)
            assert bar.active_view == "Chat"

    async def test_view_tabs_marks_exactly_one_active(self) -> None:
        """One and only one ViewTab carries the ``-active`` class."""
        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.status_bar import ViewTab

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            while not isinstance(app.screen, ChatScreen):
                app.pop_screen()
                await pilot.pause()
            tabs = list(app.screen.query(ViewTab))
            assert tabs, "ViewTabs should mount per-view ViewTab labels"
            active = [t for t in tabs if t.has_class("-active")]
            assert len(active) == 1, (
                f"exactly one tab should carry the active marker, got {len(active)}"
            )
            assert active[0].view_name == app.active_view

    async def test_view_tabs_dot_separators(self) -> None:
        """Inactive tabs should be separated by dot characters."""
        cfg.chat_model = TEST_LOCAL_REF
        cfg.embedding_model = TEST_EMBED_REF
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.status_bar import ViewTab

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            while not isinstance(app.screen, ChatScreen):
                app.pop_screen()
                await pilot.pause()
            seps = list(app.screen.query(".view-tab-sep"))
            tabs = list(app.screen.query(ViewTab))
            assert tabs, "ViewTabs should mount per-view ViewTab labels"
            assert len(seps) == len(tabs) - 1, (
                f"separator count must be tabs-1; got {len(seps)} for {len(tabs)} tabs"
            )


class TestPill:
    def test_pill_from_string(self) -> None:
        from lilbee.cli.tui.pill import pill

        result = pill("chat", "$primary", "$text")
        text = str(result)
        assert "chat" in text
        assert text == " chat "  # a space of padding on each side, no half-block ends

    def test_pill_from_content(self) -> None:
        from textual.content import Content

        from lilbee.cli.tui.pill import pill

        content_input = Content("embed")
        result = pill(content_input, "$secondary", "$text")
        assert "embed" in str(result)

    def test_pill_empty_string(self) -> None:
        from textual.content import Content

        from lilbee.cli.tui.pill import pill

        result = pill("", "$primary", "$text")
        assert isinstance(result, Content)
        assert str(result) == "  "  # just the padding

    def test_pill_returns_content(self) -> None:
        from textual.content import Content

        from lilbee.cli.tui.pill import pill

        result = pill("ok", "$success", "$text")
        assert isinstance(result, Content)


# ---------------------------------------------------------------------------
# GridSelect widget tests
# ---------------------------------------------------------------------------


class _GridApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        yield GridSelect(
            Static("A", id="item-a"),
            Static("B", id="item-b"),
            Static("C", id="item-c"),
            Static("D", id="item-d"),
            min_column_width=20,
        )


class _LargeGridApp(LilbeeAppHost):
    """Grid with enough items and wide min_column_width to guarantee multiple rows."""

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        items = [Static(f"Item {i}", id=f"item-{i}") for i in range(8)]
        yield GridSelect(*items, min_column_width=30)


class _EmptyGridApp(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        yield GridSelect(min_column_width=20)


class TestGridSelect:
    async def test_selected_control_property(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            child = grid.children[0]
            msg = GridSelect.Selected(grid, child)
            assert msg.control is grid

    async def test_highlighted_control_property(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            child = grid.children[0]
            msg = GridSelect.Highlighted(grid, child)
            assert msg.control is grid

    def test_grid_size_returns_none_when_no_grid_layout(self) -> None:
        """grid_size returns None when layout is not a GridLayout (e.g. before mount)."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        grid = GridSelect(min_column_width=20)
        # Before mount, layout is VerticalLayout, not GridLayout
        assert grid.grid_size is None

    async def test_reveal_highlight_out_of_bounds(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            # Force highlighted to an out-of-bounds index without validation
            grid._reactive_highlighted = 999
            grid.reveal_highlight()  # should not raise
            assert grid._reactive_highlighted == 999

    async def test_watch_highlighted_index_error(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            # Manually call watch with an out-of-bounds index
            grid.watch_highlighted(None, 999)  # should not raise
            assert len(grid.children) > 0

    async def test_validate_highlighted_none(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            assert grid.validate_highlighted(None) is None

    async def test_validate_highlighted_empty_children(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _EmptyGridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            assert grid.validate_highlighted(0) is None

    async def test_validate_highlighted_negative(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            assert grid.validate_highlighted(-1) == 0

    async def test_validate_highlighted_overflow(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            assert grid.validate_highlighted(100) == len(grid.children) - 1

    def test_action_cursor_up_leave_when_no_grid(self) -> None:
        """When grid_size is None, cursor_up posts LeaveUp."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        grid = GridSelect(min_column_width=20)
        # Before mount, grid_size is None (VerticalLayout)
        assert grid.grid_size is None
        messages: list[object] = []
        grid.post_message = lambda m: messages.append(m)  # type: ignore[assignment]
        grid.action_cursor_up()
        assert any(isinstance(m, GridSelect.LeaveUp) for m in messages)

    def test_action_cursor_down_leave_when_no_grid(self) -> None:
        """When grid_size is None, cursor_down posts LeaveDown."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        grid = GridSelect(min_column_width=20)
        assert grid.grid_size is None
        messages: list[object] = []
        grid.post_message = lambda m: messages.append(m)  # type: ignore[assignment]
        grid.action_cursor_down()
        assert any(isinstance(m, GridSelect.LeaveDown) for m in messages)

    async def test_action_cursor_up_when_highlighted_none(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = None
            grid.action_cursor_up()
            assert grid.highlighted == 0

    async def test_action_cursor_down_when_highlighted_none(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = None
            grid.action_cursor_down()
            assert grid.highlighted == 0

    async def test_action_cursor_left_when_highlighted_none(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = None
            grid.action_cursor_left()
            assert grid.highlighted == 0

    async def test_action_cursor_right_when_highlighted_none(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = None
            grid.action_cursor_right()
            assert grid.highlighted == 0

    async def test_action_cursor_left_decrements(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = 2
            grid.action_cursor_left()
            assert grid.highlighted == 1

    async def test_action_cursor_right_increments(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = 1
            grid.action_cursor_right()
            assert grid.highlighted == 2

    async def test_action_cursor_up_boundary_posts_leave_up(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = 0
            # At row 0, cursor_up should try to leave
            # The grid_size width determines what "row 0" means
            gs = grid.grid_size
            assert gs is not None
            # highlighted < width means top row, triggers LeaveUp
            assert grid.highlighted < gs[0]
            # Just verify it doesn't crash: LeaveUp is posted
            grid.action_cursor_up()
            await pilot.pause()

    async def test_action_cursor_down_boundary_posts_leave_down(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = len(grid.children) - 1
            gs = grid.grid_size
            assert gs is not None
            # highlighted + width >= len(children) means bottom, triggers LeaveDown
            assert grid.highlighted + gs[0] >= len(grid.children)
            grid.action_cursor_down()
            await pilot.pause()

    async def test_on_click_highlights_child(self) -> None:
        from textual import events

        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            # Simulate clicking on the second child
            child = grid.children[1]
            click_event = events.Click(
                widget=child,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=0,
                screen_y=0,
            )
            grid.on_click(click_event)
            await pilot.pause()
            assert grid.highlighted == 1

    async def test_on_click_double_click_selects(self) -> None:
        from textual import events

        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        selected: list[object] = []
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = 0
            child = grid.children[0]

            # Click on already-highlighted child triggers select
            click_event = events.Click(
                widget=child,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=0,
                screen_y=0,
            )
            grid.action_select = lambda: selected.append(True)  # type: ignore[assignment]
            grid.on_click(click_event)
            assert len(selected) == 1

    async def test_on_click_no_widget(self) -> None:
        from textual import events

        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            click_event = events.Click(
                widget=None,  # type: ignore[arg-type]
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=0,
                screen_y=0,
            )
            old_highlighted = grid.highlighted
            grid.on_click(click_event)  # should not raise
            assert grid.highlighted == old_highlighted

    async def test_action_select_when_highlighted_none(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = None
            grid.action_select()  # should not raise
            assert grid.highlighted is None

    async def test_action_select_index_error(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid._reactive_highlighted = 999
            grid.action_select()  # should not raise
            assert grid._reactive_highlighted == 999


class TestModelCardSelected:
    def test_model_card_selected_reactive(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import catalog_to_row
        from lilbee.cli.tui.widgets.model_card import ModelCard

        model = _make_model("Test 8B", task="chat", featured=True)
        row = catalog_to_row(model, installed=False)
        card = ModelCard(row)
        assert card.selected is False
        card.selected = True
        assert card.selected is True

    def _make_row(self, **overrides: Any) -> LocalCatalogRow:
        defaults: dict[str, Any] = {
            "name": "test",
            "task": "chat",
            "params": "8B",
            "size": "4 GB",
            "quant": "Q4_K_M",
            "downloads": "--",
            "featured": False,
            "installed": False,
            "sort_downloads": 0,
            "sort_size": 4.0,
        }
        defaults.update(overrides)
        return LocalCatalogRow(**defaults)

    def test_build_status_with_downloads(self) -> None:
        from lilbee.cli.tui.widgets.model_card import _build_local_status as _build_status

        row = self._make_row(downloads="1K", sort_downloads=1000)
        assert _build_status(row) is not None

    def test_build_status_installed(self) -> None:
        from lilbee.cli.tui.widgets.model_card import _build_local_status as _build_status

        result = _build_status(self._make_row(installed=True))
        assert result is not None
        assert "installed" in str(result).lower()

    def test_build_status_downloads_positive(self) -> None:
        from lilbee.cli.tui.widgets.model_card import _build_local_status as _build_status

        row = self._make_row(downloads="5K", sort_downloads=5000)
        assert _build_status(row) is not None

    def test_build_status_none(self) -> None:
        from lilbee.cli.tui.widgets.model_card import _build_local_status as _build_status

        assert _build_status(self._make_row()) is None


class TestCollectNativeModelsError:
    def test_exception_suppressed(self, tmp_path) -> None:
        from lilbee.cli.tui.widgets.model_bar import _collect_native_models

        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        buckets: dict[str, list[ModelOption]] = {
            "chat": [],
            "embedding": [],
            "vision": [],
        }
        seen: set[str] = set()
        with mock.patch(
            "lilbee.modelhub.registry.ModelRegistry",
            side_effect=RuntimeError("boom"),
        ):
            _collect_native_models(buckets, seen)
        assert buckets["chat"] == []

    def test_collect_remote_models_exception_suppressed(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _collect_remote_models

        buckets: dict[str, list[ModelOption]] = {
            "chat": [],
            "embedding": [],
            "vision": [],
        }
        seen: set[str] = set()
        with mock.patch(
            "lilbee.modelhub.model_manager.classify_remote_models",
            side_effect=RuntimeError("boom"),
        ):
            _collect_remote_models(buckets, seen)
        assert buckets["chat"] == []

    def test_collect_remote_models_adds_provider_label(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _collect_remote_models
        from lilbee.modelhub.model_manager import RemoteModel

        buckets: dict[str, list[ModelOption]] = {
            "chat": [],
            "embedding": [],
            "vision": [],
        }
        seen: set[str] = set()
        with mock.patch(
            "lilbee.modelhub.model_manager.classify_remote_models",
            return_value=[
                RemoteModel(
                    name="llama3:8b",
                    task="chat",
                    family="llama",
                    parameter_size="8B",
                    provider="Ollama",
                )
            ],
        ):
            _collect_remote_models(buckets, seen)
        assert len(buckets["chat"]) == 1
        assert buckets["chat"][0].label == "llama3:8b (Ollama)"
        assert buckets["chat"][0].ref == "ollama/llama3:8b"

    def test_collect_remote_models_unknown_task_dropped(self) -> None:
        """Remote models with an unknown task are dropped, not misclassified into chat."""
        from lilbee.cli.tui.widgets.model_bar import _collect_remote_models
        from lilbee.modelhub.model_manager import RemoteModel

        buckets: dict[str, list[ModelOption]] = {
            "chat": [],
            "embedding": [],
            "vision": [],
            "rerank": [],
        }
        seen: set[str] = set()
        with mock.patch(
            "lilbee.modelhub.model_manager.classify_remote_models",
            return_value=[
                RemoteModel(
                    name="mystery:latest",
                    task="unknown_task",
                    family="",
                    parameter_size="",
                    provider="Ollama",
                )
            ],
        ):
            _collect_remote_models(buckets, seen)
        assert all(not bucket for bucket in buckets.values())
        assert "ollama/mystery:latest" not in seen

    def test_collect_api_models_adds_frontier_models(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _collect_api_models
        from lilbee.modelhub.model_manager import RemoteModel

        buckets: dict[str, list[ModelOption]] = {
            "chat": [],
            "embedding": [],
            "vision": [],
        }
        seen: set[str] = set()
        with mock.patch(
            "lilbee.modelhub.model_manager.discover_api_models",
            return_value={
                "OpenAI": [
                    RemoteModel(
                        name="gpt-4o",
                        task="chat",
                        family="",
                        parameter_size="",
                        provider="OpenAI",
                    ),
                ],
            },
        ):
            _collect_api_models(buckets, seen)
        assert len(buckets["chat"]) == 1
        assert buckets["chat"][0].label == "gpt-4o (OpenAI)"
        assert buckets["chat"][0].ref == "openai/gpt-4o"

    def test_collect_remote_models_skipped_when_litellm_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the SDK extra the picker can't route to remote models, so skip discovery."""
        from lilbee.cli.tui.widgets.model_bar import _collect_remote_models

        monkeypatch.setattr("lilbee.providers.litellm_sdk.litellm_available", lambda: False)
        buckets: dict[str, list[ModelOption]] = {"chat": [], "embedding": [], "vision": []}
        seen: set[str] = set()
        with mock.patch("lilbee.modelhub.model_manager.classify_remote_models") as classify:
            _collect_remote_models(buckets, seen)
        classify.assert_not_called()
        assert buckets["chat"] == []

    def test_collect_api_models_skipped_when_litellm_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API discovery is gated on the same SDK availability."""
        from lilbee.cli.tui.widgets.model_bar import _collect_api_models

        monkeypatch.setattr("lilbee.providers.litellm_sdk.litellm_available", lambda: False)
        buckets: dict[str, list[ModelOption]] = {"chat": [], "embedding": [], "vision": []}
        seen: set[str] = set()
        with mock.patch("lilbee.modelhub.model_manager.discover_api_models") as discover:
            _collect_api_models(buckets, seen)
        discover.assert_not_called()
        assert buckets["chat"] == []

    def test_collect_api_models_exception_suppressed(self) -> None:
        import lilbee.modelhub.model_manager as mm
        from lilbee.cli.tui.widgets.model_bar import _collect_api_models

        buckets: dict[str, list[ModelOption]] = {
            "chat": [],
            "embedding": [],
            "vision": [],
        }
        seen: set[str] = set()
        original = mm.discover_api_models
        mm.discover_api_models = mock.Mock(side_effect=RuntimeError("boom"))
        try:
            _collect_api_models(buckets, seen)
        finally:
            mm.discover_api_models = original
        assert buckets["chat"] == []

    def test_collect_api_models_skips_duplicates(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _collect_api_models
        from lilbee.modelhub.model_manager import RemoteModel

        buckets: dict[str, list[ModelOption]] = {
            "chat": [],
            "embedding": [],
            "vision": [],
        }
        seen: set[str] = {"openai/gpt-4o"}
        with mock.patch(
            "lilbee.modelhub.model_manager.discover_api_models",
            return_value={
                "OpenAI": [
                    RemoteModel(
                        name="gpt-4o",
                        task="chat",
                        family="",
                        parameter_size="",
                        provider="OpenAI",
                    ),
                ],
            },
        ):
            _collect_api_models(buckets, seen)
        assert buckets["chat"] == []


# ---------------------------------------------------------------------------
# ModelCard additional coverage tests
# ---------------------------------------------------------------------------


class TestModelCardBuildHelpers:
    def test_build_specs_all_empty(self) -> None:
        from lilbee.cli.tui.widgets.model_card import _build_specs

        result = _build_specs("--", "--", "--")
        assert str(result) == "--"

    def test_build_specs_all_blank(self) -> None:
        from lilbee.cli.tui.widgets.model_card import _build_specs

        result = _build_specs("", "", "")
        assert str(result) == "--"

    def test_build_status_not_installed_zero_downloads(self) -> None:
        from dataclasses import dataclass

        from lilbee.cli.tui.widgets.model_card import _build_local_status as _build_status

        @dataclass
        class FakeRow:
            installed: bool
            sort_downloads: int
            downloads: str

        row = FakeRow(installed=False, sort_downloads=0, downloads="--")
        assert _build_status(row) is None  # type: ignore[arg-type]

    def test_frontier_card_adds_frontier_class(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import FrontierCatalogRow, KeyStatus
        from lilbee.cli.tui.widgets.model_card import ModelCard

        row = FrontierCatalogRow(
            name="gemini-2.0-flash",
            ref="gemini-2.0-flash",
            task="chat",
            provider="Gemini",
            provider_id="gemini",
            key_status=KeyStatus.READY,
        )
        card = ModelCard(row)
        assert card.has_class("-frontier")

    def test_key_status_pill_missing_key(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import KeyStatus
        from lilbee.cli.tui.widgets.model_card import _key_status_pill

        ready = _key_status_pill(KeyStatus.READY)
        missing = _key_status_pill(KeyStatus.MISSING_KEY)
        assert "ready" in ready.plain
        assert "needs key" in missing.plain

    async def test_compose_frontier_renders_frontier_branch(self) -> None:
        """ModelCard.compose for a FrontierCatalogRow renders the
        frontier-specific child tree (provider + key-status pills)."""

        from lilbee.cli.tui.screens.catalog_utils import FrontierCatalogRow, KeyStatus
        from lilbee.cli.tui.widgets.model_card import ModelCard

        row = FrontierCatalogRow(
            name="gpt-4o",
            ref="openai/gpt-4o",
            task="chat",
            provider="OpenAI",
            provider_id="openai",
            key_status=KeyStatus.MISSING_KEY,
        )

        class _Probe(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield ModelCard(row)

        async with _Probe().run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            card = pilot.app.query_one(ModelCard)
            body = card.query_one(".card-body")
            text = str(body.content)
            assert "gpt-4o" in text
            assert "OpenAI" in text
            assert "needs key" in text


# ---------------------------------------------------------------------------
# TaskBar additional coverage tests
# ---------------------------------------------------------------------------


class TestTaskBarAdditional:
    async def test_single_active_task_shows_name_in_label(self) -> None:
        """One active task displays its name in the status label."""
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            bar.add_task("Sync docs", "sync")
            bar.queue.advance()
            bar._refresh_display()
            await pilot.pause()
            label = bar.query_one("#task-status-label", Label)
            assert "Sync docs" in str(label._Static__content)  # type: ignore[attr-defined]

    async def test_multiple_active_tasks_shows_count(self) -> None:
        """Two or more active tasks show a running count instead of a name."""
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            bar.add_task("Download A", "download")
            bar.add_task("Sync B", "sync")
            bar.queue.advance("download")
            bar.queue.advance("sync")
            bar._refresh_display()
            await pilot.pause()
            label = bar.query_one("#task-status-label", Label)
            text = str(label._Static__content)  # type: ignore[attr-defined]
            assert "2 tasks running" in text

    async def test_queued_tasks_shown_in_label(self) -> None:
        """Queued tasks appear as 'N queued' in the status label."""
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            bar.add_task("Download A", "download")
            bar.queue.advance()
            bar.add_task("Sync B", "sync")
            bar.add_task("Crawl C", "crawl")
            bar._refresh_display()
            await pilot.pause()
            label = bar.query_one("#task-status-label", Label)
            text = str(label._Static__content)  # type: ignore[attr-defined]
            assert "2 queued" in text

    async def test_no_tasks_hides_bar(self) -> None:
        """When no tasks exist, the bar is hidden."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            assert bar.display is False

    async def test_only_queued_no_active_shows_bar(self) -> None:
        """Queued-only tasks (no active) still show the bar."""
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            bar.add_task("Sync", "sync")
            bar._refresh_display()
            await pilot.pause()
            assert bar.display is True
            label = bar.query_one("#task-status-label", Label)
            text = str(label._Static__content)  # type: ignore[attr-defined]
            assert "queued" in text

    async def test_label_contains_task_center_hint(self) -> None:
        """The status label includes the Task Center hint."""
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            bar.add_task("Sync", "sync")
            bar.queue.advance()
            bar._refresh_display()
            await pilot.pause()
            label = bar.query_one("#task-status-label", Label)
            assert "Press t for Tasks" in str(label._Static__content)  # type: ignore[attr-defined]

    async def test_active_task_with_progress_shows_percentage(self) -> None:
        """An active task with nonzero progress shows its percentage."""
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            task_id = bar.add_task("Download", "download")
            bar.queue.advance()
            bar.update_task(task_id, 45)
            bar._refresh_display()
            await pilot.pause()
            label = bar.query_one("#task-status-label", Label)
            assert "45.0%" in str(label._Static__content)  # type: ignore[attr-defined]

    async def test_refresh_display_exception_suppressed(self) -> None:
        """_refresh_display handles missing label gracefully."""
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            bar.add_task("test", "download")
            bar.queue.advance()
            # Remove the label to trigger the except path
            label = bar.query_one("#task-status-label", Label)
            label.remove()
            await pilot.pause()
            # Should not raise
            bar._refresh_display()


class TestPendingSyncHint:
    """Cover TaskBarController state for the pending-sync hint."""

    async def test_set_and_clear_pending(self) -> None:
        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.task_bar.pending_sync_count == 0
            app.task_bar.set_pending_sync(5)
            assert app.task_bar.pending_sync_count == 5
            app.task_bar.clear_pending_sync()
            assert app.task_bar.pending_sync_count == 0

    async def test_set_negative_clamps_to_zero(self) -> None:
        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.task_bar.set_pending_sync(-3)
            assert app.task_bar.pending_sync_count == 0

    async def test_singular_hint_for_one_pending(self) -> None:
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            app.task_bar.set_pending_sync(1)
            bar._refresh_display()
            await pilot.pause()
            assert bar.display is True
            label = bar.query_one("#task-status-label", Label)
            text = str(label._Static__content)  # type: ignore[attr-defined]
            assert "1 doc to sync" in text

    async def test_pending_hint_uses_input_copy_when_input_focused(self) -> None:
        """When a chat Input has focus, the hint adds an Esc prefix to the keybind."""
        from textual.widgets import Input, Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            with mock.patch.object(type(app), "focused", new=property(lambda self: Input())):
                app.task_bar.set_pending_sync(2)
                bar._refresh_display()
                await pilot.pause()
                label = bar.query_one("#task-status-label", Label)
                text = str(label._Static__content)  # type: ignore[attr-defined]
                assert "Esc then S to sync" in text

    async def test_pending_sync_template_focused_lookup_failure_falls_back(self) -> None:
        """If app.focused raises, the template falls back to the non-input copy."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)

            def _raise(_self):
                raise RuntimeError("boom")

            with mock.patch.object(type(app), "focused", new=property(_raise)):
                template = bar._pending_sync_template(3)
                assert "Esc" not in template

    async def test_active_task_overrides_pending_hint(self) -> None:
        """A live task takes the bar; pending hint is suppressed."""
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            app.task_bar.set_pending_sync(2)
            bar.add_task("Sync docs", "sync")
            bar.queue.advance()
            bar._refresh_display()
            await pilot.pause()
            label = bar.query_one("#task-status-label", Label)
            text = str(label._Static__content)  # type: ignore[attr-defined]
            assert "Sync docs" in text
            assert "docs to sync" not in text

    async def test_start_detect_pending_writes_count(self) -> None:
        """The daemon-thread detect job writes the result back to the controller."""
        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            with mock.patch("lilbee.data.ingest.detect_pending", return_value=4):
                app.task_bar.start_detect_pending()
                # Daemon thread is fast but not synchronous; spin until it lands.
                for _ in range(50):
                    if app.task_bar.pending_sync_count == 4:
                        break
                    await pilot.pause()
            assert app.task_bar.pending_sync_count == 4

    async def test_start_detect_pending_no_op_if_already_running(self) -> None:
        """A second call while a detect is in flight does not start a new thread."""
        import threading as _threading

        gate = _threading.Event()

        def _slow_detect() -> int:
            gate.wait(timeout=2)
            return 7

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            with mock.patch("lilbee.data.ingest.detect_pending", side_effect=_slow_detect):
                app.task_bar.start_detect_pending()
                first = app.task_bar._detect_thread
                app.task_bar.start_detect_pending()
                second = app.task_bar._detect_thread
                assert first is second
                gate.set()

    async def test_start_detect_pending_swallows_errors(self) -> None:
        """A detect_pending exception is logged but does not crash the worker."""
        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.task_bar.set_pending_sync(2)
            with mock.patch("lilbee.data.ingest.detect_pending", side_effect=RuntimeError("boom")):
                app.task_bar.start_detect_pending()
                for _ in range(50):
                    if (
                        app.task_bar._detect_thread is not None
                        and not app.task_bar._detect_thread.is_alive()
                    ):
                        break
                    await pilot.pause()
            # Previous count is preserved; failure does not zero it out.
            assert app.task_bar.pending_sync_count == 2


class TestEnsureChromium:
    """bb-wq8g: TaskBarController.ensure_chromium short-circuits or spawns SETUP."""

    async def test_short_circuits_when_installed(self) -> None:
        """No SETUP task enqueued; on_ready fires immediately."""
        import threading as _threading

        from lilbee.cli.tui.task_queue import TaskType

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            with mock.patch(
                "lilbee.cli.tui.widgets.task_bar_controller.chromium_installed",
                return_value=True,
            ):
                fired = _threading.Event()
                app.task_bar.ensure_chromium(fired.set)
                assert fired.is_set()
                queued = app.task_bar.queue
                all_tasks = queued.active_tasks + queued.queued_tasks + queued.history
                assert not any(t.task_type == TaskType.SETUP.value for t in all_tasks)

    async def test_enqueues_setup_task_when_missing(self) -> None:
        """bb-wq8g happy path: SETUP task calls start_task with the right args.

        Asserts against ``start_task`` directly instead of spawning a real
        worker thread, to keep this unit test off the bootstrap path.
        """
        from lilbee.cli.tui.task_queue import TaskType

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            on_ready = mock.Mock()
            with (
                mock.patch(
                    "lilbee.cli.tui.widgets.task_bar_controller.chromium_installed",
                    return_value=False,
                ),
                mock.patch.object(app.task_bar, "start_task") as mock_start,
            ):
                app.task_bar.ensure_chromium(on_ready)
            mock_start.assert_called_once()
            args, kwargs = mock_start.call_args
            assert args[1] == TaskType.SETUP
            assert kwargs.get("on_success") is on_ready
            on_ready.assert_not_called()


class TestChromiumBootstrapTarget:
    """bb-wq8g: directly drive _chromium_bootstrap_target's body."""

    def test_forwards_setup_progress_with_known_total(self) -> None:
        """With total_bytes set, the target formats 'chromium: N/M MB'."""
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.widgets import task_bar_controller
        from lilbee.runtime.progress import EventType, SetupDoneEvent, SetupProgressEvent

        reporter = mock.MagicMock()

        async def _fake_bootstrap(on_progress=None):
            on_progress(
                EventType.SETUP_DONE,  # ignored by the forward filter
                SetupDoneEvent(component="chromium", success=True, error=None),
            )
            on_progress(
                EventType.SETUP_PROGRESS,
                "not a SetupProgressEvent",  # type: ignore[arg-type]
            )
            on_progress(
                EventType.SETUP_PROGRESS,
                SetupProgressEvent(
                    component="chromium",
                    downloaded_bytes=10 * 1024 * 1024,
                    total_bytes=40 * 1024 * 1024,
                    detail="...",
                ),
            )
            on_progress(
                EventType.SETUP_PROGRESS,
                SetupProgressEvent(
                    component="chromium",
                    downloaded_bytes=5 * 1024 * 1024,
                    total_bytes=None,
                    detail="...",
                ),
            )

        with mock.patch.object(task_bar_controller, "bootstrap_chromium", new=_fake_bootstrap):
            task_bar_controller._chromium_bootstrap_target(reporter)

        pct_detail_calls = [call.args for call in reporter.update.call_args_list]
        assert (25, msg.SETUP_CHROMIUM_DETAIL.format(done=10, total=40)) in pct_detail_calls
        assert (0, msg.SETUP_CHROMIUM_DETAIL_UNKNOWN.format(done=5)) in pct_detail_calls


class TestTaskBarIndeterminate:
    """Tests for indeterminate task flag propagation."""

    async def test_add_task_indeterminate_creates_indeterminate_task(self) -> None:
        """Tasks created with indeterminate=True start in indeterminate mode."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            task_id = bar.add_task("Sync", "sync", indeterminate=True)
            task = bar.queue.get_task(task_id)
            assert task is not None
            assert task.indeterminate is True

    async def test_enqueue_indeterminate_flag(self) -> None:
        """TaskQueue.enqueue passes indeterminate to the Task."""
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "Add", "add", indeterminate=True)
        task = q.get_task(tid)
        assert task is not None
        assert task.indeterminate is True

    async def test_controller_add_task_indeterminate(self) -> None:
        """TaskBarController.add_task passes indeterminate through to queue."""
        from lilbee.cli.tui.widgets.task_bar_controller import TaskBarController

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            controller = app.task_bar
            assert isinstance(controller, TaskBarController)
            task_id = controller.add_task("Sync", "sync", indeterminate=True)
            task = controller.queue.get_task(task_id)
            assert task is not None
            assert task.indeterminate is True

    async def test_indeterminate_task_shows_in_label(self) -> None:
        """An indeterminate active task still renders its name in the label."""
        from textual.widgets import Label

        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            bar.add_task("Sync", "sync", indeterminate=True)
            bar.queue.advance()
            bar._refresh_display()
            await pilot.pause()
            label = bar.query_one("#task-status-label", Label)
            assert "Sync" in str(label._Static__content)  # type: ignore[attr-defined]
            assert bar.display is True


class TestGridSelectExtra:
    async def test_highlight_first(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlight_first()
            assert grid.highlighted == 0

    async def test_highlight_last(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlight_last()
            assert grid.highlighted == len(grid.children) - 1

    async def test_highlight_last_empty(self) -> None:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _EmptyGridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlight_last()
            # No children, highlighted stays None
            assert grid.highlighted is None

    async def test_cursor_up_within_grid(self) -> None:
        """Cover line 143: highlighted -= width (move up one row)."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _LargeGridApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            gs = grid.grid_size
            assert gs is not None
            width = gs[0]
            assert len(grid.children) > width, f"Need multiple rows: {len(grid.children)}"
            grid.highlighted = width  # first cell of second row
            grid.action_cursor_up()
            assert grid.highlighted == 0

    async def test_cursor_down_within_grid(self) -> None:
        """Cover line 156: highlighted += width (move down one row)."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _LargeGridApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            gs = grid.grid_size
            assert gs is not None
            width = gs[0]
            assert len(grid.children) > width, f"Need multiple rows: {len(grid.children)}"
            grid.highlighted = 0
            grid.action_cursor_down()
            assert grid.highlighted == width

    async def test_action_select_posts_selected(self) -> None:
        """Cover line 195: post_message(Selected(...))."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = 0
            grid.action_select()
            await pilot.pause()
            assert grid.highlighted == 0

    async def test_tab_next_escapes_at_last_card(self) -> None:
        """Tab on the last card posts LeaveDown to escape the grid."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = len(grid.children) - 1
            messages: list[object] = []
            orig_post = grid.post_message
            grid.post_message = lambda m: messages.append(m) or orig_post(m)  # type: ignore[assignment]
            grid.action_tab_next()
            assert any(isinstance(m, GridSelect.LeaveDown) for m in messages)

    async def test_tab_previous_escapes_at_first_card(self) -> None:
        """Shift+Tab on the first card posts LeaveUp to escape the grid."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = _GridApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            grid = app.query_one(GridSelect)
            grid.highlighted = 0
            messages: list[object] = []
            orig_post = grid.post_message
            grid.post_message = lambda m: messages.append(m) or orig_post(m)  # type: ignore[assignment]
            grid.action_tab_previous()
            assert any(isinstance(m, GridSelect.LeaveUp) for m in messages)

    def test_tab_next_empty_grid_posts_leave_down(self) -> None:
        """Tab on an empty grid posts LeaveDown."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        grid = GridSelect(min_column_width=20)
        messages: list[object] = []
        grid.post_message = lambda m: messages.append(m)  # type: ignore[assignment]
        grid.action_tab_next()
        assert any(isinstance(m, GridSelect.LeaveDown) for m in messages)

    def test_tab_previous_empty_grid_posts_leave_up(self) -> None:
        """Shift+Tab on an empty grid posts LeaveUp."""
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        grid = GridSelect(min_column_width=20)
        messages: list[object] = []
        grid.post_message = lambda m: messages.append(m)  # type: ignore[assignment]
        grid.action_tab_previous()
        assert any(isinstance(m, GridSelect.LeaveUp) for m in messages)


# ---------------------------------------------------------------------------
# ModelCard: _build_status with positive downloads
# ---------------------------------------------------------------------------


class TestModelCardBuildStatusDownloads:
    def test_build_status_with_downloads(self) -> None:
        from dataclasses import dataclass

        from lilbee.cli.tui.widgets.model_card import _build_local_status as _build_status

        @dataclass
        class FakeRow:
            installed: bool
            sort_downloads: int
            downloads: str

        row = FakeRow(installed=False, sort_downloads=1000, downloads="1K")
        result = _build_status(row)  # type: ignore[arg-type]
        assert result is not None
        assert "1K" in str(result)


class TestConfirmDialog:
    async def test_confirm_with_y_key(self) -> None:
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        results: list[bool] = []

        class _App(LilbeeAppHost):
            def on_mount(self):
                self.push_screen(ConfirmDialog("Title", "Message"), results.append)

        app = _App()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()

        assert results == [True]

    async def test_cancel_with_n_key(self) -> None:
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        results: list[bool] = []

        class _App(LilbeeAppHost):
            def on_mount(self):
                self.push_screen(ConfirmDialog("Title", "Message"), results.append)

        app = _App()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()

        assert results == [False]

    async def test_cancel_with_escape(self) -> None:
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        results: list[bool] = []

        class _App(LilbeeAppHost):
            def on_mount(self):
                self.push_screen(ConfirmDialog("Title", "Message"), results.append)

        app = _App()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert results == [False]

    async def test_confirm_with_yes_button(self) -> None:
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        results: list[bool] = []

        class _App(LilbeeAppHost):
            def on_mount(self):
                self.push_screen(ConfirmDialog("Title", "Message"), results.append)

        app = _App()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        assert results == [True]

    async def test_yes_pill_click(self) -> None:
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        results: list[bool] = []

        class _App(LilbeeAppHost):
            def on_mount(self):
                self.push_screen(ConfirmDialog("Title", "Message"), results.append)

        app = _App()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.click("#confirm-yes")
            await pilot.pause()

        assert results == [True]

    async def test_no_pill_click(self) -> None:
        from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

        results: list[bool] = []

        class _App(LilbeeAppHost):
            def on_mount(self):
                self.push_screen(ConfirmDialog("Title", "Message"), results.append)

        app = _App()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.click("#confirm-no")
            await pilot.pause()

        assert results == [False]


class CrawlDialogTestApp(LilbeeAppHost):
    def __init__(self):
        super().__init__()
        self.results: list = []

    def on_mount(self):
        from lilbee.cli.tui.widgets.crawl_dialog import CrawlDialog

        self.push_screen(CrawlDialog(), self.results.append)


async def test_crawl_dialog_submit_valid():
    """Submitting with a valid URL and explicit advanced caps returns CrawlParams."""
    from lilbee.cli.tui.widgets.crawl_dialog import CrawlParams

    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        depth_input = app.screen.query_one("#crawl-depth-input")
        depth_input.value = "2"
        max_input = app.screen.query_one("#crawl-max-pages-input")
        max_input.value = "10"
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()

    assert len(app.results) == 1
    result = app.results[0]
    assert isinstance(result, CrawlParams)
    assert result.url == "https://example.com"
    assert result.depth == 2
    assert result.max_pages == 10


async def test_crawl_dialog_cancel():
    """Cancel button dismisses with None."""
    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        app.screen.query_one("#crawl-cancel", Button).press()
        await pilot.pause()

    assert app.results == [None]


async def test_crawl_dialog_escape_cancels():
    """Escape key dismisses with None."""
    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert app.results == [None]


async def test_crawl_dialog_empty_url_shows_error():
    """Submitting with empty URL shows validation error."""
    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        app.screen.query_one("#crawl-submit", Button).press()
        await pilot.pause()
        error = app.screen.query_one("#crawl-error", Static)
        assert "required" in str(error.render()).lower()


async def test_crawl_dialog_invalid_url_shows_error():
    """Invalid URL shows validation error from require_valid_crawl_url."""
    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        await pilot.pause()
        with mock.patch(
            "lilbee.crawler.require_valid_crawl_url",
            side_effect=ValueError("bad url"),
        ):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()
            error = app.screen.query_one("#crawl-error", Static)
            assert "bad url" in str(error.render()).lower()


async def test_crawl_dialog_invalid_depth_shows_error():
    """Non-numeric depth shows validation error."""
    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        depth_input = app.screen.query_one("#crawl-depth-input")
        depth_input.value = "abc"
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()
            error = app.screen.query_one("#crawl-error", Static)
            assert "depth" in str(error.render()).lower()


async def test_crawl_dialog_invalid_max_pages_shows_error():
    """Non-numeric max pages shows validation error."""
    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        max_input = app.screen.query_one("#crawl-max-pages-input")
        max_input.value = "abc"
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()
            error = app.screen.query_one("#crawl-error", Static)
            assert "max pages" in str(error.render()).lower()


async def test_crawl_dialog_negative_depth_shows_error():
    """Negative depth shows validation error."""
    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        depth_input = app.screen.query_one("#crawl-depth-input")
        depth_input.value = "-1"
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()
            error = app.screen.query_one("#crawl-error", Static)
            assert "depth" in str(error.render()).lower()


async def test_crawl_dialog_input_submitted():
    """Pressing Enter in an input field triggers submit."""
    from lilbee.cli.tui.widgets.crawl_dialog import CrawlParams

    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            await pilot.press("enter")
            await pilot.pause()

    assert len(app.results) == 1
    assert isinstance(app.results[0], CrawlParams)


async def test_crawl_dialog_defaults():
    """Default submit (recursive checked, advanced blank) yields None caps."""
    from lilbee.cli.tui.widgets.crawl_dialog import CrawlParams

    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()

    result = app.results[0]
    assert isinstance(result, CrawlParams)
    assert result.depth is None
    assert result.max_pages is None


async def test_crawl_dialog_negative_max_pages_shows_error():
    """Non-positive max pages shows validation error."""
    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        max_input = app.screen.query_one("#crawl-max-pages-input")
        max_input.value = "-5"
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()
            error = app.screen.query_one("#crawl-error", Static)
            assert "max pages" in str(error.render()).lower()


async def test_crawl_dialog_zero_max_pages_shows_error():
    """Zero max pages is invalid (blank means unbounded; 0 is nonsense)."""
    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        max_input = app.screen.query_one("#crawl-max-pages-input")
        max_input.value = "0"
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()
            error = app.screen.query_one("#crawl-error", Static)
            assert "max pages" in str(error.render()).lower()


async def test_crawl_dialog_empty_advanced_fields_are_unbounded():
    """Empty Advanced fields submit as None (unbounded)."""
    from lilbee.cli.tui.widgets.crawl_dialog import CrawlParams

    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        depth_input = app.screen.query_one("#crawl-depth-input")
        depth_input.value = ""
        max_input = app.screen.query_one("#crawl-max-pages-input")
        max_input.value = ""
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()

    result = app.results[0]
    assert isinstance(result, CrawlParams)
    assert result.depth is None
    assert result.max_pages is None


async def test_crawl_dialog_unchecking_recursive_submits_depth_zero():
    """Unchecking the Recursive checkbox submits with depth=0 (single URL)."""
    from textual.widgets import Checkbox

    from lilbee.cli.tui.widgets.crawl_dialog import CrawlParams

    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "https://example.com"
        checkbox = app.screen.query_one("#crawl-recursive-checkbox", Checkbox)
        checkbox.value = False
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()

    result = app.results[0]
    assert isinstance(result, CrawlParams)
    assert result.depth == 0
    assert result.max_pages is None


async def test_crawl_dialog_recursive_checkbox_default_checked():
    """The Recursive checkbox defaults to checked on dialog open."""
    from textual.widgets import Checkbox

    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        checkbox = app.screen.query_one("#crawl-recursive-checkbox", Checkbox)
        assert checkbox.value is True


async def test_crawl_dialog_auto_prefix_https():
    """URL without scheme gets https:// auto-prefixed."""
    from lilbee.cli.tui.widgets.crawl_dialog import CrawlParams

    app = CrawlDialogTestApp()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        url_input = app.screen.query_one("#crawl-url-input")
        url_input.value = "example.com"
        await pilot.pause()
        with mock.patch("lilbee.crawler.require_valid_crawl_url"):
            app.screen.query_one("#crawl-submit", Button).press()
            await pilot.pause()

    result = app.results[0]
    assert isinstance(result, CrawlParams)
    assert result.url == "https://example.com"


def _make_list_row(
    name: str = "test",
    task: str = "chat",
    params: str = "7B",
    size: str = "4.0 GB",
    quant: str = "Q4_K_M",
    downloads: str = "1K",
    featured: bool = False,
    installed: bool = False,
    sort_downloads: int = 1000,
    backend: str = "native",
) -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task=task,
        params=params,
        size=size,
        quant=quant,
        downloads=downloads,
        featured=featured,
        installed=installed,
        sort_downloads=sort_downloads,
        sort_size=4.0,
        backend=backend,
    )


def _make_click(widget: Any, button: int = 1) -> Any:
    """Construct a Click event for a detached widget. Coordinates are arbitrary."""
    from textual.events import Click

    return Click(
        widget=widget,
        x=0,
        y=0,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=0,
        screen_y=0,
    )


class TestSearchHFCtaItem:
    """Direct-construction tests for SearchHFCtaItem."""

    def test_action_select_posts_message_with_term(self) -> None:
        from lilbee.cli.tui.widgets.search_hf_cta_item import SearchHFCtaItem

        item = SearchHFCtaItem("qwen3")
        received: list[SearchHFCtaItem.Selected] = []
        item.post_message = received.append  # type: ignore[method-assign]
        item.action_select()
        assert len(received) == 1
        assert received[0].term == "qwen3"
        assert received[0].control is item

    def test_on_click_focuses_and_posts(self) -> None:
        from lilbee.cli.tui.widgets.search_hf_cta_item import SearchHFCtaItem

        item = SearchHFCtaItem("phi-3")
        received: list[SearchHFCtaItem.Selected] = []
        focus_calls: list[bool] = []
        item.post_message = received.append  # type: ignore[method-assign]
        item.focus = lambda: focus_calls.append(True)  # type: ignore[method-assign]
        item.on_click(_make_click(item))
        assert focus_calls == [True]
        assert received and received[0].term == "phi-3"

    async def test_compose_yields_label_with_term(self) -> None:
        from textual.widgets import Static

        from lilbee.cli.tui.widgets.search_hf_cta_item import SearchHFCtaItem

        class _App(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield SearchHFCtaItem("phi-3")

        app = _App()
        async with app.run_test() as pilot:
            await pilot.pause()
            label = app.query_one("#cta-label", Static)
            assert "phi-3" in str(label.render())


def _vgrid_row(name: str = "phi-3") -> LocalCatalogRow:
    return LocalCatalogRow(
        name=name,
        task="chat",
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=False,
        sort_downloads=0,
        sort_size=0.0,
        ref=name,
        backend="native",
    )


class TestModelGridOnClick:
    """Click hit-test math: first click highlights, second click posts Selected."""

    @staticmethod
    def _click(x: int, y: int) -> object:
        click = mock.Mock()
        click.x = x
        click.y = y
        return click

    def test_first_click_highlights_second_click_selects(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(2)]
        grid = ModelGrid(rows)
        grid._cards_per_row = 2
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.focus = lambda: None  # type: ignore[method-assign]
        # Stub the size so _cell_at gets non-zero width without an app.
        grid._size = mock.Mock(width=80, height=20)  # type: ignore[attr-defined]
        received: list[ModelGrid.Selected] = []
        grid.post_message = received.append  # type: ignore[method-assign]
        click = self._click(60, 1)  # second column, first card line
        grid.on_click(click)
        assert grid.highlighted == 1
        assert received == []
        grid.on_click(click)
        assert received and isinstance(received[0], ModelGrid.Selected)
        assert received[0].row is rows[1]

    def test_click_outside_dataset_is_ignored(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a")])
        grid._cards_per_row = 4
        grid._size = mock.Mock(width=80, height=20)  # type: ignore[attr-defined]
        click = self._click(70, 1)  # column 3 with only 1 row -> out of range
        grid.on_click(click)
        assert grid.highlighted is None

    def test_click_below_grid_rows_is_ignored(self) -> None:
        """Clicks past the last data row land outside the grid and do nothing."""
        from lilbee.cli.tui.widgets.model_grid import _ROW_HEIGHT, ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(2)]  # one row of 2 cards
        grid = ModelGrid(rows)
        grid._cards_per_row = 2
        grid._size = mock.Mock(width=60, height=20)  # type: ignore[attr-defined]
        # y just past the only mounted row of cards lands in empty space.
        click = self._click(0, _ROW_HEIGHT + 1)
        grid.on_click(click)
        assert grid.highlighted is None

    def test_click_on_empty_grid_is_ignored(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([])
        grid._size = mock.Mock(width=80, height=20)  # type: ignore[attr-defined]
        click = self._click(10, 1)
        grid.on_click(click)
        assert grid.highlighted is None

    def test_click_at_negative_y_is_ignored(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a")])
        grid._cards_per_row = 1
        grid._size = mock.Mock(width=80, height=20)  # type: ignore[attr-defined]
        click = self._click(10, -1)
        grid.on_click(click)
        assert grid.highlighted is None


class TestModelGridScrollIntoView:
    """watch_highlighted must scroll the highlighted cell into view so that
    cursor moves wake the outer container's scroll watcher (the trigger that
    drives pagination)."""

    @staticmethod
    def _patch_size(grid: object, width: int, height: int = 20):
        from unittest.mock import PropertyMock

        from textual.geometry import Size

        return mock.patch.object(
            type(grid),
            "size",
            new_callable=PropertyMock,
            return_value=Size(width, height),
        )

    def test_cursor_move_calls_scroll_to_region(self) -> None:
        """The grid delegates the scroll-into-view to its parent (the outer
        VerticalScroll), translating the cell offset via ``virtual_region``.
        """
        from textual.geometry import Region

        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(8)]
        grid = ModelGrid(rows)
        grid._cards_per_row = 2
        captured: list[Region] = []
        # Stub a parent widget exposing scroll_to_region. We intentionally
        # build a minimal stand-in rather than mounting in an App because
        # virtual_region resolves only when mounted; we patch it directly.
        parent = mock.Mock()
        parent.scroll_to_region = lambda region, **_: captured.append(region)
        # `Widget` isinstance check guards parent; satisfy it via the mock spec.
        from textual.widget import Widget

        parent_widget = mock.Mock(spec=Widget)
        parent_widget.scroll_to_region = lambda region, **_: captured.append(region)
        grid._parent = parent_widget  # type: ignore[attr-defined]
        grid.refresh = lambda *_a, **_k: None  # type: ignore[method-assign]
        # virtual_region defaults to Region() (empty) when unmounted; patch.
        from unittest.mock import PropertyMock

        with (
            self._patch_size(grid, 80),
            mock.patch.object(
                type(grid),
                "virtual_region",
                new_callable=PropertyMock,
                return_value=Region(0, 0, 80, 24),
            ),
        ):
            grid.watch_highlighted(None, 4)

        assert captured, "parent.scroll_to_region must run on cursor moves"
        region = captured[0]
        from lilbee.cli.tui.widgets.model_grid import _CARD_HEIGHT, _ROW_HEIGHT

        # Cell at index 4 in 2-col grid -> row 2, col 0; offset is row*_ROW_HEIGHT.
        assert region.y == (4 // 2) * _ROW_HEIGHT
        assert region.height == _CARD_HEIGHT

    def test_highlight_set_to_none_does_not_scroll(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a")])
        grid._cards_per_row = 2
        captured: list[object] = []
        grid.scroll_to_region = lambda region, **_: captured.append(region)  # type: ignore[method-assign]
        grid.refresh = lambda *_a, **_k: None  # type: ignore[method-assign]

        with self._patch_size(grid, 80):
            grid.watch_highlighted(0, None)

        assert captured == []

    def test_zero_width_skips_scroll(self) -> None:
        """At initial mount the size is (0, 0); guard prevents a bogus Region."""
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a")])
        grid._cards_per_row = 1
        captured: list[object] = []
        grid.scroll_to_region = lambda region, **_: captured.append(region)  # type: ignore[method-assign]
        grid.refresh = lambda *_a, **_k: None  # type: ignore[method-assign]

        with self._patch_size(grid, 0, 0):
            grid.watch_highlighted(None, 0)

        assert captured == []


class TestModelPickerScopeTitles:
    """Cover the vision and rerank title branches of _picker_title."""

    def test_vision_title(self) -> None:
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.model_picker import _picker_title

        assert _picker_title("vision") == msg.MODEL_PICKER_TITLE_VISION

    def test_rerank_title(self) -> None:
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.model_picker import _picker_title

        assert _picker_title("rerank") == msg.MODEL_PICKER_TITLE_RERANK


class TestModelGridUtilityMethods:
    """Direct unit coverage of ModelGrid helpers + simple message wrappers."""

    def test_selected_carries_row(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row("a")]
        grid = ModelGrid(rows)
        message = ModelGrid.Selected(grid, rows[0])
        assert message.control is grid
        assert message.row is rows[0]

    def test_columns_for_width_zero_returns_default(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _DEFAULT_COLUMNS, ModelGrid

        assert ModelGrid._columns_for_width(0) == _DEFAULT_COLUMNS
        assert ModelGrid._columns_for_width(-5) == _DEFAULT_COLUMNS

    def test_total_rows_zero_when_empty(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        assert ModelGrid([])._total_rows() == 0

    def test_total_rows_when_columns_zero(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a")])
        grid._cards_per_row = 0
        assert grid._total_rows() == 0

    def test_get_content_height_scales_with_dataset(self) -> None:
        """Height grows linearly in the number of grid rows the dataset needs."""
        from lilbee.cli.tui.widgets.model_grid import _ROW_HEIGHT, ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(8)]
        grid = ModelGrid(rows)
        # Width 80 with the default _CARD_MIN_WIDTH lays out 2 cards per row.
        size = mock.Mock(width=80, height=24)
        height = grid.get_content_height(size, size, 80)
        # 8 rows / 2 cols = 4 grid rows; each row is _ROW_HEIGHT lines tall.
        # No trailing gutter to subtract.
        assert height == 4 * _ROW_HEIGHT

    def test_get_content_height_zero_when_empty(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        size = mock.Mock(width=80, height=24)
        assert ModelGrid([]).get_content_height(size, size, 80) == 0

    def test_get_content_width_returns_container_width(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a")])
        size = mock.Mock(width=42, height=24)
        assert grid.get_content_width(size, size) == 42

    def test_action_select_no_op_when_unset(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a")])
        received: list[ModelGrid.Selected] = []
        grid.post_message = received.append  # type: ignore[method-assign]
        grid.action_select()
        assert received == []

    def test_action_select_no_op_when_dataset_empty(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([])
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = 0  # set despite empty dataset to exercise the guard
        received: list[ModelGrid.Selected] = []
        grid.post_message = received.append  # type: ignore[method-assign]
        grid.action_select()
        assert received == []


class TestModelGridCursorEdges:
    """Cover the LeaveDown / LeaveUp and bound-clip branches of cursor actions."""

    def test_action_cursor_down_at_last_row_emits_leave_down(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(3)]
        grid = ModelGrid(rows)
        grid._cards_per_row = 4  # next_index from any cell falls past the end
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = 2
        received: list[ModelGrid.LeaveDown] = []
        grid.post_message = received.append  # type: ignore[method-assign]
        grid.action_cursor_down()
        assert received and isinstance(received[0], ModelGrid.LeaveDown)

    def test_action_cursor_up_emits_leave_up_at_first_row(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(8)]
        grid = ModelGrid(rows)
        grid._cards_per_row = 4
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = 1
        received: list[ModelGrid.LeaveUp] = []
        grid.post_message = received.append  # type: ignore[method-assign]
        grid.action_cursor_up()
        assert received and isinstance(received[0], ModelGrid.LeaveUp)

    def test_action_cursor_left_clamps_to_zero(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a"), _vgrid_row("b")])
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = 0
        grid.action_cursor_left()
        assert grid.highlighted == 0

    def test_action_cursor_right_clamps_to_last_index(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(2)]
        grid = ModelGrid(rows)
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = 1
        grid.action_cursor_right()
        assert grid.highlighted == 1

    def test_cursor_actions_initialize_highlight_when_unset(self) -> None:
        """First cursor action lands on index 0 instead of moving from None."""
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row(f"m{i}") for i in range(4)])
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        for action in (
            grid.action_cursor_up,
            grid.action_cursor_down,
            grid.action_cursor_left,
            grid.action_cursor_right,
        ):
            grid.highlighted = None
            action()
            assert grid.highlighted == 0


class TestModelGridLayoutEdges:
    """Cover render_line edge cases and the on_resize column recompute."""

    async def test_render_line_blank_above_dataset(self) -> None:
        """A negative y request returns a blank strip rather than crashing."""
        from textual.containers import VerticalScroll

        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(4)]
        grid = ModelGrid(rows, id="mg-render-test")

        class _GridApp(LilbeeAppHost):
            CSS = "ModelGrid { height: 12; width: 80; }"

            def compose(self) -> ComposeResult:
                yield VerticalScroll(grid)

        app = _GridApp()
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            strip = grid.render_line(-1)
            # Strip cell length cleanly maps onto the configured grid width.
            assert strip.cell_length == grid.size.width

    async def test_render_line_blank_past_dataset(self) -> None:
        """Rendering a line past the last grid row returns blank padding."""
        from textual.containers import VerticalScroll

        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(2)]
        grid = ModelGrid(rows, id="mg-render-past")

        class _GridApp(LilbeeAppHost):
            CSS = "ModelGrid { height: 12; width: 80; }"

            def compose(self) -> ComposeResult:
                yield VerticalScroll(grid)

        app = _GridApp()
        async with app.run_test(size=(80, 20)) as pilot:
            await pilot.pause()
            strip = grid.render_line(grid.size.height * 4)
            assert strip.cell_length == grid.size.width

    async def test_render_line_paints_blank_cells_for_short_last_row(self) -> None:
        """When the last row has fewer cards than columns, blank padding fills the gap."""
        from textual.containers import VerticalScroll

        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(3)]  # 3 rows, 2-cols => one short
        grid = ModelGrid(rows, id="mg-short-row")

        class _GridApp(LilbeeAppHost):
            CSS = "ModelGrid { height: 12; width: 60; }"

            def compose(self) -> ComposeResult:
                yield VerticalScroll(grid)

        app = _GridApp()
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause()
            grid._cards_per_row = 2
            # Second grid row, first card line.
            strip = grid.render_line(_row_height_offset(1))
            assert strip.cell_length == grid.size.width

    async def test_on_resize_recomputes_columns(self) -> None:
        """Resizing the container shrinks/grows the column count."""
        from textual.containers import VerticalScroll

        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(8)]
        grid = ModelGrid(rows, id="mg-resize")

        class _GridApp(LilbeeAppHost):
            CSS = "ModelGrid { height: 12; width: 30; }"

            def compose(self) -> ComposeResult:
                yield VerticalScroll(grid)

        app = _GridApp()
        async with app.run_test(size=(30, 20)) as pilot:
            await pilot.pause()
            assert grid.columns_per_row == 1


class TestModelGridHighlightHelpers:
    """Cover highlight_first / highlight_last and the cursor-step branches."""

    def test_highlight_first_sets_index_zero_when_rows_present(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a"), _vgrid_row("b")])
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = None
        grid.highlight_first()
        assert grid.highlighted == 0

    def test_highlight_first_no_op_when_empty(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([])
        grid.highlight_first()
        assert grid.highlighted is None

    def test_highlight_last_lands_on_final_row(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a"), _vgrid_row("b"), _vgrid_row("c")])
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlight_last()
        assert grid.highlighted == 2

    def test_highlight_last_no_op_when_empty(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([])
        grid.highlight_last()
        assert grid.highlighted is None

    def test_action_cursor_up_steps_one_row(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(8)]
        grid = ModelGrid(rows)
        grid._cards_per_row = 4
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = 5
        grid.action_cursor_up()
        # 5 - 4 = 1
        assert grid.highlighted == 1

    def test_action_cursor_down_steps_one_row(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(8)]
        grid = ModelGrid(rows)
        grid._cards_per_row = 4
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = 1
        grid.action_cursor_down()
        # 1 + 4 = 5
        assert grid.highlighted == 5

    def test_set_rows_replaces_dataset_and_resets_highlight(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a"), _vgrid_row("b")])
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = 1
        new_rows = [_vgrid_row("c"), _vgrid_row("d"), _vgrid_row("e")]
        grid.refresh = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.set_rows(new_rows)
        assert grid.rows == new_rows
        assert grid.highlighted is None


def _row_height_offset(grid_row: int) -> int:
    from lilbee.cli.tui.widgets.model_grid import _ROW_HEIGHT

    return grid_row * _ROW_HEIGHT


def _dummy_border_style() -> str:
    """A dummy theme-token string for unit tests that don't render via a theme."""
    return "$primary on $panel"


class TestModelGridCardRendering:
    """``_render_card_strip`` covers both row dataclasses and selection state."""

    def test_local_row_unselected_renders_full_height(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _CARD_HEIGHT, _render_card_strip

        out = _render_card_strip(
            _vgrid_row("phi-3"), selected=False, width=40, border_style=_dummy_border_style()
        )
        assert len(out.lines) == _CARD_HEIGHT

    def test_local_row_selected_paints_install_hint(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _render_card_strip

        out = _render_card_strip(
            _vgrid_row("phi-3"), selected=True, width=40, border_style=_dummy_border_style()
        )
        # The hint slot is the last line; rendered text contains the
        # SETUP_CARD_HINT copy when the local card is highlighted-but-not-installed.
        rendered = "\n".join(str(line) for line in out.lines)
        assert "Enter to install" in rendered

    def test_installed_row_renders_status_pill(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _render_card_strip

        row = _vgrid_row("phi-3")
        row.installed = True
        out = _render_card_strip(row, selected=False, width=40, border_style=_dummy_border_style())
        rendered = "\n".join(str(line) for line in out.lines)
        assert "installed" in rendered

    def test_installed_row_selected_paints_delete_hint(self) -> None:
        """Highlighting an installed card surfaces the D / Backspace
        delete affordance so removing a model is discoverable from the
        card itself, not just the footer."""
        from lilbee.cli.tui.widgets.model_grid import _render_card_strip

        row = _vgrid_row("phi-3")
        row.installed = True
        out = _render_card_strip(row, selected=True, width=40, border_style=_dummy_border_style())
        rendered = "\n".join(str(line) for line in out.lines)
        assert "delete" in rendered
        assert "D" in rendered

    def test_frontier_row_renders_provider_and_key_status(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _render_card_strip

        row = FrontierCatalogRow(
            name="gpt-4o",
            ref="openai/gpt-4o",
            task="chat",
            provider="OpenAI",
            provider_id="openai",
            key_status=KeyStatus.READY,
        )
        out = _render_card_strip(row, selected=False, width=40, border_style=_dummy_border_style())
        rendered = "\n".join(str(line) for line in out.lines)
        assert "OpenAI" in rendered
        assert "ready" in rendered

    def test_frontier_row_missing_key_renders_warning_pill(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _render_card_strip

        row = FrontierCatalogRow(
            name="claude",
            ref="anthropic/claude",
            task="chat",
            provider="Anthropic",
            provider_id="anthropic",
            key_status=KeyStatus.MISSING_KEY,
        )
        out = _render_card_strip(row, selected=False, width=40, border_style=_dummy_border_style())
        rendered = "\n".join(str(line) for line in out.lines)
        assert "needs key" in rendered

    def test_local_row_with_zero_downloads_skips_status_line(self) -> None:
        """Local rows with no downloads omit the download-count line entirely."""
        from lilbee.cli.tui.widgets.model_grid import _render_card_strip

        row = _vgrid_row("noisy")
        # _vgrid_row defaults sort_downloads=0, so this branch is hit by default.
        out = _render_card_strip(row, selected=False, width=40, border_style=_dummy_border_style())
        rendered = "\n".join(str(line) for line in out.lines)
        assert "↓" not in rendered

    def test_local_row_with_downloads_renders_download_count(self) -> None:
        """A non-zero sort_downloads value renders the ``↓ ...`` muted glyph."""
        from lilbee.cli.tui.widgets.model_grid import _render_card_strip

        row = _vgrid_row("popular")
        row.sort_downloads = 12345
        row.downloads = "12K"
        out = _render_card_strip(row, selected=False, width=40, border_style=_dummy_border_style())
        rendered = "\n".join(str(line) for line in out.lines)
        assert "↓ 12K" in rendered

    def test_local_row_pads_short_specs_with_double_dash(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _build_specs

        # All-default-row specs produce the placeholder.
        assert str(_build_specs("--", "--", "--")) == "--"

    def test_pad_line_keeps_content_when_already_full(self) -> None:
        """Content at the available width returns unmodified body."""
        from textual.content import Content

        from lilbee.cli.tui.widgets.model_grid import _pad_line

        content = Content("0123456789")
        assert _pad_line(content, 5) is content

    def test_pad_line_pads_with_spaces_when_short(self) -> None:
        """Short content gets right-padded with plain spaces to the requested width."""
        from textual.content import Content

        from lilbee.cli.tui.widgets.model_grid import _pad_line

        out = _pad_line(Content("hi"), 6)
        assert "hi    " in str(out)
        assert out.cell_length == 6

    def test_card_height_matches_body_line_count(self) -> None:
        """``_CARD_HEIGHT`` must equal body line count + reserved border rows.

        Locks the layout invariant: if someone adds a body line in
        ``_local_lines`` without bumping ``_CARD_HEIGHT``, this test fails fast.
        """
        from lilbee.cli.tui.widgets.model_grid import (
            _BORDER_RESERVED_LINES,
            _CARD_HEIGHT,
            _local_lines,
        )

        body = _local_lines(_vgrid_row("phi-3"), selected=True)
        assert len(body) + _BORDER_RESERVED_LINES == _CARD_HEIGHT

    def test_unselected_card_draws_default_border(self) -> None:
        """Every card draws a round border so the grid reads as discrete tiles.

        The unselected card uses ``_DEFAULT_BORDER_STYLE`` (a dim tone); the
        selected card swaps to ``border_style``. Both states emit the same
        box-drawing characters so the layout doesn't shift on focus changes.
        """
        from lilbee.cli.tui.widgets.model_grid import (
            _BORDER_BOTTOM_LEFT,
            _BORDER_HORIZONTAL,
            _BORDER_TOP_LEFT,
            _BORDER_VERTICAL,
            _CARD_HEIGHT,
            _render_card_strip,
        )

        out = _render_card_strip(
            _vgrid_row("phi-3"), selected=False, width=40, border_style=_dummy_border_style()
        )
        assert len(out.lines) == _CARD_HEIGHT
        rendered = [str(line) for line in out.lines]
        assert _BORDER_TOP_LEFT in rendered[0]
        assert _BORDER_HORIZONTAL in rendered[0]
        assert _BORDER_BOTTOM_LEFT in rendered[-1]
        for body_line in rendered[1:-1]:
            assert _BORDER_VERTICAL in body_line

    def test_selected_card_emits_round_border_chars(self) -> None:
        """Focused cards draw the round border with box-drawing characters."""
        from lilbee.cli.tui.widgets.model_grid import (
            _BORDER_BOTTOM_LEFT,
            _BORDER_BOTTOM_RIGHT,
            _BORDER_HORIZONTAL,
            _BORDER_TOP_LEFT,
            _BORDER_TOP_RIGHT,
            _BORDER_VERTICAL,
            _render_card_strip,
        )

        out = _render_card_strip(
            _vgrid_row("phi-3"), selected=True, width=40, border_style=_dummy_border_style()
        )
        rendered = [str(line) for line in out.lines]
        assert _BORDER_TOP_LEFT in rendered[0]
        assert _BORDER_TOP_RIGHT in rendered[0]
        assert _BORDER_HORIZONTAL in rendered[0]
        assert _BORDER_BOTTOM_LEFT in rendered[-1]
        assert _BORDER_BOTTOM_RIGHT in rendered[-1]
        # Body lines have side bars on both edges.
        for body_line in rendered[1:-1]:
            assert _BORDER_VERTICAL in body_line


class TestModelGridBorderStyleSelection:
    """``render_line`` picks ``_FOCUSED_BORDER_STYLE`` vs ``_BLURRED_BORDER_STYLE``
    via ``self.has_focus``; the two strings must resolve to different colors so
    the user can tell which grid owns focus.
    """

    async def test_focused_grid_passes_primary_token_to_card_strip(self) -> None:
        """When the grid has focus, render_line picks the $primary border token."""
        from textual.containers import VerticalScroll

        from lilbee.cli.tui.widgets.model_grid import (
            _FOCUSED_BORDER_STYLE,
            ModelGrid,
        )

        rows = [_vgrid_row(f"m{i}") for i in range(2)]
        grid = ModelGrid(rows, id="mg-focus-token")

        class _GridApp(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield VerticalScroll(grid)

        captured: list[str] = []
        from lilbee.cli.tui.widgets import model_grid as _mg

        original = _mg._render_card_strip

        def _spy(*args, border_style: str, **kwargs):
            captured.append(border_style)
            return original(*args, border_style=border_style, **kwargs)

        _mg._render_card_strip = _spy  # type: ignore[assignment]
        try:
            app = _GridApp()
            async with app.run_test(size=(80, 20)) as pilot:
                await pilot.pause()
                grid.focus()
                await pilot.pause()
                # Force a repaint so render_line runs after focus.
                grid.refresh()
                await pilot.pause()
                assert any(style == _FOCUSED_BORDER_STYLE for style in captured)
        finally:
            _mg._render_card_strip = original  # type: ignore[assignment]

    async def test_blurred_grid_passes_blurred_token_to_card_strip(self) -> None:
        """A grid that does NOT have focus picks the $border-blurred token."""
        from textual.containers import VerticalScroll
        from textual.widgets import Input

        from lilbee.cli.tui.widgets.model_grid import (
            _BLURRED_BORDER_STYLE,
            ModelGrid,
        )

        rows = [_vgrid_row(f"m{i}") for i in range(2)]
        grid = ModelGrid(rows, id="mg-blur-token")
        focus_sink = Input(id="focus-sink")

        class _GridApp(LilbeeAppHost):
            def compose(self) -> ComposeResult:
                yield focus_sink
                yield VerticalScroll(grid)

        captured: list[str] = []
        from lilbee.cli.tui.widgets import model_grid as _mg

        original = _mg._render_card_strip

        def _spy(*args, border_style: str, **kwargs):
            captured.append(border_style)
            return original(*args, border_style=border_style, **kwargs)

        _mg._render_card_strip = _spy  # type: ignore[assignment]
        try:
            app = _GridApp()
            async with app.run_test(size=(80, 20)) as pilot:
                await pilot.pause()
                focus_sink.focus()
                await pilot.pause()
                grid.refresh()
                await pilot.pause()
                assert any(style == _BLURRED_BORDER_STYLE for style in captured)
        finally:
            _mg._render_card_strip = original  # type: ignore[assignment]


class TestCardBodyStyleConstants:
    """The body / focused / blurred string constants must reference $panel."""

    def test_card_body_style_is_panel(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _CARD_BODY_STYLE

        assert "$panel" in _CARD_BODY_STYLE

    def test_focused_border_style_is_primary_on_panel(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _FOCUSED_BORDER_STYLE

        assert "$primary" in _FOCUSED_BORDER_STYLE
        assert "$panel" in _FOCUSED_BORDER_STYLE

    def test_blurred_border_style_is_dim_on_panel(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import _BLURRED_BORDER_STYLE

        assert "$border-blurred" in _BLURRED_BORDER_STYLE
        assert "$panel" in _BLURRED_BORDER_STYLE

    def test_default_border_style_is_dim_on_panel(self) -> None:
        """Every unselected card uses the default dim border tone."""
        from lilbee.cli.tui.widgets.model_grid import _DEFAULT_BORDER_STYLE

        assert "$panel" in _DEFAULT_BORDER_STYLE


class TestModelGridBlurClearsHighlight:
    """Cross-grid focus discipline: blurred grid drops its highlight."""

    def test_on_blur_clears_highlight(self) -> None:
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid = ModelGrid([_vgrid_row("a"), _vgrid_row("b")])
        grid.watch_highlighted = lambda *_a, **_k: None  # type: ignore[method-assign]
        grid.highlighted = 1
        grid.on_blur()
        assert grid.highlighted is None


class TestModelGridGetContentHeight:
    """Catalog screen relies on the height formula to lay out stacked sections."""

    def test_height_uses_row_height_per_grid_row(self) -> None:
        """Content height = grid_rows * _ROW_HEIGHT (no trailing-gutter math)."""
        from lilbee.cli.tui.widgets.model_grid import _ROW_HEIGHT, ModelGrid

        rows = [_vgrid_row(f"m{i}") for i in range(6)]
        grid = ModelGrid(rows)
        size = mock.Mock(width=80, height=24)
        # 6 items / 2 cols = 3 grid rows.
        height = grid.get_content_height(size, size, 80)
        assert height == 3 * _ROW_HEIGHT


class TestCatalogFocusEdgeGuards:
    """LeaveUp/LeaveDown handlers in the catalog screen trap focus at the stack
    edges so the user's cursor doesn't leak to the toolbar / dock."""

    @staticmethod
    async def _catalog_with_grids(rows_a: int, rows_b: int):
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        return app, rows_a, rows_b

    async def test_leave_up_at_first_grid_keeps_focus(self) -> None:
        """Pressing Up at the top row of the topmost grid keeps focus there."""

        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid_a = ModelGrid([_vgrid_row(f"a{i}") for i in range(2)], id="mg-a")
        grid_b = ModelGrid([_vgrid_row(f"b{i}") for i in range(2)], id="mg-b")

        class _StubScreen:
            _grid_container = type(
                "C", (), {"query": staticmethod(lambda _cls: [grid_a, grid_b])}
            )()
            _loading_more = False
            focus_previous_called = False
            focus_next_called = False

            def _active_task_has_more(self) -> bool:
                return False

            def focus_previous(self) -> None:
                self.focus_previous_called = True

            def focus_next(self) -> None:
                self.focus_next_called = True

        screen = _StubScreen()
        # Mimic the screen handler executing on the first grid's LeaveUp.
        event = ModelGrid.LeaveUp(grid_a)
        # Bind the catalog method to the stub; the real method only reads
        # _grid_container, _active_task_has_more, and _loading_more, all of
        # which the stub provides.
        CatalogScreen._on_grid_leave_up(screen, event)  # type: ignore[arg-type]
        assert screen.focus_previous_called is False

        # Sanity: from a non-first grid, focus DOES move.
        event2 = ModelGrid.LeaveUp(grid_b)
        CatalogScreen._on_grid_leave_up(screen, event2)  # type: ignore[arg-type]
        assert screen.focus_previous_called is True

    async def test_leave_down_at_last_grid_with_no_more_keeps_focus(self) -> None:
        """Pressing Down at the last row of the bottom grid (no load_more) parks."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid_a = ModelGrid([_vgrid_row(f"a{i}") for i in range(2)], id="mg-a")
        grid_b = ModelGrid([_vgrid_row(f"b{i}") for i in range(2)], id="mg-b")
        scroll_end_calls: list[bool] = []

        class _StubScreen:
            _grid_container = type(
                "C",
                (),
                {
                    "query": staticmethod(lambda _cls: [grid_a, grid_b]),
                    "scroll_end": lambda self, **_kw: scroll_end_calls.append(True),
                },
            )()
            _loading_more = False
            focus_next_called = False
            load_more_called = False

            def _active_task_has_more(self) -> bool:
                return False

            def focus_next(self) -> None:
                self.focus_next_called = True

            def _load_more(self) -> None:
                self.load_more_called = True

        screen = _StubScreen()
        event = ModelGrid.LeaveDown(grid_b)
        CatalogScreen._on_grid_leave_down(screen, event)  # type: ignore[arg-type]
        assert screen.focus_next_called is False
        assert screen.load_more_called is False
        assert scroll_end_calls, "last-grid LeaveDown must scroll to end to reveal hint"

        # From a non-last grid, focus DOES move (no scroll_end).
        scroll_end_calls.clear()
        event2 = ModelGrid.LeaveDown(grid_a)
        CatalogScreen._on_grid_leave_down(screen, event2)  # type: ignore[arg-type]
        assert screen.focus_next_called is True
        assert not scroll_end_calls

    async def test_leave_down_at_last_grid_with_more_loads_more(self) -> None:
        """Last grid + active task has more pages triggers load_more."""
        from lilbee.cli.tui.screens.catalog import CatalogScreen
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        grid_b = ModelGrid([_vgrid_row(f"b{i}") for i in range(2)], id="mg-b")
        scroll_end_calls: list[bool] = []

        class _StubScreen:
            _grid_container = type(
                "C",
                (),
                {
                    "query": staticmethod(lambda _cls: [grid_b]),
                    "scroll_end": lambda self, **_kw: scroll_end_calls.append(True),
                },
            )()
            _loading_more = False
            focus_next_called = False
            load_more_called = False

            def _active_task_has_more(self) -> bool:
                return True

            def focus_next(self) -> None:
                self.focus_next_called = True

            def _load_more(self) -> None:
                self.load_more_called = True

        screen = _StubScreen()
        event = ModelGrid.LeaveDown(grid_b)
        CatalogScreen._on_grid_leave_down(screen, event)  # type: ignore[arg-type]
        assert screen.load_more_called is True
        assert screen.focus_next_called is False
        assert scroll_end_calls, "last-grid LeaveDown must scroll to end to reveal hint"
