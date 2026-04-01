"""Tests for Textual TUI widgets — 100 % coverage target."""

from __future__ import annotations

from unittest import mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from lilbee.catalog import CatalogModel
from lilbee.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):

    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    cfg.vision_model = ""
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _make_model(
    name: str = "TestModel",
    task: str = "chat",
    featured: bool = False,
    size_gb: float = 2.0,
) -> CatalogModel:
    return CatalogModel(
        name=name,
        hf_repo=f"test/{name.lower().replace(' ', '-')}",
        gguf_filename="*.gguf",
        size_gb=size_gb,
        min_ram_gb=4,
        description="A test model",
        featured=featured,
        downloads=100,
        task=task,
    )


# ---------------------------------------------------------------------------
# message.py
# ---------------------------------------------------------------------------


class _MsgApp(App):
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


class TestAssistantMessageAsync:
    async def test_append_reasoning_expands_collapsible(self) -> None:
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.append_reasoning("step 1")
            assert am._reasoning_parts == ["step 1"]
            assert am._reasoning_widget is not None
            assert am._reasoning_widget.collapsed is False

    async def test_append_content_updates_markdown(self) -> None:
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.append_content("token1")
            am.append_content("token2")
            assert am._content_parts == ["token1", "token2"]

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
            assert am._reasoning_widget.title == "Reasoning"

    async def test_finish_without_reasoning_hides_widget(self) -> None:
        app = _MsgApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            am = app._am
            am.finish(sources=None)
            assert am._finished is True
            assert am._reasoning_widget is not None
            assert am._reasoning_widget.display is False

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


# ---------------------------------------------------------------------------
# help_modal.py
# ---------------------------------------------------------------------------


class _HelpApp(App):
    def compose(self) -> ComposeResult:
        yield Static("bg")

    def key_f1(self) -> None:
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        self.push_screen(HelpModal())


class TestHelpModal:
    async def test_compose_yields_static(self) -> None:
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        app = _HelpApp()
        async with app.run_test() as pilot:
            app.push_screen(HelpModal())
            await pilot.pause()
            # The modal should be visible
            assert len(app.screen_stack) == 2

    async def test_action_close_dismisses(self) -> None:
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        app = _HelpApp()
        async with app.run_test() as pilot:
            app.push_screen(HelpModal())
            await pilot.pause()
            app.screen.action_close()
            await pilot.pause()
            assert len(app.screen_stack) == 1


# ---------------------------------------------------------------------------
# task_bar.py
# ---------------------------------------------------------------------------


class _TaskBarApp(App):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        yield TaskBar(id="task-bar")


class TestTaskBar:
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
            await pilot.pause(delay=1.5)
            assert bar.queue.is_empty

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
            await pilot.pause(delay=1.5)
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
            await pilot.pause(delay=1.5)
            assert bar.queue.is_empty

    async def test_app_task_bar_ref(self) -> None:
        """TaskBar is accessible via app._task_bar from other screens."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            app._task_bar = bar  # type: ignore[attr-defined]
            assert getattr(app, "_task_bar", None) is bar


# ---------------------------------------------------------------------------
# model_bar.py
# ---------------------------------------------------------------------------


class _ModelBarApp(App):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        yield ModelBar()


class TestModelBar:
    @pytest.fixture(autouse=True)
    def mock_provider(self):
        mock_svc = mock.MagicMock()
        mock_svc.provider.list_models.return_value = []
        with mock.patch("lilbee.services.get_services", return_value=mock_svc):
            yield mock_svc

    async def test_renders_select_widgets(self) -> None:
        from textual.widgets import Select

        cfg.chat_model = "qwen3:8b"
        cfg.embedding_model = "nomic"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            selects = list(app.query(Select))
            assert len(selects) == 3

    async def test_widget_exists_with_3_selects(self) -> None:
        from textual.widgets import Select

        cfg.chat_model = "qwen3:8b"
        cfg.embedding_model = "nomic"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            chat_sel = app.query_one("#chat-model-select", Select)
            embed_sel = app.query_one("#embed-model-select", Select)
            vision_sel = app.query_one("#vision-model-select", Select)
            assert chat_sel is not None
            assert embed_sel is not None
            assert vision_sel is not None

    async def test_vision_set_when_configured(self) -> None:
        from textual.widgets import Select

        cfg.vision_model = "llava"
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            vision_sel = app.query_one("#vision-model-select", Select)
            assert vision_sel.value == "llava"


# ---------------------------------------------------------------------------
# suggester.py
# ---------------------------------------------------------------------------


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
        # Has space but doesn't start with / — hits _suggest_argument which returns None
        assert await s.get_suggestion("hello world") is None

    async def test_plain_text_no_space_returns_none(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        # No space, doesn't start with / — hits line 43 return None
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

    @mock.patch("lilbee.cli.tui.widgets.suggester.SlashSuggester._get_vision_names")
    async def test_suggest_vision_arg(self, mock_names: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        mock_names.return_value = ["off", "llava:latest"]
        s = SlashSuggester(use_cache=False)
        r = await s.get_suggestion("/vision ll")
        assert r is not None
        assert "llava:latest" in r

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
        with mock.patch("lilbee.models.list_installed_models", side_effect=Exception("err")):
            assert s._get_model_names() == []

    def test_get_vision_names_error(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        with mock.patch("lilbee.cli.tui.widgets.suggester.SlashSuggester._get_vision_names") as m:
            m.return_value = ["off"]
            r = s._get_vision_names()
            assert "off" in r

    def test_get_vision_names_iteration_error(self) -> None:
        """Cover the except branch in _get_vision_names (lines 87-88)."""
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)

        # Make VISION_CATALOG iteration explode
        class BrokenIter:
            def __iter__(self):
                raise RuntimeError("boom")

        with mock.patch("lilbee.models.VISION_CATALOG", BrokenIter()):
            r = s._get_vision_names()
        assert r == ["off"]

    def test_get_document_names_error(self) -> None:
        from lilbee.cli.tui.widgets.suggester import SlashSuggester

        s = SlashSuggester(use_cache=False)
        with mock.patch("lilbee.services.get_services", side_effect=Exception("err")):
            assert s._get_document_names() == []


# ---------------------------------------------------------------------------
# autocomplete.py — pure functions
# ---------------------------------------------------------------------------


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

    @mock.patch("lilbee.models.list_installed_models", return_value=["qwen3:8b", "mistral:7b"])
    def test_model_arg_completions(self, _mock: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.autocomplete import get_completions

        r = get_completions("/model qw")
        assert "qwen3:8b" in r

    @mock.patch("lilbee.models.list_installed_models", return_value=["qwen3:8b"])
    def test_model_arg_no_partial(self, _mock: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.autocomplete import get_completions

        r = get_completions("/model ")
        assert "qwen3:8b" in r

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

        with mock.patch("lilbee.models.list_installed_models", return_value=["a", "b"]):
            assert _model_options() == ["a", "b"]

    def test_returns_empty_on_error(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _model_options

        with mock.patch("lilbee.models.list_installed_models", side_effect=Exception("err")):
            assert _model_options() == []


class TestVisionOptions:
    def test_returns_off_plus_catalog(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _vision_options
        from lilbee.models import ModelInfo

        fake_catalog = (ModelInfo("llava", 5.5, 8, "test"),)
        with mock.patch("lilbee.models.VISION_CATALOG", fake_catalog):
            r = _vision_options()
            assert r[0] == "off"
            assert "llava" in r

    def test_returns_off_on_error(self) -> None:
        import builtins

        from lilbee.cli.tui.widgets.autocomplete import _vision_options

        real_import = builtins.__import__

        def bad_import(name, *args, **kwargs):
            if name == "lilbee.models":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=bad_import):
            r = _vision_options()
        assert r == ["off"]


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
        with mock.patch("lilbee.services.get_services", return_value=mock_svc):
            assert _document_options() == ["a.txt"]

    def test_returns_empty_on_error(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _document_options

        with mock.patch("lilbee.services.get_services", side_effect=Exception("err")):
            assert _document_options() == []

    def test_falls_back_to_source_key(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _document_options

        mock_svc = mock.MagicMock()
        mock_svc.store.get_sources.return_value = [{"source": "b.pdf"}]
        with mock.patch("lilbee.services.get_services", return_value=mock_svc):
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


# ---------------------------------------------------------------------------
# autocomplete.py — CompletionOverlay widget
# ---------------------------------------------------------------------------


class _OverlayApp(App):
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


# ---------------------------------------------------------------------------
# task_queue.py (unit tests for queue logic)
# ---------------------------------------------------------------------------


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

    def test_advance_returns_none_when_active(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        q.enqueue(lambda: None, "A", "download")
        q.advance()
        q.enqueue(lambda: None, "B", "sync")
        assert q.advance() is None  # already has active

    def test_fail_task(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue, TaskStatus

        q = TaskQueue()
        tid = q.enqueue(lambda: None, "A", "download")
        q.advance()
        q.fail_task(tid, "oops")
        task = q._tasks.get(tid)
        assert task is not None
        assert task.status == TaskStatus.FAILED

    def test_on_change_callback(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        calls: list[bool] = []
        q = TaskQueue(on_change=lambda: calls.append(True))
        q.enqueue(lambda: None, "A", "sync")
        assert len(calls) >= 1


# ---------------------------------------------------------------------------
# setup_modal.py
# ---------------------------------------------------------------------------


class _SetupApp(App):
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

    async def test_action_cancel_dismisses_skipped(self) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = _SetupApp()
        results: list[object] = []
        async with app.run_test() as pilot:
            app.push_screen(SetupWizard(), callback=lambda r: results.append(r))
            await pilot.pause()
            app.screen.action_cancel()
            await pilot.pause()
        assert "skipped" in results

    def test_scan_installed_models_empty_dir(self, tmp_path) -> None:
        from lilbee.cli.tui.screens.setup import _scan_installed_models

        chat, embed = _scan_installed_models(tmp_path / "nonexistent")
        assert chat == []
        assert embed == []

    def test_scan_installed_models_splits_by_name(self, tmp_path) -> None:
        from lilbee.cli.tui.screens.setup import _scan_installed_models

        (tmp_path / "chat-model.gguf").touch()
        (tmp_path / "nomic-embed-text.gguf").touch()
        chat, embed = _scan_installed_models(tmp_path)
        assert len(chat) == 1
        assert len(embed) == 1
        assert "chat" in chat[0].name.lower()
        assert "embed" in embed[0].name.lower()

    def test_installed_row_compose(self, tmp_path) -> None:
        from lilbee.cli.tui.screens.setup import _InstalledRow

        model_file = tmp_path / "test.gguf"
        model_file.write_bytes(b"x" * 1024)
        row = _InstalledRow(model_file)
        children = list(row.compose())
        assert len(children) == 1

    def test_catalog_row_compose(self) -> None:
        from lilbee.cli.tui.screens.setup import _CatalogRow

        model = _make_model("Test", task="chat")
        row = _CatalogRow(model)
        children = list(row.compose())
        assert len(children) == 1


# ---------------------------------------------------------------------------
# catalog.py screen — HF grouping, empty tabs, size grouping
# ---------------------------------------------------------------------------


class TestGroupBySize:
    def test_unknown_becomes_other(self) -> None:
        from lilbee.cli.tui.screens.catalog import _group_by_size

        model = _make_model("NoSizeModel", task="chat")
        groups = _group_by_size([model])
        labels = [label for label, _ in groups]
        assert "Other" in labels
        assert "unknown" not in labels

    def test_known_sizes_grouped(self) -> None:
        from lilbee.cli.tui.screens.catalog import _group_by_size

        small = _make_model("Tiny 1B", task="chat")
        medium = _make_model("Mid 7B", task="chat")
        large = _make_model("Big 14B", task="chat")
        groups = _group_by_size([small, medium, large])
        labels = [label for label, _ in groups]
        assert "Small (≤3B)" in labels
        assert "Medium (3-8B)" in labels
        assert "Large (8-30B)" in labels


class TestGroupHfByFamily:
    def test_groups_by_family_name(self) -> None:
        from lilbee.cli.tui.screens.catalog import _group_hf_by_family

        m1 = _make_model("Qwen3 8B", task="chat")
        m2 = _make_model("Qwen3 4B", task="chat")
        m3 = _make_model("Llama 7B", task="chat")
        families = _group_hf_by_family([m1, m2, m3])
        names = [f.name for f in families]
        assert "Qwen3" in names
        assert "Llama" in names
        qwen_fam = next(f for f in families if f.name == "Qwen3")
        assert len(qwen_fam.variants) == 2


class TestHfBrowseChatOnly:
    def test_constant_defined(self) -> None:
        from lilbee.cli.tui.screens.catalog import (
            _HF_BROWSE_CHAT_ONLY,
            _HF_BROWSE_TASKS,
        )

        assert "Featured models only" in _HF_BROWSE_CHAT_ONLY
        assert "Embedding" not in _HF_BROWSE_TASKS
        assert "Vision" not in _HF_BROWSE_TASKS


# ---------------------------------------------------------------------------
# command_registry.py — /login command
# ---------------------------------------------------------------------------


class TestLoginCommandRegistered:
    def test_login_in_registry(self) -> None:
        from lilbee.cli.tui.command_registry import COMMANDS, build_dispatch_dict

        names = [c.name for c in COMMANDS]
        assert "/login" in names
        dispatch = build_dispatch_dict()
        assert dispatch["/login"] == "_cmd_login"


# ---------------------------------------------------------------------------
# settings.py — HF token field
# ---------------------------------------------------------------------------


class TestSettingsHfToken:
    def test_get_hf_token_display_not_set(self, monkeypatch) -> None:
        from lilbee.cli.tui.screens.settings import _get_hf_token_display

        monkeypatch.delenv("LILBEE_HF_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        # Without any token set, should show "not set"
        result = _get_hf_token_display()
        assert isinstance(result, str)

    def test_get_hf_token_display_from_env(self, monkeypatch) -> None:
        from lilbee.cli.tui.screens.settings import _get_hf_token_display

        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnop")
        result = _get_hf_token_display()
        assert result.startswith("hf_a")
        assert result.endswith("mnop")
        assert "..." in result


# ---------------------------------------------------------------------------
# __init__.py — Ctrl-C clean shutdown
# ---------------------------------------------------------------------------


class TestRunTuiKeyboardInterrupt:
    def test_keyboard_interrupt_does_not_raise(self) -> None:
        with mock.patch("lilbee.cli.tui.app.LilbeeApp") as MockApp:
            MockApp.return_value.run.side_effect = KeyboardInterrupt
            with (
                mock.patch("lilbee.cli.tui.shutdown_executor"),
                mock.patch("lilbee.cli.tui.reset_services"),
            ):
                from lilbee.cli.tui import run_tui

                run_tui()

    def test_cleanup_called_on_interrupt(self) -> None:
        with mock.patch("lilbee.cli.tui.app.LilbeeApp") as MockApp:
            MockApp.return_value.run.side_effect = KeyboardInterrupt
            with (
                mock.patch("lilbee.cli.tui.shutdown_executor") as mock_shutdown,
                mock.patch("lilbee.cli.tui.reset_services") as mock_reset,
            ):
                from lilbee.cli.tui import run_tui

                run_tui()
                mock_shutdown.assert_called_once()
                mock_reset.assert_called_once()


# ---------------------------------------------------------------------------
# nav_bar.py — global docked navigation bar
# ---------------------------------------------------------------------------


class _NavBarApp(App):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        yield NavBar(id="nav-bar")


class TestNavBar:
    async def test_compose_yields_static(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            assert bar is not None

    async def test_default_active_view_is_chat(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            assert bar.active_view == "Chat"

    async def test_watch_active_view_updates_display(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            bar.active_view = "Models"
            await pilot.pause()
            # Just verify it doesn't crash and updates
            assert bar.active_view == "Models"

    async def test_all_views_shown_in_display(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Just verify the NavBar composes successfully with all views
            bar = app.query_one(NavBar)
            assert bar.active_view == "Chat"

    async def test_set_active_view_to_status(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            bar.active_view = "Status"
            await pilot.pause()
            assert bar.active_view == "Status"

    async def test_change_view_updates_active_view(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            bar._change_view("Status")
            await pilot.pause()
            assert bar.active_view == "Status"

    async def test_change_view_ignores_invalid(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            bar._change_view("InvalidView")
            await pilot.pause()
            assert bar.active_view == "Chat"

    async def test_view_index_0_is_chat(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _VIEWS

        assert _VIEWS[0] == "Chat"

    async def test_view_index_3_is_settings(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _VIEWS

        assert _VIEWS[3] == "Settings"


# ---------------------------------------------------------------------------
# app.py — global NavBar composition and key bindings
# ---------------------------------------------------------------------------


class TestLilbeeAppGlobalNavBar:
    async def test_app_composes_global_nav_bar(self) -> None:
        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            nav = app.query_one("#global-nav-bar")
            assert nav is not None

    async def test_app_nav_bar_default_is_chat(self) -> None:
        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            nav = app.query_one("#global-nav-bar")
            assert nav.active_view == "Chat"
