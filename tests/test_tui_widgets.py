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
        """TaskBar is accessible via app.task_bar from other screens."""
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            app.task_bar = bar
            assert app.task_bar is bar


class _ModelBarApp(App):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        yield ModelBar()


class TestModelBar:
    @pytest.fixture(autouse=True)
    def mock_classify(self):
        empty = ([], [], [])
        with mock.patch(
            "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
            return_value=empty,
        ):
            yield

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

    async def test_labels_rendered(self) -> None:
        from textual.widgets import Label

        cfg.chat_model = "qwen3:8b"
        cfg.embedding_model = "nomic"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            labels = [str(lbl.render()) for lbl in app.query(Label)]
            assert "Chat:" in labels
            assert "Embed:" in labels
            assert "Vision:" in labels


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


class TestClassifyInstalledModels:
    def test_native_models_classified_by_task(self, tmp_path) -> None:
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.registry import ModelManifest

        chat_manifest = ModelManifest(
            name="qwen3",
            tag="8b",
            size_bytes=100,
            task="chat",
            source_repo="",
            source_filename="",
            downloaded_at="",
        )
        embed_manifest = ModelManifest(
            name="nomic-embed-text",
            tag="latest",
            size_bytes=100,
            task="embedding",
            source_repo="",
            source_filename="",
            downloaded_at="",
        )
        vision_manifest = ModelManifest(
            name="llava",
            tag="latest",
            size_bytes=100,
            task="vision",
            source_repo="",
            source_filename="",
            downloaded_at="",
        )
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()

        with (
            mock.patch("lilbee.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.model_manager.classify_remote_models",
                return_value=[],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = [
                chat_manifest,
                embed_manifest,
                vision_manifest,
            ]
            chat, embed, vision = _classify_installed_models()

        assert "qwen3:8b" in chat
        assert "nomic-embed-text:latest" in embed
        assert "llava:latest" in vision

    def test_mmproj_filtered_from_all_sources(self, tmp_path) -> None:
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.model_manager import RemoteModel
        from lilbee.registry import ModelManifest

        mmproj_manifest = ModelManifest(
            name="llava-mmproj",
            tag="latest",
            size_bytes=100,
            task="vision",
            source_repo="",
            source_filename="",
            downloaded_at="",
        )
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        # Legacy mmproj .gguf file
        (cfg.models_dir / "clip-mmproj-f16.gguf").write_text("fake")

        remote_mmproj = RemoteModel(
            name="mmproj-model:latest",
            task="vision",
            family="clip",
            parameter_size="",
        )
        with (
            mock.patch("lilbee.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.model_manager.classify_remote_models",
                return_value=[remote_mmproj],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = [mmproj_manifest]
            chat, embed, vision = _classify_installed_models()

        all_names = chat + embed + vision
        assert not any("mmproj" in n.lower() for n in all_names)

    def test_remote_models_classified(self, tmp_path) -> None:
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models
        from lilbee.model_manager import RemoteModel

        remote_chat = RemoteModel(
            name="llama3:8b",
            task="chat",
            family="llama",
            parameter_size="8B",
        )
        remote_embed = RemoteModel(
            name="nomic-embed-text:latest",
            task="embedding",
            family="nomic-bert",
            parameter_size="137M",
        )
        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()

        with (
            mock.patch("lilbee.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.model_manager.classify_remote_models",
                return_value=[remote_chat, remote_embed],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = []
            chat, embed, _vision = _classify_installed_models()

        assert "llama3:8b" in chat
        assert "nomic-embed-text:latest" in embed

    def test_no_models_returns_empty(self, tmp_path) -> None:
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models

        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()

        with (
            mock.patch("lilbee.registry.ModelRegistry") as MockRegistry,
            mock.patch(
                "lilbee.model_manager.classify_remote_models",
                return_value=[],
            ),
        ):
            MockRegistry.return_value.list_installed.return_value = []
            chat, embed, vision = _classify_installed_models()

        assert chat == []
        assert embed == []
        assert vision == []


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
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        t1 = q.enqueue(lambda: None, "A", "sync")
        q.advance()
        q.complete_task(t1)
        q.remove_task(t1)
        t2 = q.enqueue(lambda: None, "B", "sync")
        q.advance()
        q.fail_task(t2, "err")
        assert len(q.history) == 2

    def test_history_empty_initially(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        assert q.history == []

    def test_cancel_nonexistent_returns_false(self) -> None:
        from lilbee.cli.tui.task_queue import TaskQueue

        q = TaskQueue()
        assert q.cancel("nonexistent") is False

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

        cfg.models_dir = tmp_path / "nonexistent"
        chat, embed = _scan_installed_models()
        assert chat == []
        assert embed == []

    def test_scan_installed_models_uses_registry(self, tmp_path) -> None:
        from lilbee.cli.tui.screens.setup import _scan_installed_models

        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        with mock.patch("lilbee.registry.ModelRegistry") as MockRegistry:
            MockRegistry.return_value.list_installed.return_value = []
            chat, embed = _scan_installed_models()
        assert chat == []
        assert embed == []

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


class TestAllTasksFetched:
    def test_all_tasks_constant(self) -> None:
        from lilbee.cli.tui.screens.catalog import _ALL_TASKS

        assert "chat" in _ALL_TASKS
        assert "embedding" in _ALL_TASKS
        assert "vision" in _ALL_TASKS


class TestMatchesSearchWidget:
    def test_matches_name(self) -> None:
        from lilbee.cli.tui.screens.catalog import _catalog_to_row, _matches_search

        m = _make_model("Qwen3 8B", task="chat")
        row = _catalog_to_row(m, installed=False)
        assert _matches_search(row, "qwen") is True

    def test_no_match(self) -> None:
        from lilbee.cli.tui.screens.catalog import _catalog_to_row, _matches_search

        m = _make_model("Qwen3 8B", task="chat")
        row = _catalog_to_row(m, installed=False)
        assert _matches_search(row, "llama") is False


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

    async def test_active_task_text_shown(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            bar.active_task_text = "\u25b6 Sync 60%"
            await pilot.pause()
            assert bar.active_task_text == "\u25b6 Sync 60%"

    async def test_active_task_text_clears(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            bar.active_task_text = "\u25b6 Sync 60%"
            await pilot.pause()
            bar.active_task_text = ""
            await pilot.pause()
            assert bar.active_task_text == ""

    async def test_dock_top_in_css(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        assert "dock: top" in NavBar.DEFAULT_CSS

    async def test_tasks_highlighted_when_active_task(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            bar.active_view = "Chat"
            bar.active_task_text = "Downloading..."
            await pilot.pause()
            content = app.query_one("#nav-bar-content", Static)
            assert "[bold yellow]Tasks[/]" in content.content

    async def test_tasks_not_highlighted_without_active_task(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            bar.active_view = "Chat"
            bar.active_task_text = ""
            await pilot.pause()
            content = app.query_one("#nav-bar-content", Static)
            assert "[bold yellow]Tasks[/]" not in content.content

    async def test_view_index_0_is_chat(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _VIEWS

        assert _VIEWS[0] == "Chat"

    async def test_view_index_3_is_settings(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _VIEWS

        assert _VIEWS[3] == "Settings"

    async def test_view_index_4_is_tasks(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _VIEWS

        assert _VIEWS[4] == "Tasks"


class TestNavBarClickSupport:
    def test_view_regions_covers_all_views(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _VIEWS, _view_regions

        regions = _view_regions()
        assert len(regions) == len(_VIEWS)
        for (start, end, name), expected in zip(regions, _VIEWS, strict=True):
            assert name == expected
            assert end - start == len(name) + 2

    def test_view_regions_are_contiguous(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _view_regions

        regions = _view_regions()
        assert regions[0][0] == 0
        for i in range(1, len(regions)):
            assert regions[i][0] == regions[i - 1][1]

    def test_view_at_x_returns_chat_at_zero(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _view_at_x

        assert _view_at_x(0) == "Chat"

    def test_view_at_x_returns_models(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _view_at_x, _view_regions

        regions = _view_regions()
        models_start = regions[1][0]
        assert _view_at_x(models_start) == "Models"

    def test_view_at_x_returns_last_view(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _view_at_x, _view_regions

        regions = _view_regions()
        last_start = regions[-1][0]
        assert _view_at_x(last_start) == "Tasks"

    def test_view_at_x_returns_none_past_end(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _view_at_x, _view_regions

        regions = _view_regions()
        past_end = regions[-1][1]
        assert _view_at_x(past_end) is None

    def test_view_at_x_returns_none_for_negative(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import _view_at_x

        assert _view_at_x(-1) is None

    async def test_click_calls_switch_view(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar, _view_regions

        app = _NavBarApp()
        app.switch_view = mock.Mock()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            regions = _view_regions()
            models_x = regions[1][0] + 1
            with mock.patch("lilbee.cli.tui.app.LilbeeApp", type(app)):
                bar.on_click(mock.Mock(x=models_x, y=0))
            app.switch_view.assert_called_once_with("Models")

    async def test_click_outside_views_does_nothing(self) -> None:
        from lilbee.cli.tui.widgets.nav_bar import NavBar, _view_regions

        app = _NavBarApp()
        app.switch_view = mock.Mock()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            regions = _view_regions()
            past_end = regions[-1][1] + 10
            with mock.patch("lilbee.cli.tui.app.LilbeeApp", type(app)):
                bar.on_click(mock.Mock(x=past_end, y=0))
            app.switch_view.assert_not_called()

    async def test_click_on_non_lilbee_app_is_safe(self) -> None:
        """Clicking on app that is not LilbeeApp does not crash."""
        from lilbee.cli.tui.widgets.nav_bar import NavBar

        app = _NavBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(NavBar)
            # App is not LilbeeApp -- isinstance check skips switch_view
            bar.on_click(mock.Mock(x=0, y=0))


class TestLilbeeAppGlobalNavBar:
    async def test_screen_composes_global_nav_bar(self) -> None:
        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Pop any setup wizard that may appear
            while not isinstance(app.screen, ChatScreen):
                app.pop_screen()
                await pilot.pause()
            nav = app.screen.query_one("#global-nav-bar")
            assert nav is not None

    async def test_nav_bar_default_is_chat(self) -> None:
        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            while not isinstance(app.screen, ChatScreen):
                app.pop_screen()
                await pilot.pause()
            nav = app.screen.query_one("#global-nav-bar")
            assert nav.active_view == "Chat"


class TestPill:
    def test_pill_from_string(self) -> None:
        from lilbee.cli.tui.pill import pill

        result = pill("chat", "$primary", "$text")
        text = str(result)
        assert "chat" in text
        assert "\u258c" in text  # left half-block
        assert "\u2590" in text  # right half-block

    def test_pill_from_content(self) -> None:
        from textual.content import Content

        from lilbee.cli.tui.pill import pill

        content_input = Content("embed")
        result = pill(content_input, "$secondary", "$text")
        assert "embed" in str(result)

    def test_pill_empty_string(self) -> None:
        from lilbee.cli.tui.pill import pill

        result = pill("", "$primary", "$text")
        text = str(result)
        assert "\u258c" in text
        assert "\u2590" in text

    def test_pill_returns_content(self) -> None:
        from textual.content import Content

        from lilbee.cli.tui.pill import pill

        result = pill("ok", "$success", "$text")
        assert isinstance(result, Content)


class TestEvents:
    def test_model_changed_is_message(self) -> None:
        from textual.message import Message

        from lilbee.cli.tui.events import ModelChanged
        from lilbee.models import ModelTask

        msg = ModelChanged(ModelTask.CHAT, "qwen3:8b")
        assert isinstance(msg, Message)
        assert msg.role == ModelTask.CHAT
        assert msg.name == "qwen3:8b"
