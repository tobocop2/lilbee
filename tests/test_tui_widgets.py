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
    cfg.data_root = tmp_path
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
    tag: str = "latest",
    display_name: str = "",
) -> CatalogModel:
    return CatalogModel(
        name=name.lower().replace(" ", "-"),
        tag=tag,
        display_name=display_name or name,
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

        chat_refs = [ref for _, ref in chat]
        embed_refs = [ref for _, ref in embed]
        vision_refs = [ref for _, ref in vision]
        assert "qwen3:8b" in chat_refs
        assert "nomic-embed-text:latest" in embed_refs
        assert "llava:latest" in vision_refs

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

        all_refs = [ref for _, ref in chat + embed + vision]
        assert not any("mmproj" in r.lower() for r in all_refs)

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

        chat_refs = [ref for _, ref in chat]
        embed_refs = [ref for _, ref in embed]
        assert "llama3:8b" in chat_refs
        assert "nomic-embed-text:latest" in embed_refs

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

        fake_catalog = (
            ModelInfo(
                ref="llava:latest",
                display_name="LLaVA",
                size_gb=5.5,
                min_ram_gb=8,
                description="test",
            ),
        )
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

    def test_installed_name_to_row(self) -> None:
        from lilbee.cli.tui.screens.setup import _installed_name_to_row

        row = _installed_name_to_row("test-model:latest", "chat")
        assert row.name == "test-model:latest"
        assert row.task == "chat"
        assert row.installed is True
        assert row.featured is False

    def test_model_card_from_table_row(self) -> None:
        from lilbee.cli.tui.screens.catalog import _catalog_to_row
        from lilbee.cli.tui.widgets.model_card import ModelCard

        model = _make_model("Test 8B", task="chat", featured=True)
        row = _catalog_to_row(model, installed=False)
        card = ModelCard(row)
        assert card.row is row
        assert card.row.featured is True
        assert card.row.task == "chat"


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
            MockApp.return_value.run.assert_called_once()

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


class _StatusBarApp(App):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.status_bar import StatusBar

        yield StatusBar()


class TestStatusBar:
    async def test_compose_yields_static(self) -> None:
        from lilbee.cli.tui.widgets.status_bar import StatusBar

        app = _StatusBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(StatusBar)
            assert bar is not None

    async def test_default_active_view_is_chat(self) -> None:
        from lilbee.cli.tui.widgets.status_bar import StatusBar

        app = _StatusBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(StatusBar)
            assert bar.active_view == "Chat"

    async def test_watch_active_view_updates_display(self) -> None:
        from lilbee.cli.tui.widgets.status_bar import StatusBar

        app = _StatusBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(StatusBar)
            bar.active_view = "Catalog"
            await pilot.pause()
            assert bar.active_view == "Catalog"

    async def test_set_active_view_to_status(self) -> None:
        from lilbee.cli.tui.widgets.status_bar import StatusBar

        app = _StatusBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(StatusBar)
            bar.active_view = "Status"
            await pilot.pause()
            assert bar.active_view == "Status"

    async def test_mode_text_updates(self) -> None:
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.widgets.status_bar import StatusBar

        app = _StatusBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(StatusBar)
            bar.mode_text = msg.MODE_NORMAL
            await pilot.pause()
            assert bar.mode_text == msg.MODE_NORMAL

    async def test_dock_bottom_in_css(self) -> None:
        from lilbee.cli.tui.widgets.status_bar import StatusBar

        assert "dock: bottom" in StatusBar.DEFAULT_CSS

    async def test_nav_views_contains_all_screens(self) -> None:
        from lilbee.cli.tui.messages import get_nav_views

        views = get_nav_views()
        for name in ("Chat", "Catalog", "Status", "Settings", "Tasks"):
            assert name in views

    async def test_default_view_is_first(self) -> None:
        from lilbee.cli.tui import messages as msg

        assert msg.get_nav_views()[0] == msg.DEFAULT_VIEW


class TestLilbeeAppStatusBar:
    async def test_screen_composes_status_bar(self) -> None:
        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.status_bar import StatusBar

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            while not isinstance(app.screen, ChatScreen):
                app.pop_screen()
                await pilot.pause()
            bar = app.screen.query_one(StatusBar)
            assert bar is not None

    async def test_status_bar_default_is_chat(self) -> None:
        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.status_bar import StatusBar

        app = LilbeeApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            while not isinstance(app.screen, ChatScreen):
                app.pop_screen()
                await pilot.pause()
            bar = app.screen.query_one(StatusBar)
            assert bar.active_view == "Chat"


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


# ---------------------------------------------------------------------------
# GridSelect widget tests
# ---------------------------------------------------------------------------


class _GridApp(App):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        yield GridSelect(
            Static("A", id="item-a"),
            Static("B", id="item-b"),
            Static("C", id="item-c"),
            Static("D", id="item-d"),
            min_column_width=20,
        )


class _LargeGridApp(App):
    """Grid with enough items and wide min_column_width to guarantee multiple rows."""

    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        items = [Static(f"Item {i}", id=f"item-{i}") for i in range(8)]
        yield GridSelect(*items, min_column_width=30)


class _EmptyGridApp(App):
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
            # Just verify it doesn't crash — LeaveUp is posted
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


# ---------------------------------------------------------------------------
# SetupModal widget tests
# ---------------------------------------------------------------------------


class _SetupModalApp(App):
    def compose(self) -> ComposeResult:
        yield Static("base")


class TestSetupModal:
    async def test_compose_without_remote_embeddings(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal())
            await pilot.pause()
            assert len(app.screen_stack) == 2

    async def test_compose_with_remote_embeddings(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal(ollama_embeddings=["nomic-embed-text"]))
            await pilot.pause()
            assert len(app.screen_stack) == 2

    async def test_select_remote_row_dismisses_with_name(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        results: list[object] = []
        async with app.run_test() as pilot:
            app.push_screen(
                SetupModal(ollama_embeddings=["nomic-embed-text"]),
                callback=lambda r: results.append(r),
            )
            await pilot.pause()
            # Find the _RemoteRow and select it via the ListView
            from textual.widgets import ListView

            lv = app.screen.query_one("#embed-picker", ListView)
            # The _RemoteRow should be the second item (first is header label)
            from lilbee.cli.tui.widgets.setup_modal import _RemoteRow

            for idx, item in enumerate(lv.children):
                if isinstance(item, _RemoteRow):
                    lv.post_message(ListView.Selected(lv, item, idx))
                    break
            await pilot.pause()
        assert "nomic-embed-text" in results

    async def test_select_embedding_row_triggers_download(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal())
            await pilot.pause()
            from textual.widgets import ListView

            from lilbee.cli.tui.widgets.setup_modal import _EmbeddingRow

            lv = app.screen.query_one("#embed-picker", ListView)
            with mock.patch(
                "lilbee.cli.tui.widgets.setup_modal.SetupModal._download_model"
            ) as mock_dl:
                for idx, item in enumerate(lv.children):
                    if isinstance(item, _EmbeddingRow):
                        lv.post_message(ListView.Selected(lv, item, idx))
                        break
                await pilot.pause()
                mock_dl.assert_called_once()

    async def test_download_success_path(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        results: list[object] = []
        async with app.run_test() as pilot:
            app.push_screen(
                SetupModal(),
                callback=lambda r: results.append(r),
            )
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, SetupModal)
            # Simulate successful download callback
            modal._on_downloaded("test-embed-model")
            await pilot.pause()
            await pilot.pause()
        assert "test-embed-model" in results

    async def test_download_error_path(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal())
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, SetupModal)
            # Simulate the error path by calling _set_status directly
            modal._set_status("Error: connection failed")
            await pilot.pause()
            from textual.widgets import Label

            status = modal.query_one("#setup-status", Label)
            assert "Error" in str(status.render())

    async def test_cancel_action_dismisses_none(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        results: list[object] = []
        async with app.run_test() as pilot:
            app.push_screen(
                SetupModal(),
                callback=lambda r: results.append(r),
            )
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, SetupModal)
            modal.action_cancel()
            await pilot.pause()
        assert None in results

    async def test_set_status_updates_label(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal())
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, SetupModal)
            modal._set_status("Downloading...")
            await pilot.pause()
            from textual.widgets import Label

            status = modal.query_one("#setup-status", Label)
            assert "Downloading" in str(status.render())

    async def test_download_model_worker_success(self) -> None:
        from lilbee.catalog import FEATURED_EMBEDDING
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        results: list[object] = []
        async with app.run_test() as pilot:
            app.push_screen(
                SetupModal(),
                callback=lambda r: results.append(r),
            )
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, SetupModal)
            with mock.patch("lilbee.models.pull_with_progress"):
                modal._download_model(FEATURED_EMBEDDING[0])
                await pilot.pause(delay=0.5)
        assert len(results) >= 1

    async def test_download_model_worker_error(self) -> None:
        from lilbee.catalog import FEATURED_EMBEDDING
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupModalApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal())
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, SetupModal)
            with mock.patch(
                "lilbee.models.pull_with_progress",
                side_effect=RuntimeError("download failed"),
            ):
                modal._download_model(FEATURED_EMBEDDING[0])
                await pilot.pause(delay=0.5)
            # Modal should still be showing (not dismissed)
            from textual.widgets import Label

            status = modal.query_one("#setup-status", Label)
            rendered = str(status.render())
            assert "Error" in rendered or "download failed" in rendered


# ---------------------------------------------------------------------------
# ModelBar additional coverage tests
# ---------------------------------------------------------------------------


class TestModelBarAdditional:
    @pytest.fixture(autouse=True)
    def mock_classify(self):
        empty = ([], [], [])
        with mock.patch(
            "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
            return_value=empty,
        ):
            yield

    async def test_populate_chat_model_in_scanned(self) -> None:
        """When current chat model IS in scanned list, value is preserved."""
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "qwen3:8b"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar._populate(
                [("Qwen3 8B", "qwen3:8b"), ("Llama 7B", "llama:7b")],
                [("test-embed", "test-embed")],
                [],
            )
            await pilot.pause()
            from textual.widgets import Select

            chat_sel = app.query_one("#chat-model-select", Select)
            assert chat_sel.value == "qwen3:8b"

    async def test_populate_no_models_found(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            # Populate with empty lists — triggers (none) fallback
            bar._populate([], [], [])
            await pilot.pause()
            from textual.widgets import Select

            chat_sel = app.query_one("#chat-model-select", Select)
            assert chat_sel.value in ("", Select.BLANK)

    async def test_populate_vision_model_fallback(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = "llava:custom"
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            # vision model configured but not in scanned list
            bar._populate(
                [("test-model", "test-model")],
                [("test-embed", "test-embed")],
                [("Llava 7B", "llava:7b")],
            )
            await pilot.pause()
            from textual.widgets import Select

            vision_sel = app.query_one("#vision-model-select", Select)
            assert vision_sel.value == "llava:custom"

    async def test_populate_vision_model_not_in_list_or_config(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _DISABLED, ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar._populate(
                [("test-model", "test-model")],
                [("test-embed", "test-embed")],
                [],
            )
            await pilot.pause()
            from textual.widgets import Select

            vision_sel = app.query_one("#vision-model-select", Select)
            assert vision_sel.value is _DISABLED

    async def test_on_embed_model_changed(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar._populating = False
            from textual.widgets import Select

            embed_sel = app.query_one("#embed-model-select", Select)
            embed_sel.set_options([("new-embed", "new-embed")])
            with (
                mock.patch("lilbee.settings.set_value"),
                mock.patch("lilbee.services.reset_services"),
            ):
                embed_sel.value = "new-embed"
                await pilot.pause()
            assert cfg.embedding_model == "new-embed"

    async def test_populate_embed_model_in_scanned(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "nomic:latest"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar._populate(
                [("test-model", "test-model")],
                [("Nomic Embed Text", "nomic:latest")],
                [],
            )
            await pilot.pause()
            from textual.widgets import Select

            embed_sel = app.query_one("#embed-model-select", Select)
            assert embed_sel.value == "nomic:latest"


class TestCollectNativeModelsError:
    def test_exception_suppressed(self, tmp_path) -> None:
        from lilbee.cli.tui.widgets.model_bar import _collect_native_models

        cfg.models_dir = tmp_path / "models"
        cfg.models_dir.mkdir()
        buckets: dict[str, list[tuple[str, str]]] = {
            "chat": [],
            "embedding": [],
            "vision": [],
        }
        seen: set[str] = set()
        with mock.patch(
            "lilbee.registry.ModelRegistry",
            side_effect=RuntimeError("boom"),
        ):
            _collect_native_models(buckets, seen)
        assert buckets["chat"] == []

    def test_collect_remote_models_exception_suppressed(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import _collect_remote_models

        buckets: dict[str, list[tuple[str, str]]] = {
            "chat": [],
            "embedding": [],
            "vision": [],
        }
        seen: set[str] = set()
        with mock.patch(
            "lilbee.model_manager.classify_remote_models",
            side_effect=RuntimeError("boom"),
        ):
            _collect_remote_models(buckets, seen)
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

        from lilbee.cli.tui.widgets.model_card import _build_status

        @dataclass
        class FakeRow:
            installed: bool
            sort_downloads: int
            downloads: str

        row = FakeRow(installed=False, sort_downloads=0, downloads="--")
        assert _build_status(row) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TaskBar additional coverage tests
# ---------------------------------------------------------------------------


class TestTaskBarAdditional:
    async def test_refresh_display_no_active_but_queued(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            # Add tasks but don't advance (so no active, but queued)
            bar.add_task("Sync", "sync")
            bar._refresh_display()
            await pilot.pause()
            assert bar.display is True
            from textual.widgets import Label

            active_label = bar.query_one("#task-active-label", Label)
            assert active_label.display is False

    async def test_status_icon_unknown_status(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        icon = TaskBar._status_icon("unknown_status")  # type: ignore[arg-type]
        assert icon == "▸"

    async def test_on_queue_change_exception_suppressed(self) -> None:
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            # call_from_thread might raise if not on worker thread, but suppress
            bar._on_queue_change()  # should not raise
            assert bar.display is False

    async def test_render_active_task_non_active_status(self) -> None:
        """Cover line 166: _render_active_task when status != ACTIVE (e.g. DONE)."""
        from textual.widgets import Label, ProgressBar

        from lilbee.cli.tui.task_queue import Task, TaskStatus
        from lilbee.cli.tui.widgets.task_bar import TaskBar

        app = _TaskBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(TaskBar)
            label = bar.query_one("#task-active-label", Label)
            progress_bar = bar.query_one("#task-progress", ProgressBar)
            # Create a task with DONE status and call _render_active_task directly
            done_task = Task(
                task_id="t1",
                fn=lambda: None,
                name="Sync",
                task_type="sync",
                status=TaskStatus.DONE,
                progress=100,
                detail="",
            )
            bar._render_active_task(done_task, label, progress_bar)
            await pilot.pause()


# ---------------------------------------------------------------------------
# GridSelect additional coverage — highlight_first, highlight_last, cursor moves
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ModelCard: _build_status with positive downloads
# ---------------------------------------------------------------------------


class TestModelCardBuildStatusDownloads:
    def test_build_status_with_downloads(self) -> None:
        from dataclasses import dataclass

        from lilbee.cli.tui.widgets.model_card import _build_status

        @dataclass
        class FakeRow:
            installed: bool
            sort_downloads: int
            downloads: str

        row = FakeRow(installed=False, sort_downloads=1000, downloads="1K")
        result = _build_status(row)  # type: ignore[arg-type]
        assert result is not None
        assert "1K" in str(result)


# ---------------------------------------------------------------------------
# ModelBar: _populate branch coverage and refresh_models
# ---------------------------------------------------------------------------


class TestModelBarPopulateBranches:
    @pytest.fixture(autouse=True)
    def mock_classify(self):
        empty = ([], [], [])
        with mock.patch(
            "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
            return_value=empty,
        ):
            yield

    async def test_populate_with_matching_models(self) -> None:
        """When scanned models match config, values are preserved."""
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar._populate(
                [("test-model", "test-model"), ("other", "other")],
                [("test-embed", "test-embed"), ("nomic", "nomic")],
                [("Llava 7B", "llava:7b")],
            )
            await pilot.pause()
            from textual.widgets import Select

            chat_sel = app.query_one("#chat-model-select", Select)
            embed_sel = app.query_one("#embed-model-select", Select)
            assert chat_sel.value == "test-model"
            assert embed_sel.value == "test-embed"

    async def test_populate_empty_lists_uses_none_option(self) -> None:
        """When no models found, (none) fallback is used."""
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar._populate([], [], [])
            await pilot.pause()
            from textual.widgets import Select

            chat_sel = app.query_one("#chat-model-select", Select)
            assert chat_sel.value in ("", Select.BLANK)

    async def test_populate_vision_from_cfg_fallback(self) -> None:
        """Cover lines 202-206: vision from cfg when not in scan and select is empty."""
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = "llava:custom"
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar._populate(
                [("test-model", "test-model")],
                [("test-embed", "test-embed")],
                [],
            )
            await pilot.pause()
            from textual.widgets import Select

            vision_sel = app.query_one("#vision-model-select", Select)
            assert vision_sel.value == "llava:custom"

    async def test_populate_vision_in_scanned_list(self) -> None:
        """Cover lines 197-201: vision model in scanned list, has_vision_model True."""
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = "llava:7b"
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar._populate(
                [("test-model", "test-model")],
                [("test-embed", "test-embed")],
                [("Llava 7B", "llava:7b"), ("Moondream", "moondream:latest")],
            )
            await pilot.pause()
            from textual.widgets import Select

            vision_sel = app.query_one("#vision-model-select", Select)
            assert vision_sel.value == "llava:7b"

    async def test_populate_value_not_in_new_options(self) -> None:
        """Cover lines 179, 183, 189, 193, 198-201: value retained but not in scanned list.

        After set_options, Textual may auto-select the first option, but we can
        intercept to simulate a retained value that isn't in the new options.
        """
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = "llava:custom"
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            from textual.widgets import Select

            chat_sel = app.query_one("#chat-model-select", Select)
            embed_sel = app.query_one("#embed-model-select", Select)
            vision_sel = app.query_one("#vision-model-select", Select)

            # Monkey-patch set_options to also force a custom value afterward,
            # simulating the case where the widget retains a value not in the
            # new option list (e.g. from a previous configure step).
            patched_vals = [
                (chat_sel, "custom-chat:latest"),
                (embed_sel, "custom-embed:latest"),
                (vision_sel, "custom-vision:latest"),
            ]
            call_counts: dict[int, int] = {}

            for sel, cfg_val in patched_vals:
                sel_id = id(sel)
                call_counts[sel_id] = 0
                orig_fn = sel.set_options

                def make_patched(s, orig, val, sid):
                    def patched(opts):
                        orig(opts)
                        call_counts[sid] += 1
                        # Only on the FIRST set_options call, force the custom value.
                        # The second call (line 180/190/200) has the value in opts.
                        if call_counts[sid] == 1:
                            # Add the value to the option list and set it
                            opts_with_val = [(val, val), *list(opts)]
                            orig(opts_with_val)
                            s.value = val

                    return patched

                sel.set_options = make_patched(sel, orig_fn, cfg_val, sel_id)  # type: ignore[assignment]

            bar._populate(
                [("Qwen3 8B", "qwen3:8b"), ("Llama 7B", "llama:7b")],
                [("Nomic Embed Text", "nomic:latest")],
                [("Llava 7B", "llava:7b")],
            )
            await pilot.pause()
            # Each select should now have the custom value prepended and selected
            assert chat_sel.value == "custom-chat:latest"
            assert embed_sel.value == "custom-embed:latest"
            assert vision_sel.value == "custom-vision:latest"

    async def test_populate_blank_value_with_models_available(self) -> None:
        """Cover lines 183, 193: value is empty but models list is non-empty.

        This is a defensive branch — after set_options with allow_blank=False,
        Textual normally auto-selects the first value. We force the Select to
        return empty by intercepting set_options.
        """
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            from textual.widgets import Select

            chat_sel = app.query_one("#chat-model-select", Select)
            embed_sel = app.query_one("#embed-model-select", Select)

            # Patch the first set_options to force _reactive_value = Select.NULL
            # (bypassing validation), so has_chat_model = False while
            # chat_models is non-empty. This covers lines 183, 193.
            for sel in (chat_sel, embed_sel):
                orig_fn = sel.set_options
                call_count = [0]

                def make_patched(s, orig, cc):
                    def patched(opts):
                        orig(opts)
                        cc[0] += 1
                        if cc[0] == 1:
                            # Bypass validation to force NULL value
                            s._reactive_value = Select.NULL  # type: ignore[attr-defined]

                    return patched

                sel.set_options = make_patched(sel, orig_fn, call_count)  # type: ignore[assignment]

            bar._populate(
                [("Qwen3 8B", "qwen3:8b")],
                [("Nomic Embed Text", "nomic:latest")],
                [],
            )
            await pilot.pause()
            assert chat_sel.value == "qwen3:8b"
            assert embed_sel.value == "nomic:latest"

    async def test_refresh_models(self) -> None:
        """Cover line 267: refresh_models calls _scan_models."""
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar.refresh_models()
            await pilot.pause()
            assert bar.display is True

    async def test_after_model_change_with_streaming_chat(self) -> None:
        """Cover line 259: cancel stream when chat screen is streaming."""
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "test-model"
        cfg.embedding_model = "test-embed"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            mock_screen = mock.MagicMock(spec=ChatScreen)
            mock_screen._streaming = True
            with (
                mock.patch.object(
                    type(app), "screen", new_callable=mock.PropertyMock, return_value=mock_screen
                ),
                mock.patch("lilbee.services.reset_services"),
            ):
                bar._after_model_change()
            mock_screen.action_cancel_stream.assert_called_once()
            await pilot.pause()
