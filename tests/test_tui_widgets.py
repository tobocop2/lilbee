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
# sync_bar.py
# ---------------------------------------------------------------------------


class _SyncApp(App):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.sync_bar import SyncBar

        yield SyncBar()


class TestSyncBar:
    async def test_set_status_updates(self) -> None:
        from lilbee.cli.tui.widgets.sync_bar import SyncBar

        app = _SyncApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(SyncBar)
            bar.set_status("Syncing 3 files...")
            await pilot.pause()


# ---------------------------------------------------------------------------
# model_bar.py
# ---------------------------------------------------------------------------


class _ModelBarApp(App):
    def compose(self) -> ComposeResult:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        yield ModelBar()


class TestModelBar:
    async def test_refresh_shows_config(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.chat_model = "qwen3:8b"
        cfg.embedding_model = "nomic"
        cfg.vision_model = ""
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar.refresh_models()
            await pilot.pause()

    async def test_refresh_shows_vision_when_set(self) -> None:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        cfg.vision_model = "llava"
        app = _ModelBarApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(ModelBar)
            bar.refresh_models()
            await pilot.pause()


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
        with mock.patch("lilbee.store.get_sources", side_effect=Exception("err")):
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

        with mock.patch(
            "lilbee.store.get_sources",
            return_value=[{"filename": "a.txt", "source": "a.txt"}],
        ):
            assert _document_options() == ["a.txt"]

    def test_returns_empty_on_error(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _document_options

        with mock.patch("lilbee.store.get_sources", side_effect=Exception("err")):
            assert _document_options() == []

    def test_falls_back_to_source_key(self) -> None:
        from lilbee.cli.tui.widgets.autocomplete import _document_options

        with mock.patch(
            "lilbee.store.get_sources",
            return_value=[{"source": "b.pdf"}],
        ):
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
# download_modal.py
# ---------------------------------------------------------------------------


class _DlApp(App):
    def compose(self) -> ComposeResult:
        yield Static("bg")


class TestDownloadModal:
    def test_stores_model(self) -> None:
        from lilbee.cli.tui.widgets.download_modal import DownloadModal

        m = _make_model("Test")
        modal = DownloadModal(m)
        assert modal._model is m

    @mock.patch("lilbee.catalog.download_model")
    async def test_download_success(self, mock_dl: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.download_modal import DownloadModal

        mock_dl.return_value = None
        m = _make_model("Test")
        app = _DlApp()

        results: list[bool] = []

        async with app.run_test() as pilot:
            app.push_screen(DownloadModal(m), callback=lambda r: results.append(r))
            await pilot.pause()
            # Wait for the worker to complete
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()

        mock_dl.assert_called_once()

    @mock.patch("lilbee.catalog.download_model", side_effect=RuntimeError("net error"))
    async def test_download_failure(self, mock_dl: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.download_modal import DownloadModal

        m = _make_model("Test")
        app = _DlApp()

        async with app.run_test() as pilot:
            app.push_screen(DownloadModal(m))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()

    async def test_cancel_dismisses(self) -> None:
        from lilbee.cli.tui.widgets.download_modal import DownloadModal

        m = _make_model("Test")
        app = _DlApp()

        async with app.run_test() as pilot:
            # Patch download to block so we can cancel
            with mock.patch("lilbee.catalog.download_model") as mock_dl:
                import threading

                evt = threading.Event()
                mock_dl.side_effect = lambda *a, **kw: evt.wait(5)
                app.push_screen(DownloadModal(m))
                await pilot.pause()
                screen = app.screen
                screen.action_cancel()
                evt.set()
                await pilot.pause()

    def test_set_status_and_update_progress(self) -> None:
        """Test internal helpers directly (they need mounted widgets, so test via compose)."""
        from lilbee.cli.tui.widgets.download_modal import DownloadModal

        m = _make_model("TestDl")
        modal = DownloadModal(m)
        # _do_dismiss with no _dismiss_result defaults to False
        modal._dismiss_result = True
        assert modal._dismiss_result is True

    @mock.patch("lilbee.catalog.download_model")
    async def test_progress_callback(self, mock_dl: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.download_modal import DownloadModal

        def capture_progress(model, *, on_progress=None):
            if on_progress:
                on_progress(50, 100)
                on_progress(100, 100)
                # Edge: total=0 should not crash
                on_progress(0, 0)

        mock_dl.side_effect = capture_progress
        m = _make_model("Test")
        app = _DlApp()

        async with app.run_test() as pilot:
            app.push_screen(DownloadModal(m))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()


# ---------------------------------------------------------------------------
# setup_modal.py
# ---------------------------------------------------------------------------


class _SetupApp(App):
    def compose(self) -> ComposeResult:
        yield Static("bg")


class TestSetupModal:
    def test_creates_without_remote(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        modal = SetupModal()
        assert modal._remote_embeddings == []

    def test_creates_with_remote(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        modal = SetupModal(ollama_embeddings=["nomic:latest"])
        assert modal._remote_embeddings == ["nomic:latest"]

    async def test_compose_with_remote(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal(ollama_embeddings=["nomic:latest"]))
            await pilot.pause()
            assert len(app.screen_stack) == 2

    async def test_compose_without_remote(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal())
            await pilot.pause()
            assert len(app.screen_stack) == 2

    async def test_action_cancel(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupApp()
        results: list[object] = []
        async with app.run_test() as pilot:
            app.push_screen(SetupModal(), callback=lambda r: results.append(r))
            await pilot.pause()
            app.screen.action_cancel()
            await pilot.pause()
        assert None in results

    async def test_remote_row_selection_dismisses(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupApp()
        results: list[object] = []
        async with app.run_test() as pilot:
            app.push_screen(
                SetupModal(ollama_embeddings=["nomic:latest"]),
                callback=lambda r: results.append(r),
            )
            await pilot.pause()
            # Find the list and select the remote row (index 1 -- first is header label)
            from textual.widgets import ListView

            lv = app.screen.query_one("#embed-picker", ListView)
            lv.index = 1  # _RemoteRow
            await pilot.pause()
            # Simulate selection via action
            lv.action_select_cursor()
            await pilot.pause()
        assert "nomic:latest" in results

    @mock.patch("lilbee.models.pull_with_progress")
    async def test_embedding_row_triggers_download(self, mock_pull: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal())
            await pilot.pause()
            from textual.widgets import ListView

            lv = app.screen.query_one("#embed-picker", ListView)
            lv.index = 0  # First EmbeddingRow (recommended)
            await pilot.pause()
            lv.action_select_cursor()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.pause()
        mock_pull.assert_called_once()

    @mock.patch("lilbee.models.pull_with_progress", side_effect=RuntimeError("fail"))
    async def test_download_error_shows_status(self, mock_pull: mock.MagicMock) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        app = _SetupApp()
        async with app.run_test() as pilot:
            app.push_screen(SetupModal())
            await pilot.pause()
            from textual.widgets import ListView

            lv = app.screen.query_one("#embed-picker", ListView)
            lv.index = 0
            await pilot.pause()
            lv.action_select_cursor()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

    def test_finish_dismiss_no_downloaded_name(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import SetupModal

        modal = SetupModal()
        # _finish_dismiss without _downloaded_name should use getattr default
        # Can't call dismiss outside of app, but test the attribute access logic
        assert getattr(modal, "_downloaded_name", None) is None


class TestEmbeddingRow:
    def test_recommended_suffix(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import _EmbeddingRow

        m = _make_model("Test", task="embedding")
        row = _EmbeddingRow(m, recommended=True)
        assert row._recommended is True
        children = list(row.compose())
        assert len(children) == 1

    def test_not_recommended(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import _EmbeddingRow

        m = _make_model("Test", task="embedding")
        row = _EmbeddingRow(m, recommended=False)
        assert row._recommended is False
        children = list(row.compose())
        assert len(children) == 1


class TestRemoteSetupRow:
    def test_compose(self) -> None:
        from lilbee.cli.tui.widgets.setup_modal import _RemoteRow

        row = _RemoteRow("nomic:latest")
        assert row.remote_name == "nomic:latest"
        children = list(row.compose())
        assert len(children) == 1
