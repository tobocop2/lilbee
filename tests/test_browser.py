"""Tests for the interactive model catalog browser (Textual TUI)."""

from __future__ import annotations

from unittest import mock

from lilbee.catalog import CatalogModel, CatalogResult
from lilbee.cli.browser import (
    _TAB_TO_TASK,
    BrowserState,
    CatalogBrowser,
    ModelRow,
    run_browser,
)


def _make_model(
    name: str = "TestModel",
    task: str = "chat",
    featured: bool = False,
    size_gb: float = 2.0,
    description: str = "A test model",
) -> CatalogModel:
    return CatalogModel(
        name=name,
        hf_repo=f"test/{name.lower().replace(' ', '-')}",
        gguf_filename="*.gguf",
        size_gb=size_gb,
        min_ram_gb=4,
        description=description,
        featured=featured,
        downloads=100,
        task=task,
    )


def _list_children(app: CatalogBrowser, list_id: str) -> list:
    """Get children of a ListView by ID."""
    lv = app.query_one(f"#{list_id}")
    return list(lv.children)


def _model_rows(app: CatalogBrowser, list_id: str) -> list[ModelRow]:
    """Get only ModelRow children from a ListView."""
    return [c for c in _list_children(app, list_id) if isinstance(c, ModelRow)]


class TestBrowserState:
    def test_all_models_combines_featured_and_hf(self) -> None:
        f = [_make_model("F1", featured=True)]
        h = [_make_model("H1")]
        state = BrowserState(featured=f, hf_models=h)
        assert len(state.all_models) == 2
        assert state.all_models[0].name == "F1"
        assert state.all_models[1].name == "H1"

    def test_empty_state(self) -> None:
        state = BrowserState(featured=[], hf_models=[])
        assert state.all_models == []


class TestModelRow:
    def test_stores_model(self) -> None:
        m = _make_model("Qwen3 8B", featured=True)
        row = ModelRow(m)
        assert row.model is m

    def test_compose_yields_static(self) -> None:
        m = _make_model("TestModel", task="chat", size_gb=5.0)
        row = ModelRow(m)
        children = list(row.compose())
        assert len(children) == 1


class TestTabToTask:
    def test_all_tab_maps_to_none(self) -> None:
        assert _TAB_TO_TASK["All"] is None

    def test_chat_tab_maps_to_chat(self) -> None:
        assert _TAB_TO_TASK["Chat"] == "chat"

    def test_embedding_tab_maps_to_embedding(self) -> None:
        assert _TAB_TO_TASK["Embedding"] == "embedding"

    def test_vision_tab_maps_to_vision(self) -> None:
        assert _TAB_TO_TASK["Vision"] == "vision"


class TestCatalogBrowserCompose:
    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_browse_shows_featured(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.pause()
            rows = _model_rows(app, "list-all")
            assert len(rows) > 0
            # At least one should be featured
            assert any(r.model.featured for r in rows)

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_tab_filters_by_task(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.pause()
            rows = _model_rows(app, "list-embedding")
            for row in rows:
                assert row.model.task == "embedding"

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_search_filters_models(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            search_input = app.query_one("#search-input")
            search_input.value = "nomic"
            await pilot.pause()
            rows = _model_rows(app, "list-all")
            for row in rows:
                combined = (
                    row.model.name.lower()
                    + row.model.hf_repo.lower()
                    + row.model.description.lower()
                )
                assert "nomic" in combined

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_enter_selects_model(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.pause()
            all_list = app.query_one("#list-all")
            all_list.focus()
            await pilot.pause()
            rows = _model_rows(app, "list-all")
            if rows:
                idx = list(all_list.children).index(rows[0])
                all_list.index = idx
                await pilot.press("enter")
                assert app.return_value is not None
                assert isinstance(app.return_value, CatalogModel)

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_quit_returns_none(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.press("q")
            assert app.return_value is None

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_escape_returns_none(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.press("escape")
            assert app.return_value is None

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_slash_focuses_search(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.press("slash")
            assert app.query_one("#search-input").has_focus

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_initial_task_sets_tab(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser(initial_task="embedding")
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one("#task-tabs")
            assert tabs.active == "tab-embedding"

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_initial_search_populates_input(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser(initial_search="qwen")
        async with app.run_test() as pilot:
            await pilot.pause()
            search_input = app.query_one("#search-input")
            assert search_input.value == "qwen"

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_hf_models_appear_after_worker(self, mock_catalog: mock.MagicMock) -> None:
        hf_model = _make_model("HF-Model", featured=False)
        mock_catalog.return_value = CatalogResult(total=1, limit=50, offset=0, models=[hf_model])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            # State should be updated with HF models
            assert isinstance(app._state.hf_models, list)

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_no_matches_shows_fallback(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            search_input = app.query_one("#search-input")
            search_input.value = "zzzznonexistent"
            await pilot.pause()
            # All tabs should have at least a fallback "no models" message
            children = _list_children(app, "list-all")
            assert len(children) > 0

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_vision_tab_shows_vision_models(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.pause()
            rows = _model_rows(app, "list-vision")
            for row in rows:
                assert row.model.task == "vision"

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_chat_tab_shows_chat_models(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.pause()
            rows = _model_rows(app, "list-chat")
            assert len(rows) > 0
            for row in rows:
                assert row.model.task == "chat"

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_action_focus_search_directly(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_focus_search()
            await pilot.pause()
            assert app.query_one("#search-input").has_focus

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_action_quit_browser_directly(self, mock_catalog: mock.MagicMock) -> None:
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_quit_browser()
            assert app.return_value is None

    @mock.patch("lilbee.cli.browser.get_catalog")
    async def test_select_non_model_row_noop(self, mock_catalog: mock.MagicMock) -> None:
        """Selecting a header row (not a ModelRow) should not exit."""
        mock_catalog.return_value = CatalogResult(total=0, limit=50, offset=0, models=[])
        app = CatalogBrowser()
        async with app.run_test() as pilot:
            await pilot.pause()
            all_list = app.query_one("#list-all")
            all_list.focus()
            await pilot.pause()
            # Select the first item which is the FEATURED header, not a ModelRow
            if all_list.children:
                all_list.index = 0
                await pilot.press("enter")
                # App should still be running (not exited with a model)
                assert app._selected is None


class TestRunBrowser:
    @mock.patch("lilbee.cli.browser.CatalogBrowser.run")
    def test_returns_selected_model(self, mock_run: mock.MagicMock) -> None:
        model = _make_model("SelectedModel")
        mock_run.return_value = model
        result = run_browser()
        assert result is model

    @mock.patch("lilbee.cli.browser.CatalogBrowser.run")
    def test_returns_none_on_quit(self, mock_run: mock.MagicMock) -> None:
        mock_run.return_value = None
        result = run_browser()
        assert result is None

    @mock.patch("lilbee.cli.browser.CatalogBrowser.run")
    def test_passes_task_and_search(self, mock_run: mock.MagicMock) -> None:
        mock_run.return_value = None
        run_browser(task="embedding", search="nomic")
        mock_run.assert_called_once()


class TestBrowseInteractiveCLI:
    """Test the CLI integration path through models_cmd._browse_interactive."""

    @mock.patch("lilbee.cli.browser.run_browser")
    def test_non_tty_uses_static(self, mock_browser: mock.MagicMock) -> None:
        """CliRunner is not a TTY, so browse should use static table path."""
        from typer.testing import CliRunner

        from lilbee.cli.app import app

        runner = CliRunner()
        with mock.patch("lilbee.catalog.get_catalog") as mock_catalog:
            mock_catalog.return_value = CatalogResult(total=0, limit=20, offset=0, models=[])
            result = runner.invoke(app, ["models", "browse"])
            assert result.exit_code == 0
            mock_browser.assert_not_called()

    @mock.patch("lilbee.cli.models_cmd._install_from_catalog")
    @mock.patch("lilbee.cli.browser.run_browser")
    def test_interactive_installs_on_select(
        self, mock_browser: mock.MagicMock, mock_install: mock.MagicMock
    ) -> None:
        from lilbee.cli.models_cmd import _browse_interactive

        selected = _make_model("PickedModel")
        mock_browser.return_value = selected
        _browse_interactive(task=None, search="")
        mock_install.assert_called_once_with("PickedModel")

    @mock.patch("lilbee.cli.browser.run_browser")
    def test_interactive_noop_on_quit(self, mock_browser: mock.MagicMock) -> None:
        from lilbee.cli.models_cmd import _browse_interactive

        mock_browser.return_value = None
        _browse_interactive(task=None, search="")

    @mock.patch("lilbee.cli.models_cmd._browse_interactive")
    @mock.patch("lilbee.cli.models_cmd.sys")
    def test_tty_calls_browse_interactive(
        self, mock_sys: mock.MagicMock, mock_interactive: mock.MagicMock
    ) -> None:
        """When stdin/stdout are TTY and not json_mode, browse calls _browse_interactive."""
        from lilbee.cli.models_cmd import browse

        mock_sys.stdin.isatty.return_value = True
        mock_sys.stdout.isatty.return_value = True

        from lilbee.config import cfg

        orig_json = cfg.json_mode
        cfg.json_mode = False
        try:
            # Call the internal logic directly — the TTY check uses sys module
            # which we've patched above. We still need to avoid apply_overrides.
            with mock.patch("lilbee.cli.models_cmd.apply_overrides"):
                browse(
                    task="chat",
                    search="test",
                    size=None,
                    featured=False,
                    limit=20,
                    offset=0,
                    data_dir=None,
                    use_global=False,
                )
                mock_interactive.assert_called_once_with(task="chat", search="test")
        finally:
            cfg.json_mode = orig_json


class TestInstallFromCatalog:
    """Test _install_from_catalog helper in models_cmd."""

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_already_installed(self, mock_get: mock.MagicMock) -> None:
        from lilbee.cli.models_cmd import _install_from_catalog

        manager = mock.MagicMock()
        manager.is_installed.return_value = True
        mock_get.return_value = manager
        _install_from_catalog("TestModel")
        manager.pull.assert_not_called()

    @mock.patch("lilbee.models.pull_with_progress")
    @mock.patch("lilbee.catalog.find_catalog_entry")
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_catalog_entry_found(
        self,
        mock_get: mock.MagicMock,
        mock_find: mock.MagicMock,
        mock_pull: mock.MagicMock,
    ) -> None:
        from lilbee.cli.models_cmd import _install_from_catalog

        manager = mock.MagicMock()
        manager.is_installed.return_value = False
        mock_get.return_value = manager
        mock_find.return_value = _make_model("CatalogModel")
        _install_from_catalog("CatalogModel")
        mock_pull.assert_called_once_with("CatalogModel")

    @mock.patch("lilbee.catalog.find_catalog_entry", return_value=None)
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_ollama_fallback(self, mock_get: mock.MagicMock, mock_find: mock.MagicMock) -> None:
        from lilbee.cli.models_cmd import _install_from_catalog
        from lilbee.model_manager import ModelSource

        manager = mock.MagicMock()
        manager.is_installed.return_value = False
        mock_get.return_value = manager
        _install_from_catalog("OllamaModel")
        manager.pull.assert_called_once_with("OllamaModel", ModelSource.OLLAMA)

    @mock.patch("lilbee.catalog.find_catalog_entry", return_value=None)
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_ollama_failure_raises_exit(
        self, mock_get: mock.MagicMock, mock_find: mock.MagicMock
    ) -> None:
        import pytest

        from lilbee.cli.models_cmd import _install_from_catalog

        manager = mock.MagicMock()
        manager.is_installed.return_value = False
        manager.pull.side_effect = RuntimeError("Connection refused")
        mock_get.return_value = manager
        with pytest.raises(SystemExit):
            _install_from_catalog("BadModel")
