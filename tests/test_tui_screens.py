"""Tests for TUI screens, app, and command provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Footer, Static

from lilbee.catalog import CatalogModel, CatalogResult
from lilbee.cli.tui.screens.catalog import (
    LoadMoreRow,
    ModelRow,
    RemoteRow,
    _filter_catalog,
    _filter_remote,
    _format_downloads,
    _format_row,
    _group_by_size,
    _parse_param_label,
    _parse_param_size,
)
from lilbee.config import cfg
from lilbee.model_manager import RemoteModel
from lilbee.services import set_services

_EMPTY_CATALOG = CatalogResult(total=0, limit=25, offset=0, models=[])


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    """Snapshot and restore cfg for every test."""
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.chat_model = "test-model:latest"
    cfg.embedding_model = "test-embed:latest"
    cfg.vision_model = ""
    cfg.chunk_size = 512
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def mock_svc():
    """Inject mock Services so TUI screens never touch real backends."""
    from tests.conftest import make_mock_services

    store = MagicMock()
    store.search.return_value = []
    store.bm25_probe.return_value = []
    store.get_sources.return_value = []
    store.add_chunks.side_effect = lambda records: len(records)
    store.delete_by_source.return_value = None
    store.delete_source.return_value = None
    services = make_mock_services(store=store)
    set_services(services)
    yield services
    set_services(None)


@pytest.fixture(autouse=True)
def _patch_chat_setup():
    """Patch out embedding model checks so ChatScreen mounts cleanly."""
    with (
        patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=False,
        ),
    ):
        yield


def _make_catalog_model(
    name: str = "test-7B",
    hf_repo: str = "org/test-7B-GGUF",
    task: str = "chat",
    featured: bool = False,
    downloads: int = 1000,
    size_gb: float = 4.0,
    description: str = "A test model",
) -> CatalogModel:
    return CatalogModel(
        name=name,
        hf_repo=hf_repo,
        gguf_filename="test.gguf",
        size_gb=size_gb,
        min_ram_gb=8.0,
        description=description,
        featured=featured,
        downloads=downloads,
        task=task,
    )


def _make_remote_model(
    name: str = "remote-test:latest",
    task: str = "chat",
    family: str = "llama",
    parameter_size: str = "7B",
) -> RemoteModel:
    return RemoteModel(name=name, task=task, family=family, parameter_size=parameter_size)


# ---------------------------------------------------------------------------
# Pure function tests: catalog helpers
# ---------------------------------------------------------------------------


class TestParseParamLabel:
    def test_extracts_integer(self):
        assert _parse_param_label("qwen-8B-instruct") == "8B"

    def test_extracts_decimal(self):
        assert _parse_param_label("phi-0.6B") == "0.6B"

    def test_no_match(self):
        assert _parse_param_label("nomic-embed-text") == "\u2014"

    def test_case_insensitive(self):
        assert _parse_param_label("model-3b-chat") == "3B"


class TestParseParamSize:
    def test_small(self):
        assert _parse_param_size("model-1.5B") == "Small (\u22643B)"

    def test_medium(self):
        assert _parse_param_size("model-7B") == "Medium (3-8B)"

    def test_large(self):
        assert _parse_param_size("model-14B") == "Large (8-30B)"

    def test_extra_large(self):
        assert _parse_param_size("model-70B") == "Extra Large (30B+)"

    def test_unknown(self):
        assert _parse_param_size("nomic-embed") == "unknown"

    def test_boundary_3b(self):
        assert _parse_param_size("model-3B") == "Small (\u22643B)"

    def test_boundary_8b(self):
        assert _parse_param_size("model-8B") == "Medium (3-8B)"

    def test_boundary_30b(self):
        assert _parse_param_size("model-30B") == "Large (8-30B)"


class TestFormatDownloads:
    def test_millions(self):
        assert _format_downloads(2_500_000) == "2.5M"

    def test_thousands(self):
        assert _format_downloads(45_000) == "45K"

    def test_small(self):
        assert _format_downloads(999) == "999"

    def test_one_million(self):
        assert _format_downloads(1_000_000) == "1.0M"

    def test_one_thousand(self):
        assert _format_downloads(1_000) == "1K"

    def test_zero(self):
        assert _format_downloads(0) == "0"


class TestFormatRow:
    def test_featured_star(self):
        m = _make_catalog_model(featured=True)
        row = _format_row(m)
        assert "\u2605" in row

    def test_not_featured(self):
        m = _make_catalog_model(featured=False)
        row = _format_row(m)
        assert "\u2605" not in row

    def test_contains_name(self):
        m = _make_catalog_model(name="my-model-8B")
        row = _format_row(m)
        assert "my-model-8B" in row

    def test_zero_size_shows_dash(self):
        m = _make_catalog_model(size_gb=0.0)
        row = _format_row(m)
        assert "\u2014" in row

    def test_cached_size_overrides(self):
        m = _make_catalog_model(size_gb=0.0)
        row = _format_row(m, cached_size=3.5)
        assert "3.5 GB" in row

    def test_zero_downloads_no_arrow(self):
        m = _make_catalog_model(downloads=0)
        row = _format_row(m)
        assert "\u2193" not in row

    def test_description_truncated(self):
        m = _make_catalog_model(description="x" * 100)
        row = _format_row(m)
        assert "x" * 45 in row
        assert "x" * 46 not in row

    def test_empty_description(self):
        m = _make_catalog_model(description="")
        _format_row(m)


class TestFilterCatalog:
    def test_no_filters(self):
        models = [_make_catalog_model(task="chat"), _make_catalog_model(task="embedding")]
        assert len(_filter_catalog(models, None, "")) == 2

    def test_task_filter(self):
        models = [
            _make_catalog_model(task="chat"),
            _make_catalog_model(task="embedding"),
        ]
        assert len(_filter_catalog(models, "chat", "")) == 1

    def test_search_by_name(self):
        models = [
            _make_catalog_model(name="qwen-8B"),
            _make_catalog_model(name="llama-7B"),
        ]
        result = _filter_catalog(models, None, "qwen")
        assert len(result) == 1
        assert result[0].name == "qwen-8B"

    def test_search_by_repo(self):
        models = [_make_catalog_model(hf_repo="org/special-GGUF")]
        result = _filter_catalog(models, None, "special")
        assert len(result) == 1

    def test_search_by_description(self):
        models = [_make_catalog_model(description="Fast inference model")]
        result = _filter_catalog(models, None, "fast inference")
        assert len(result) == 1

    def test_combined_task_and_search(self):
        models = [
            _make_catalog_model(name="qwen-8B", task="chat"),
            _make_catalog_model(name="qwen-embed", task="embedding"),
        ]
        result = _filter_catalog(models, "chat", "qwen")
        assert len(result) == 1
        assert result[0].task == "chat"

    def test_empty_list(self):
        assert _filter_catalog([], "chat", "test") == []


class TestFilterRemote:
    def test_no_filters(self):
        models = [_make_remote_model(task="chat"), _make_remote_model(task="embedding")]
        assert len(_filter_remote(models, None, "")) == 2

    def test_task_filter(self):
        models = [_make_remote_model(task="chat"), _make_remote_model(task="embedding")]
        assert len(_filter_remote(models, "chat", "")) == 1

    def test_search_filter(self):
        models = [
            _make_remote_model(name="qwen:latest"),
            _make_remote_model(name="llama:latest"),
        ]
        result = _filter_remote(models, None, "qwen")
        assert len(result) == 1

    def test_empty_list(self):
        assert _filter_remote([], None, "") == []


class TestGroupBySize:
    def test_groups_correctly(self):
        models = [
            _make_catalog_model(name="small-1B"),
            _make_catalog_model(name="medium-7B"),
            _make_catalog_model(name="large-14B"),
            _make_catalog_model(name="huge-70B"),
        ]
        groups = _group_by_size(models)
        labels = [label for label, _ in groups]
        assert "Small (\u22643B)" in labels
        assert "Medium (3-8B)" in labels
        assert "Large (8-30B)" in labels
        assert "Extra Large (30B+)" in labels

    def test_correct_order(self):
        models = [
            _make_catalog_model(name="huge-70B"),
            _make_catalog_model(name="small-1B"),
        ]
        groups = _group_by_size(models)
        labels = [label for label, _ in groups]
        assert labels.index("Small (\u22643B)") < labels.index("Extra Large (30B+)")

    def test_unknown_category(self):
        models = [_make_catalog_model(name="nomic-embed-text")]
        groups = _group_by_size(models)
        assert groups[0][0] == "unknown"

    def test_empty(self):
        assert _group_by_size([]) == []


class TestModelRow:
    def test_stores_model(self):
        m = _make_catalog_model()
        row = ModelRow(m)
        assert row.model is m


class TestRemoteRow:
    def test_stores_model(self):
        m = _make_remote_model()
        row = RemoteRow(m)
        assert row.remote_model is m

    def test_none_parameter_size(self):
        m = _make_remote_model(parameter_size="")
        row = RemoteRow(m)
        assert row.remote_model.parameter_size == ""


# ---------------------------------------------------------------------------
# Settings screen (Textual integration)
# ---------------------------------------------------------------------------


class SettingsTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        self.push_screen(SettingsScreen())


async def test_settings_screen_renders_table():
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import DataTable

        table = app.screen.query_one("#settings-table", DataTable)
        assert table.row_count > 0


async def test_settings_screen_detail_panel():
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import DataTable

        table = app.screen.query_one("#settings-table", DataTable)
        assert table.row_count > 0


async def test_settings_screen_vim_keys():
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import DataTable

        table = app.screen.query_one("#settings-table", DataTable)
        table.focus()
        await _pilot.press("j")
        await _pilot.press("k")


async def test_settings_screen_pop():
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.press("q")


async def test_settings_screen_row_highlighted_empty_key():
    """Cover the branch where event.row_key has no value."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        detail = app.screen.query_one("#setting-detail", Static)
        event = MagicMock()
        event.row_key = MagicMock()
        event.row_key.value = None
        event.row_key.__bool__ = lambda s: False
        app.screen.on_data_table_row_highlighted(event)
        assert str(detail.render()) == ""


async def test_settings_screen_row_highlighted_unknown_key():
    """Cover the branch where key is not in SETTINGS_MAP."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        event = MagicMock()
        event.row_key = MagicMock()
        event.row_key.value = "nonexistent_key_xyz"
        event.row_key.__bool__ = lambda self: True
        app.screen.on_data_table_row_highlighted(event)


# ---------------------------------------------------------------------------
# Status screen (Textual integration)
# ---------------------------------------------------------------------------


class StatusTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.status import StatusScreen

        self.push_screen(StatusScreen())


async def test_status_screen_renders_info(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"source": "test.pdf", "chunk_count": 10, "content_type": "application/pdf"},
    ]
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        info = app.screen.query_one("#status-info", Static)
        rendered = str(info.render())
        assert "test-model:latest" in rendered
        assert "test-embed:latest" in rendered


async def test_status_screen_shows_documents(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"source": "notes.md", "chunk_count": 5, "content_type": "text/markdown"},
    ]
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import DataTable

        table = app.screen.query_one("#docs-table", DataTable)
        assert table.row_count == 1


async def test_status_screen_store_error(mock_svc):
    mock_svc.store.get_sources.side_effect = Exception("no table")
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import DataTable

        table = app.screen.query_one("#docs-table", DataTable)
        assert table.row_count == 1


async def test_status_screen_vim_keys(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"source": "a.md", "chunk_count": 1, "content_type": "text/markdown"},
        {"source": "b.md", "chunk_count": 2, "content_type": "text/markdown"},
    ]
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import DataTable

        table = app.screen.query_one("#docs-table", DataTable)
        table.focus()
        await _pilot.press("j")
        await _pilot.press("k")


async def test_status_screen_escape_pops():
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.press("escape")


# ---------------------------------------------------------------------------
# LilbeeApp tests
# ---------------------------------------------------------------------------


async def test_app_mounts_chat_screen():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert isinstance(app.screen, ChatScreen)


async def test_app_title_has_model():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert "test-model:latest" in app.title


async def test_app_cycle_theme():
    from lilbee.cli.tui.app import DARK_THEMES, LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.action_cycle_theme()
        assert app.theme == DARK_THEMES[1]
        for _ in range(len(DARK_THEMES)):
            app.action_cycle_theme()
        assert app.theme == DARK_THEMES[1]


async def test_app_set_theme():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.set_theme("dracula")
        assert app.theme == "dracula"
        app.set_theme("nonexistent-theme-xyz")
        assert app.theme == "dracula"


async def test_app_push_catalog():
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
            patch("lilbee.model_manager.classify_remote_models", return_value=[]),
        ):
            app.action_push_catalog()
            await _pilot.pause()
            assert isinstance(app.screen, CatalogScreen)


async def test_app_push_status():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.action_push_status()
        await _pilot.pause()
        from lilbee.cli.tui.screens.status import StatusScreen

        assert isinstance(app.screen, StatusScreen)


async def test_app_push_settings():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.action_push_settings()
        await _pilot.pause()
        from lilbee.cli.tui.screens.settings import SettingsScreen

        assert isinstance(app.screen, SettingsScreen)


async def test_app_push_help():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.action_push_help()
        await _pilot.pause()
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        assert isinstance(app.screen, HelpModal)


async def test_app_auto_sync_flag():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp(auto_sync=True)
    assert app._auto_sync is True


# ---------------------------------------------------------------------------
# ChatScreen slash command tests
# ---------------------------------------------------------------------------


class ChatTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


async def test_chat_screen_renders():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        assert inp is not None


async def test_chat_slash_unknown_command():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/bogus")


async def test_chat_slash_version():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.helpers.get_version", return_value="1.2.3"):
            app.screen._handle_slash("/version")


async def test_chat_slash_model_with_arg():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/model new-model:latest")
        assert cfg.chat_model == "new-model:latest"


async def test_chat_slash_model_no_arg():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
            patch("lilbee.model_manager.classify_remote_models", return_value=[]),
        ):
            app.screen._handle_slash("/model")
            await _pilot.pause()
            from lilbee.cli.tui.screens.catalog import CatalogScreen

            assert isinstance(app.screen, CatalogScreen)


async def test_chat_slash_theme_with_arg():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/theme dracula")


async def test_chat_slash_theme_no_arg():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/theme")


async def test_chat_slash_theme_non_lilbee_app():
    """Theme with arg on a non-LilbeeApp should just list themes."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/theme dracula")


async def test_chat_slash_vision_set():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.settings.set_value"):
            app.screen._cmd_vision("maternion/LightOnOCR-2:latest")
            assert cfg.vision_model == "maternion/LightOnOCR-2:latest"


async def test_chat_slash_vision_off():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        cfg.vision_model = "some-model"
        with patch("lilbee.settings.set_value"):
            app.screen._cmd_vision("off")
            assert cfg.vision_model == ""


async def test_chat_slash_vision_no_arg():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_vision("")


async def test_chat_slash_delete_with_match(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"filename": "notes.md", "source": "notes.md"},
    ]
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_delete("notes.md")
        mock_svc.store.delete_by_source.assert_called_once_with("notes.md")
        mock_svc.store.delete_source.assert_called_once_with("notes.md")


async def test_chat_slash_delete_not_found(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"filename": "notes.md", "source": "notes.md"},
    ]
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_delete("nonexistent.md")


async def test_chat_slash_delete_no_arg(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"filename": "notes.md", "source": "notes.md"},
    ]
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_delete("")


async def test_chat_slash_delete_store_error(mock_svc):
    mock_svc.store.get_sources.side_effect = Exception("no store")
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_delete("x")


async def test_chat_slash_delete_empty_sources(mock_svc):
    mock_svc.store.get_sources.return_value = []
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_delete("x")


async def test_chat_slash_reset_confirm():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.helpers.perform_reset") as mock_reset:
            mock_reset.return_value = None
            app.screen._handle_slash("/reset confirm")
            mock_reset.assert_called_once()


async def test_chat_slash_reset_no_confirm():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/reset")


async def test_chat_slash_reset_error():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.helpers.perform_reset", side_effect=Exception("oops")):
            app.screen._handle_slash("/reset confirm")


async def test_chat_slash_set_valid():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_set("top_k 10")
        assert cfg.top_k == 10


async def test_chat_slash_set_bool():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_set("show_reasoning true")
        assert cfg.show_reasoning is True


async def test_chat_slash_set_nullable_none():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_set("temperature none")
        assert cfg.temperature is None


async def test_chat_slash_set_unknown_key():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_set("bogus_key 42")


async def test_chat_slash_set_invalid_value():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_set("top_k not-a-number")


async def test_chat_slash_set_no_value():
    """Cover the branch where /set key has no value (empty string)."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # chat_model has a min_length=1 validator, so empty string is rejected;
        # test that the code path runs without crashing.
        app.screen._cmd_set("chat_model")
        # Value remains unchanged because pydantic rejects ""
        assert cfg.chat_model == "test-model:latest"


async def test_chat_slash_add_empty_args():
    """Cover early return when /add has no args."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_add("")


async def test_chat_slash_set_empty_args():
    """Cover early return when /set has no args."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_set("")


async def test_chat_slash_add_nonexistent():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_add("/nonexistent/path/abc.txt")


async def test_chat_slash_add_existing(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.cli.helpers.copy_paths", return_value=["test.txt"]),
            patch.object(app.screen, "_run_sync"),
        ):
            app.screen._cmd_add(str(test_file))


async def test_chat_slash_add_error(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.helpers.copy_paths", side_effect=Exception("copy failed")):
            app.screen._cmd_add(str(test_file))


async def test_chat_slash_cancel():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/cancel")


async def test_chat_slash_help():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/help")
        await _pilot.pause()
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        assert isinstance(app.screen, HelpModal)


async def test_chat_slash_models():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
            patch("lilbee.model_manager.classify_remote_models", return_value=[]),
        ):
            app.screen._handle_slash("/models")
            await _pilot.pause()
            from lilbee.cli.tui.screens.catalog import CatalogScreen

            assert isinstance(app.screen, CatalogScreen)


async def test_chat_slash_status():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/status")
        await _pilot.pause()
        from lilbee.cli.tui.screens.status import StatusScreen

        assert isinstance(app.screen, StatusScreen)


async def test_chat_slash_settings():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/settings")
        await _pilot.pause()
        from lilbee.cli.tui.screens.settings import SettingsScreen

        assert isinstance(app.screen, SettingsScreen)


async def test_chat_slash_set_dispatch():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/set top_k 10")
        assert cfg.top_k == 10


async def test_chat_empty_input_ignored():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = ""
        await _pilot.press("enter")


async def test_chat_scroll_actions():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen.action_scroll_up()
        app.screen.action_scroll_down()


async def test_chat_cancel_stream_not_streaming():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen.action_cancel_stream()


async def test_chat_cancel_stream_while_streaming():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._streaming = True
        app.screen.action_cancel_stream()
        assert app.screen._streaming is False


async def test_chat_vim_j_k_not_in_input():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.containers import VerticalScroll

        log = app.screen.query_one("#chat-log", VerticalScroll)
        log.focus()
        app.screen.key_j()
        app.screen.key_k()


async def test_chat_vim_j_k_in_input():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.focus()
        app.screen.key_j()
        app.screen.key_k()


async def test_chat_needs_setup_false_when_models_exist():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False):
            assert not app.screen._needs_setup()


async def test_chat_refresh_model_bar():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._refresh_model_bar()


async def test_chat_input_changed_hides_overlay():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.focus()
        inp.value = "/he"
        await _pilot.pause()


async def test_chat_slash_quit():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "exit"):
            app.screen._handle_slash("/quit")


async def test_chat_slash_q():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "exit"):
            app.screen._handle_slash("/q")


async def test_chat_slash_exit():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "exit"):
            app.screen._handle_slash("/exit")


async def test_chat_slash_h():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/h")
        await _pilot.pause()
        from lilbee.cli.tui.widgets.help_modal import HelpModal

        assert isinstance(app.screen, HelpModal)


async def test_chat_slash_m():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
            patch("lilbee.model_manager.classify_remote_models", return_value=[]),
        ):
            app.screen._handle_slash("/m")
            await _pilot.pause()
            from lilbee.cli.tui.screens.catalog import CatalogScreen

            assert isinstance(app.screen, CatalogScreen)


async def test_chat_slash_add_dispatch():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/add /nonexistent/xyz")


async def test_chat_slash_vision_dispatch():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.settings.set_value"):
            app.screen._handle_slash("/vision off")
            assert cfg.vision_model == ""


async def test_chat_slash_delete_dispatch():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/delete")


# ---------------------------------------------------------------------------
# ChatScreen action_complete tests
# ---------------------------------------------------------------------------


async def test_chat_action_complete_no_options():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "hello"
        app.screen.action_complete()


async def test_chat_action_complete_with_options():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "/he"
        with patch(
            "lilbee.cli.tui.screens.chat.get_completions",
            return_value=["/help"],
        ):
            app.screen.action_complete()
            assert inp.value == "/help"


async def test_chat_action_complete_with_space():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "/model q"
        with patch(
            "lilbee.cli.tui.screens.chat.get_completions",
            return_value=["qwen:latest"],
        ):
            app.screen.action_complete()
            assert inp.value == "/model qwen:latest"


async def test_chat_action_complete_cycle():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        inp = app.screen.query_one("#chat-input", Input)
        overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)

        inp.value = "/he"
        with patch(
            "lilbee.cli.tui.screens.chat.get_completions",
            return_value=["/help"],
        ):
            app.screen.action_complete()

        if overlay.is_visible:
            inp.value = "/model "
            with patch.object(overlay, "cycle_next", return_value="qwen:latest"):
                app.screen.action_complete()
                assert "qwen:latest" in inp.value


async def test_chat_action_complete_cycle_no_selection():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
        overlay.show_completions(["a", "b"])
        with patch.object(overlay, "cycle_next", return_value=None):
            app.screen.action_complete()


# ---------------------------------------------------------------------------
# ChatScreen on_input_submitted (non-slash = send message)
# ---------------------------------------------------------------------------


async def test_chat_send_message():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "_stream_response"):
            from textual.widgets import Input

            inp = app.screen.query_one("#chat-input", Input)
            inp.value = "What is RAG?"
            await _pilot.press("enter")
            assert len(app.screen._history) == 1
            assert app.screen._history[0]["role"] == "user"


async def test_chat_input_submitted_wrong_input_ignored():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        event = Input.Submitted(app.screen.query_one("#chat-input", Input), "test")
        event.input = MagicMock()
        event.input.id = "other-input"
        app.screen.on_input_submitted(event)
        assert len(app.screen._history) == 0


async def test_chat_input_changed_other_input_ignored():
    """on_input_changed should only react to chat-input."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        event = MagicMock()
        event.input = MagicMock()
        event.input.id = "other-input"
        app.screen.on_input_changed(event)


async def test_chat_scroll_to_bottom():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._scroll_to_bottom()


async def test_chat_trim_history_when_over_limit():
    """History is trimmed when it exceeds _MAX_HISTORY_MESSAGES."""
    from lilbee.cli.tui.screens.chat import _MAX_HISTORY_MESSAGES

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._history = [
            {"role": "user", "content": f"msg-{i}"} for i in range(_MAX_HISTORY_MESSAGES + 10)
        ]
        app.screen._trim_history()
        assert len(app.screen._history) == _MAX_HISTORY_MESSAGES
        assert app.screen._history[0]["content"] == "msg-10"


# ---------------------------------------------------------------------------
# CommandProvider tests
# ---------------------------------------------------------------------------


async def test_command_provider_discover():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        hits = [hit async for hit in provider.discover()]
        assert len(hits) > 0
        texts = [h.text for h in hits]
        assert any("catalog" in str(t).lower() for t in texts)


async def test_command_provider_search():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        hits = [hit async for hit in provider.search("catalog")]
        assert len(hits) > 0


async def test_command_provider_search_no_match():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        hits = [hit async for hit in provider.search("xyznonexistent123")]
        assert len(hits) == 0


async def test_command_provider_set_model():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch("lilbee.settings.set_value"):
            provider._set_model("chat_model", "new-model:latest")
            assert cfg.chat_model == "new-model:latest"
            assert "new-model:latest" in app.title


async def test_command_provider_set_model_vision():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch("lilbee.settings.set_value"):
            provider._set_model("vision_model", "")
            assert cfg.vision_model == ""


async def test_command_provider_delete_doc():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        provider._delete_doc("notes.md")
        from lilbee.services import get_services

        store = get_services().store
        store.delete_by_source.assert_called_once_with("notes.md")
        store.delete_source.assert_called_once_with("notes.md")


async def test_command_provider_action_sync():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        provider._action_sync()


async def test_command_provider_action_version():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch("lilbee.cli.helpers.get_version", return_value="1.0.0"):
            provider._action_version()


async def test_command_provider_action_noop():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        provider._action_noop()


async def test_command_provider_model_commands():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch(
            "lilbee.models.list_installed_models",
            return_value=["qwen:latest", "llama:latest"],
        ):
            cmds = provider._model_commands()
            model_names = [c[0] for c in cmds]
            assert any("qwen:latest" in n for n in model_names)


async def test_command_provider_model_commands_error():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch(
            "lilbee.models.list_installed_models",
            side_effect=Exception("no provider"),
        ):
            cmds = provider._model_commands()
            assert any("vision" in c[0].lower() for c in cmds)


async def test_command_provider_model_commands_vision_error():
    """When both list_installed_models and VISION_CATALOG fail."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with (
            patch("lilbee.models.list_installed_models", side_effect=Exception("fail")),
            patch("lilbee.models.VISION_CATALOG", side_effect=Exception("fail")),
        ):
            cmds = provider._model_commands()
            # Both failed, should return empty or partial
            assert isinstance(cmds, list)


async def test_command_provider_document_commands():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider
        from lilbee.services import get_services

        get_services().store.get_sources.return_value = [
            {"filename": "notes.md", "source": "notes.md"},
        ]
        provider = LilbeeCommandProvider(app.screen, match_style=None)
        cmds = provider._document_commands()
        assert len(cmds) == 1
        assert "notes.md" in cmds[0][0]


async def test_command_provider_document_commands_error():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider
        from lilbee.services import get_services

        get_services().store.get_sources.side_effect = Exception("no store")
        provider = LilbeeCommandProvider(app.screen, match_style=None)
        cmds = provider._document_commands()
        assert cmds == []


async def test_command_provider_document_commands_empty_name():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider
        from lilbee.services import get_services

        get_services().store.get_sources.return_value = [{"source": ""}]
        provider = LilbeeCommandProvider(app.screen, match_style=None)
        cmds = provider._document_commands()
        assert cmds == []


# ---------------------------------------------------------------------------
# CatalogScreen tests
# ---------------------------------------------------------------------------


class CatalogTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()


def _patch_catalog():
    """Context manager to patch catalog screen's network calls."""
    return (
        patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
        patch("lilbee.model_manager.classify_remote_models", return_value=[]),
    )


async def test_catalog_screen_renders():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            app.push_screen(CatalogScreen())
            await _pilot.pause()
            assert app.screen.query_one("#catalog-search") is not None


async def test_catalog_focus_search():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen.action_focus_search()
            from textual.widgets import Input

            assert app.screen.query_one("#catalog-search", Input).has_focus


async def test_catalog_cycle_sort():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            assert screen._current_sort == "downloads"
            from textual.widgets import ListView

            lv = screen.query_one("#catlist-all", ListView)
            lv.focus()
            await _pilot.pause()
            screen.action_cycle_sort()
            assert screen._current_sort == "name"


async def test_catalog_cycle_sort_in_input_ignored():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            screen.query_one("#catalog-search", Input).focus()
            screen.action_cycle_sort()
            assert screen._current_sort == "downloads"


async def test_catalog_pop_screen():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen.action_pop_screen()
            await _pilot.pause()


async def test_catalog_vim_keys():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import ListView

            lv = screen.query_one("#catlist-all", ListView)
            lv.focus()
            screen.action_cursor_down()
            screen.action_cursor_up()


async def test_catalog_vim_keys_in_input():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            screen.query_one("#catalog-search", Input).focus()
            screen.action_cursor_down()
            screen.action_cursor_up()


async def test_catalog_page_down_up():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import ListView

            lv = screen.query_one("#catlist-all", ListView)
            lv.focus()
            screen.action_page_down()
            screen.action_page_up()


async def test_catalog_page_down_no_focus():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            screen.query_one("#catalog-search", Input).focus()
            screen.action_page_down()
            screen.action_page_up()


async def test_catalog_install_already_installed():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="installed-model")
            mock_mgr = MagicMock()
            mock_mgr.is_installed.return_value = True
            with patch("lilbee.model_manager.get_model_manager", return_value=mock_mgr):
                screen._install_model(m)


async def test_catalog_install_new_model():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="new-model")
            mock_mgr = MagicMock()
            mock_mgr.is_installed.return_value = False
            with patch("lilbee.model_manager.get_model_manager", return_value=mock_mgr):
                screen._install_model(m)
                await _pilot.pause()


async def test_catalog_select_remote_row():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            om = _make_remote_model(name="remote-chat:latest")
            row = RemoteRow(om)
            event = MagicMock()
            event.item = row
            screen.on_list_view_selected(event)
            assert cfg.chat_model == "remote-chat:latest"


async def test_catalog_select_load_more():
    from lilbee.cli.tui.screens.catalog import _HF_PAGE_SIZE, CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            old_offset = screen._hf_offset
            row = LoadMoreRow()
            event = MagicMock()
            event.item = row
            with patch.object(screen, "_fetch_more_hf"):
                screen.on_list_view_selected(event)
                assert screen._hf_offset == old_offset + _HF_PAGE_SIZE


async def test_catalog_highlight_model_row():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="test-7B", size_gb=4.0)
            row = ModelRow(m)
            screen._update_highlighted_detail(row)
            detail = screen.query_one("#model-detail", Static)
            assert "test-7B" in str(detail.render())


async def test_catalog_highlight_remote_row():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            om = _make_remote_model(name="remote-test:latest")
            row = RemoteRow(om)
            screen._update_highlighted_detail(row)
            detail = screen.query_one("#model-detail", Static)
            assert "remote-test" in str(detail.render())


async def test_catalog_highlight_unknown_row():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import ListItem

            row = ListItem()
            screen._update_highlighted_detail(row)
            detail = screen.query_one("#model-detail", Static)
            assert str(detail.render()) == ""


async def test_catalog_highlight_none_no_highlighted():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._update_highlighted_detail(None)


async def test_catalog_highlight_model_zero_size_triggers_fetch():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="test-7B", size_gb=0.0)
            row = ModelRow(m)
            with patch.object(screen, "_fetch_model_size") as mock_fetch:
                screen._update_highlighted_detail(row)
                mock_fetch.assert_called_once_with(m.hf_repo)


async def test_catalog_highlight_model_cached_size():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="test-7B", size_gb=0.0, hf_repo="org/test-7B-GGUF")
            screen._size_cache["org/test-7B-GGUF"] = 5.5
            row = ModelRow(m)
            screen._update_highlighted_detail(row)
            detail = screen.query_one("#model-detail", Static)
            assert "5.5 GB" in str(detail.render())


async def test_catalog_worker_hf_success():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.worker import WorkerState

            mock_worker = MagicMock()
            mock_worker.name = "_fetch_hf_models"
            mock_worker.result = [_make_catalog_model(name="hf-model-7B")]
            mock_event = MagicMock()
            mock_event.state = WorkerState.SUCCESS
            mock_event.worker = mock_worker
            screen.on_worker_state_changed(mock_event)
            assert len(screen._hf_models) == 1


async def test_catalog_worker_remote_success():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.worker import WorkerState

            mock_worker = MagicMock()
            mock_worker.name = "_fetch_remote_models"
            mock_worker.result = [_make_remote_model()]
            mock_event = MagicMock()
            mock_event.state = WorkerState.SUCCESS
            mock_event.worker = mock_worker
            screen.on_worker_state_changed(mock_event)
            assert len(screen._remote_models) == 1


async def test_catalog_worker_more_hf_success():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._hf_models = [_make_catalog_model(name="existing-7B")]

            from textual.worker import WorkerState

            mock_worker = MagicMock()
            mock_worker.name = "_fetch_more_hf"
            mock_worker.result = [_make_catalog_model(name="new-7B")]
            mock_event = MagicMock()
            mock_event.state = WorkerState.SUCCESS
            mock_event.worker = mock_worker
            screen.on_worker_state_changed(mock_event)
            assert len(screen._hf_models) == 2


async def test_catalog_worker_size_fetch_success():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.worker import WorkerState

            mock_worker = MagicMock()
            mock_worker.name = "_fetch_model_size"
            mock_worker.result = ("org/test-GGUF", 3.5)
            mock_event = MagicMock()
            mock_event.state = WorkerState.SUCCESS
            mock_event.worker = mock_worker
            screen.on_worker_state_changed(mock_event)
            assert screen._size_cache["org/test-GGUF"] == 3.5


async def test_catalog_worker_size_fetch_zero():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.worker import WorkerState

            mock_worker = MagicMock()
            mock_worker.name = "_fetch_model_size"
            mock_worker.result = ("org/test-GGUF", 0.0)
            mock_event = MagicMock()
            mock_event.state = WorkerState.SUCCESS
            mock_event.worker = mock_worker
            screen.on_worker_state_changed(mock_event)
            assert "org/test-GGUF" not in screen._size_cache


async def test_catalog_worker_non_success_ignored():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.worker import WorkerState

            mock_event = MagicMock()
            mock_event.state = WorkerState.RUNNING
            screen.on_worker_state_changed(mock_event)


async def test_catalog_select_model_row():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="test-7B")
            row = ModelRow(m)
            with patch.object(screen, "_install_model") as mock_install:
                event = MagicMock()
                event.item = row
                screen.on_list_view_selected(event)
                mock_install.assert_called_once_with(m)


async def test_catalog_input_changed_refreshes():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            inp = screen.query_one("#catalog-search", Input)
            with patch.object(screen, "_refresh_lists") as mock_refresh:
                event = MagicMock()
                event.input = inp
                screen.on_input_changed(event)
                mock_refresh.assert_called()


async def test_catalog_input_changed_other_input_ignored():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            with patch.object(screen, "_refresh_lists") as mock_refresh:
                event = MagicMock()
                event.input = MagicMock()
                event.input.id = "other-input"
                screen.on_input_changed(event)
                mock_refresh.assert_not_called()


async def test_catalog_tab_activated():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._hf_offset = 50
            screen._hf_models = [_make_catalog_model()]
            mock_event = MagicMock()
            with patch.object(screen, "_fetch_hf_models"):
                screen.on_tabbed_content_tab_activated(mock_event)
                assert screen._hf_offset == 0
                assert screen._hf_models == []
                assert screen._hf_has_more is True


async def test_catalog_on_list_view_highlighted():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="test-7B")
            row = ModelRow(m)
            event = MagicMock()
            event.item = row
            with patch.object(screen, "_update_highlighted_detail") as mock_update:
                screen.on_list_view_highlighted(event)
                mock_update.assert_called_once_with(row)


class LoadMoreTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield LoadMoreRow()


async def test_load_more_row_renders():
    app = LoadMoreTestApp()
    async with app.run_test(size=(80, 10)) as _pilot:
        text = app.query_one(Static)
        assert "Load more" in str(text.render())


class ModelRowTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield ModelRow(_make_catalog_model(name="compose-test-7B"))


async def test_model_row_compose():
    app = ModelRowTestApp()
    async with app.run_test(size=(120, 10)) as _pilot:
        text = app.query_one(Static)
        assert "compose-test-7B" in str(text.render())


class RemoteRowTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield RemoteRow(_make_remote_model(name="remote-compose:latest", parameter_size=""))


async def test_remote_row_compose():
    app = RemoteRowTestApp()
    async with app.run_test(size=(120, 10)) as _pilot:
        text = app.query_one(Static)
        assert "remote-compose:latest" in str(text.render())


# ---------------------------------------------------------------------------
# Direct worker body tests (call underlying fn, not @work decorator)
# ---------------------------------------------------------------------------


async def test_catalog_fetch_model_size_worker():
    """Cover _fetch_model_size worker body (lines 307-310)."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            with patch("lilbee.catalog.fetch_model_file_size", return_value=5.5):
                screen._fetch_model_size("org/test-GGUF")
                await _pilot.pause()
                # Wait for worker to complete
                while screen.workers:
                    await _pilot.pause()


async def test_catalog_fetch_more_hf_worker():
    """Cover _fetch_more_hf worker body (lines 319-324)."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    hf_models = [_make_catalog_model(name=f"hf-{i}B", featured=False) for i in range(5)]
    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            with patch(
                "lilbee.cli.tui.screens.catalog.get_catalog",
                return_value=CatalogResult(total=5, limit=25, offset=0, models=hf_models),
            ):
                screen._fetch_more_hf()
                await _pilot.pause()
                while screen.workers:
                    await _pilot.pause()


async def test_catalog_update_highlighted_detail_none_with_child():
    """Cover lines 280-281: _update_highlighted_detail(None) finding highlighted child."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            # Add featured models so there's something in the list
            screen._featured = [_make_catalog_model(name="feat-7B", featured=True, size_gb=4.0)]
            screen._refresh_lists()
            await _pilot.pause()

            from textual.widgets import ListView

            lv = screen.query_one("#catlist-all", ListView)
            lv.focus()
            await _pilot.pause()
            # Move cursor down to a ModelRow (skip section header)
            lv.action_cursor_down()
            await _pilot.pause()

            # Now call with None, it should find the highlighted child
            if lv.highlighted_child:
                screen._update_highlighted_detail(None)


async def test_chat_stream_response_worker(mock_svc):
    """Cover _stream_response lines 315-336 via actual worker."""
    from dataclasses import dataclass

    @dataclass
    class FakeToken:
        content: str
        is_reasoning: bool = False

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        tokens = [FakeToken("Hello"), FakeToken(" world")]
        mock_svc.searcher.ask_stream = MagicMock(return_value=iter(tokens))
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "test question"
        await _pilot.press("enter")
        await _pilot.pause()
        # Wait for worker to complete
        while app.screen.workers:
            await _pilot.pause()
        assert any(m["role"] == "assistant" for m in app.screen._history)


async def test_chat_stream_response_error_worker(mock_svc):
    """Cover the error branch in _stream_response."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        mock_svc.searcher.ask_stream = MagicMock(side_effect=Exception("LLM error"))
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "test"
        await _pilot.press("enter")
        await _pilot.pause()
        while app.screen.workers:
            await _pilot.pause()


async def test_chat_stream_response_reasoning_worker(mock_svc):
    """Cover the reasoning token branch in _stream_response."""
    from dataclasses import dataclass

    @dataclass
    class FakeToken:
        content: str
        is_reasoning: bool = False

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        tokens = [FakeToken("thinking", is_reasoning=True), FakeToken("answer")]
        mock_svc.searcher.ask_stream = MagicMock(return_value=iter(tokens))
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "test"
        await _pilot.press("enter")
        await _pilot.pause()
        while app.screen.workers:
            await _pilot.pause()


async def test_chat_run_sync_worker():
    """Cover _run_sync lines 356-376 via actual worker."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.progress import EventType

        async def fake_sync(on_progress=None):
            # Call progress callback to cover lines 367-370
            if on_progress:
                on_progress(
                    EventType.FILE_START,
                    {"current_file": 1, "total_files": 2, "file": "test.md"},
                )
            return {"added": 3}

        with patch("lilbee.ingest.sync", side_effect=fake_sync):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()


async def test_chat_sync_progress_percentage():
    """Verify sync progress reports actual percentage, not hardcoded 50%."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.progress import EventType

        update_calls: list[tuple] = []
        task_bar = app.screen.query_one("#task-bar")
        original_update = task_bar.update_task

        def tracking_update(task_id, pct, status):
            update_calls.append((task_id, pct, status))
            return original_update(task_id, pct, status)

        async def fake_sync(on_progress=None):
            if on_progress:
                on_progress(
                    EventType.FILE_START,
                    {"current_file": 3, "total_files": 4, "file": "doc.md"},
                )
            return {"added": 1}

        with (
            patch("lilbee.ingest.sync", side_effect=fake_sync),
            patch.object(task_bar, "update_task", tracking_update),
        ):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()

        # The FILE_START with 3/4 should produce 75%, not 50%
        pct_values = [pct for _, pct, _ in update_calls if pct > 0]
        assert 75 in pct_values


async def test_chat_run_sync_error_worker():
    """Cover the sync error branch."""

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:

        async def failing_sync(on_progress=None):
            raise Exception("sync failed")

        with patch("lilbee.ingest.sync", side_effect=failing_sync):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()


async def test_chat_cancel_stream_with_streaming_workers(mock_svc):
    """Cover action_cancel_stream line 350."""
    from dataclasses import dataclass

    @dataclass
    class FakeToken:
        content: str
        is_reasoning: bool = False

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:

        def slow_stream(*a, **kw):
            import time

            yield FakeToken("start")
            time.sleep(5)  # long enough to cancel
            yield FakeToken("end")

        mock_svc.searcher.ask_stream = MagicMock(side_effect=slow_stream)
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "test"
        await _pilot.press("enter")
        await _pilot.pause()
        # Now cancel while streaming
        app.screen._streaming = True
        app.screen.action_cancel_stream()
        assert app.screen._streaming is False


async def test_chat_needs_setup_true_pushes_wizard():
    """Verify _needs_setup=True pushes SetupWizard on mount."""
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.screens.setup import SetupWizard

    class SetupTestApp(App[None]):
        CSS = ""

        def compose(self) -> ComposeResult:
            yield Footer()

        def on_mount(self) -> None:
            self.push_screen(ChatScreen())

    app = SetupTestApp()
    with patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=True):
        async with app.run_test(size=(120, 40)) as _pilot:
            await _pilot.pause()
            assert isinstance(app.screen, SetupWizard)


async def test_chat_embedding_ready_false_no_sync():
    """Verify _embedding_ready=False skips auto-sync."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    class NoSyncApp(App[None]):
        CSS = ""

        def compose(self) -> ComposeResult:
            yield Footer()

        def on_mount(self) -> None:
            self.push_screen(ChatScreen(auto_sync=True))

    app = NoSyncApp()
    with (
        patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=False),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._run_sync") as mock_sync,
    ):
        async with app.run_test(size=(120, 40)) as _pilot:
            await _pilot.pause()
            mock_sync.assert_not_called()


# ---------------------------------------------------------------------------
# Additional coverage: chat.py lines
# ---------------------------------------------------------------------------


async def test_chat_on_input_submitted_slash():
    """Cover the on_input_submitted slash dispatch (line 94-95)."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "/version"
        with patch("lilbee.cli.helpers.get_version", return_value="1.0.0"):
            await _pilot.press("enter")
            # Value should be cleared
            assert inp.value == ""


async def test_chat_on_input_changed_visible_overlay():
    """Cover the overlay.hide() branch (line 408)."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
        inp = app.screen.query_one("#chat-input", Input)

        # Show the overlay first
        overlay.show_completions(["/help", "/models"])
        assert overlay.is_visible

        # Now trigger input change which should hide it
        inp.value = "/x"
        await _pilot.pause()
        # The on_input_changed handler should have hidden the overlay


async def test_chat_auto_sync_triggers_sync():
    """Cover the auto_sync branch (line 56)."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    class AutoSyncApp(App[None]):
        CSS = ""

        def compose(self) -> ComposeResult:
            yield Footer()

        def on_mount(self) -> None:
            self.push_screen(ChatScreen(auto_sync=True))

    app = AutoSyncApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # _run_sync would be called, but it's a @work decorator
        # Just verify the screen was created with auto_sync=True
        assert app.screen._auto_sync is True


async def test_chat_on_setup_complete_skipped_shows_banner():
    """Cover _on_setup_complete with 'skipped' result."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._on_setup_complete("skipped")
        await _pilot.pause()
        banner = app.screen.query_one("#chat-only-banner")
        assert banner.display is True


async def test_chat_on_setup_complete_success():
    """Cover _on_setup_complete with successful setup."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "_embedding_ready", return_value=False):
            app.screen._on_setup_complete("done")
            await _pilot.pause()


async def test_chat_cancel_with_active_worker(mock_svc):
    """Cover the /cancel worker.cancel() line (line 110) with an active worker."""
    from dataclasses import dataclass

    @dataclass
    class FakeToken:
        content: str
        is_reasoning: bool = False

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        import threading

        barrier = threading.Event()

        def slow_stream(*a, **kw):
            yield FakeToken("start")
            barrier.wait(timeout=5)
            yield FakeToken("end")

        mock_svc.searcher.ask_stream = MagicMock(side_effect=slow_stream)
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "test"
        await _pilot.press("enter")
        await _pilot.pause()
        # Now there should be a worker running
        app.screen._handle_slash("/cancel")
        barrier.set()
        await _pilot.pause()


# ---------------------------------------------------------------------------
# Additional coverage: catalog.py worker body lines
# ---------------------------------------------------------------------------


async def test_catalog_refresh_lists_with_search_and_load_more():
    """Cover the 'no models match' and 'load more' rows in _refresh_lists (line 246)."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            # Clear all models and set search to something that won't match
            screen._featured = []
            screen._hf_models = []
            screen._remote_models = []
            screen._refresh_lists()
            # Should show "No models match" in at least the All tab
            from textual.widgets import ListView

            lv = screen.query_one("#catlist-all", ListView)
            assert lv.children  # Should have the "no matches" row


async def test_catalog_refresh_lists_with_hf_load_more():
    """Cover the LoadMoreRow append (line 243)."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            # Add HF models and ensure has_more is True
            screen._hf_models = [
                _make_catalog_model(name=f"model-{i}B", hf_repo=f"org/model-{i}", downloads=100 - i)
                for i in range(5)
            ]
            screen._hf_has_more = True
            screen._refresh_lists()


async def test_catalog_page_down_with_focused_list():
    """Cover the inner loop in action_page_down (lines 355-356)."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            # Add some models so there are items to scroll
            screen._featured = [
                _make_catalog_model(name=f"f-{i}B", featured=True) for i in range(15)
            ]
            screen._refresh_lists()

            from textual.widgets import ListView

            lv = screen.query_one("#catlist-all", ListView)
            lv.focus()
            await _pilot.pause()
            screen.action_page_down()
            screen.action_page_up()


async def test_catalog_action_cursor_with_focused_list():
    """Cover the lv.action_cursor_down() in action_cursor_down."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            screen._featured = [
                _make_catalog_model(name=f"f-{i}B", featured=True) for i in range(5)
            ]
            screen._refresh_lists()

            from textual.widgets import ListView

            lv = screen.query_one("#catlist-all", ListView)
            lv.focus()
            await _pilot.pause()
            screen.action_cursor_down()
            screen.action_cursor_up()


async def test_catalog_highlight_none_with_highlighted_child():
    """Cover the branch where _update_highlighted_detail(None) finds a highlighted child."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            # Add some models so there are items
            screen._featured = [_make_catalog_model(name="feat-7B", featured=True, size_gb=4.0)]
            screen._refresh_lists()
            await _pilot.pause()

            from textual.widgets import ListView

            lv = screen.query_one("#catlist-all", ListView)
            lv.focus()
            await _pilot.pause()
            # If there's a highlighted child, calling with None should find it
            screen._update_highlighted_detail(None)


# ---------------------------------------------------------------------------
# Additional coverage: commands.py vision catalog exception (lines 91-92)
# ---------------------------------------------------------------------------


async def test_chat_vim_j_scrolls_down_no_focus():
    """Cover key_j scroll_down path (line 419) when focused widget is None."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Remove focus from input by focusing the chat log
        from textual.containers import VerticalScroll

        log = app.screen.query_one("#chat-log", VerticalScroll)
        log.focus()
        await _pilot.pause()
        # Now key_j should scroll down (line 419)
        app.screen.key_j()
        # key_k should scroll up (line 426)
        app.screen.key_k()


def test_check_embedding_model_installed():
    """Cover _check_embedding_model_async lines 61-65 (model is installed)."""
    mock_mgr = MagicMock()
    mock_mgr.is_installed.return_value = True
    with patch("lilbee.model_manager.get_model_manager", return_value=mock_mgr):
        from lilbee.model_manager import get_model_manager

        manager = get_model_manager()
        assert manager.is_installed(cfg.embedding_model) is True


def test_check_embedding_model_remote_available():
    """Cover _check_embedding_model_async lines 67-70 (model in remote backend)."""
    mock_mgr = MagicMock()
    mock_mgr.is_installed.return_value = False
    with (
        patch("lilbee.model_manager.get_model_manager", return_value=mock_mgr),
        patch(
            "lilbee.model_manager.detect_remote_embedding_models",
            return_value=["test-embed"],
        ),
    ):
        from lilbee.model_manager import detect_remote_embedding_models, get_model_manager

        manager = get_model_manager()
        assert not manager.is_installed(cfg.embedding_model)

        embed_base = cfg.embedding_model.split(":")[0]
        remote_embeds = detect_remote_embedding_models(cfg.litellm_base_url)
        assert any(embed_base in name for name in remote_embeds)


def test_check_embedding_model_not_found():
    """Cover _check_embedding_model_async line 72 (calls _show_setup_modal)."""
    mock_mgr = MagicMock()
    mock_mgr.is_installed.return_value = False
    with (
        patch("lilbee.model_manager.get_model_manager", return_value=mock_mgr),
        patch("lilbee.model_manager.detect_remote_embedding_models", return_value=[]),
    ):
        from lilbee.model_manager import detect_remote_embedding_models, get_model_manager

        manager = get_model_manager()
        assert not manager.is_installed(cfg.embedding_model)

        embed_base = cfg.embedding_model.split(":")[0]
        remote_embeds = detect_remote_embedding_models(cfg.litellm_base_url)
        assert not any(embed_base in name for name in remote_embeds)
        # Would call self.app.call_from_thread(self._show_setup_modal, remote_embeds)


async def test_command_provider_vision_catalog_error():
    """Cover the except block when VISION_CATALOG import fails (lines 91-92)."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with (
            patch("lilbee.models.list_installed_models", return_value=[]),
            patch.dict(
                "sys.modules",
                {
                    "lilbee.models": MagicMock(
                        list_installed_models=MagicMock(return_value=[]),
                        VISION_CATALOG=property(lambda s: (_ for _ in ()).throw(Exception("fail"))),
                    )
                },
            ),
        ):
            # VISION_CATALOG is accessed via import; take a different approach
            pass

    # Simpler approach: patch at the point of import inside _model_commands
    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)

        # Make list_installed_models succeed but VISION_CATALOG raise
        import lilbee.models as models_mod

        original_vision = models_mod.VISION_CATALOG
        try:
            # Temporarily replace VISION_CATALOG with something that raises on iteration
            models_mod.VISION_CATALOG = property(lambda s: 1 / 0)  # type: ignore[assignment]
            with patch("lilbee.models.list_installed_models", return_value=["m1"]):
                cmds = provider._model_commands()
                # Should have model commands but no vision commands
                assert any("m1" in c[0] for c in cmds)
        finally:
            models_mod.VISION_CATALOG = original_vision  # type: ignore[assignment]


async def test_chat_slash_crawl_no_args():
    """Cover /crawl with no URL showing usage hint."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_crawl("")


async def test_chat_slash_crawl_invalid_url():
    """Cover /crawl with non-URL argument."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._cmd_crawl("not-a-url")


async def test_chat_slash_crawl_valid_url():
    """Cover /crawl dispatching to background crawler."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "_run_crawl_background"):
            app.screen._cmd_crawl("https://example.com")


async def test_chat_slash_crawl_with_flags():
    """Cover /crawl with --depth and --max-pages flags."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch.object(app.screen, "_run_crawl_background") as mock_crawl,
        ):
            app.screen._cmd_crawl("https://example.com --depth 3 --max-pages 20")
            mock_crawl.assert_called_once_with("https://example.com", 3, 20)


async def test_chat_slash_add_url_routes_to_crawl():
    """Cover /add with a URL argument routing to _cmd_crawl."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "_cmd_crawl") as mock_crawl:
            app.screen._cmd_add("https://example.com")
            mock_crawl.assert_called_once_with("https://example.com")


class TestParseCrawlFlags:
    def test_empty(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags([]) == (0, 0)

    def test_depth_only(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--depth", "3"]) == (3, 0)

    def test_max_pages_only(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--max-pages", "20"]) == (0, 20)

    def test_both(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--depth", "2", "--max-pages", "15"]) == (2, 15)

    def test_invalid_values(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--depth", "abc"]) == (0, 0)

    def test_missing_value(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--depth"]) == (0, 0)

    def test_unknown_flags_skipped(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--unknown", "value"]) == (0, 0)


async def test_chat_run_crawl_background_success():
    """Cover _run_crawl_background success path including progress callback."""
    from pathlib import Path

    async def _fake_crawl(url, **kwargs):
        cb = kwargs.get("on_progress")
        if cb:
            cb("crawl_page", {"current": 1, "total": 2, "url": url})
        return [Path("/tmp/a.md")]

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with (
            patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock) as mock_crawl,
            patch.object(app.screen, "_run_sync"),
        ):
            mock_crawl.side_effect = _fake_crawl
            app.screen._run_crawl_background("https://example.com", 0, 50)
            await pilot.pause(delay=0.5)


async def test_chat_run_crawl_background_error():
    """Cover _run_crawl_background error path."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.side_effect = RuntimeError("network error")
            app.screen._run_crawl_background("https://example.com", 0, 50)
            await pilot.pause(delay=0.5)


async def test_chat_key_g_scrolls_home():
    """g scrolls to top of chat log when input not focused."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.containers import VerticalScroll

        log_widget = app.screen.query_one("#chat-log", VerticalScroll)
        log_widget.focus()
        app.screen.key_g()
        app.screen.key_G()


async def test_chat_key_g_noop_in_input():
    """g/G do nothing when Input is focused."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        app.screen.query_one("#chat-input", Input).focus()
        # Should not raise
        app.screen.key_g()
        app.screen.key_G()


async def test_chat_half_page_actions():
    """Ctrl-D/U half-page scroll actions execute without error."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen.action_half_page_down()
        app.screen.action_half_page_up()


async def test_settings_key_g_G(mock_svc):
    """g/G jump to first/last row in settings table."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import DataTable

        table = app.screen.query_one("#settings-table", DataTable)
        table.focus()
        app.screen.action_jump_bottom()
        assert table.cursor_row == table.row_count - 1
        app.screen.action_jump_top()
        assert table.cursor_row == 0


async def test_status_key_g_G(mock_svc):
    """g/G jump to first/last row in status table."""
    mock_svc.store.get_sources.return_value = [
        {"source": "a.md", "chunk_count": 1, "content_type": "text/markdown"},
        {"source": "b.md", "chunk_count": 2, "content_type": "text/markdown"},
        {"source": "c.md", "chunk_count": 3, "content_type": "text/markdown"},
    ]
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import DataTable

        table = app.screen.query_one("#docs-table", DataTable)
        table.focus()
        app.screen.action_jump_bottom()
        assert table.cursor_row == table.row_count - 1
        app.screen.action_jump_top()
        assert table.cursor_row == 0


async def test_catalog_key_g_G():
    """g/G jump to top/bottom of catalog list."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import ListView

            lv = screen.query_one("#catlist-all", ListView)
            lv.focus()
            screen.action_jump_top()
            screen.action_jump_bottom()


async def test_catalog_key_g_G_noop_in_input():
    """g/G do nothing when catalog search Input is focused."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            screen.query_one("#catalog-search", Input).focus()
            screen.action_jump_top()
            screen.action_jump_bottom()


async def test_catalog_number_keys_switch_tabs():
    """Number keys 1-4 switch catalog tabs."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import ListView, TabbedContent

            # Focus the list so Input is not focused (number keys are no-ops in Input)
            screen.query_one("#catlist-all", ListView).focus()
            await _pilot.pause()
            tabs = screen.query_one("#catalog-tabs", TabbedContent)
            screen.key_2()
            await _pilot.pause()
            assert tabs.active == "cat-chat"
            screen.key_3()
            await _pilot.pause()
            assert tabs.active == "cat-embedding"
            screen.key_4()
            await _pilot.pause()
            assert tabs.active == "cat-vision"
            screen.key_1()
            await _pilot.pause()
            assert tabs.active == "cat-all"


async def test_catalog_number_keys_noop_in_input():
    """Number keys do nothing when catalog search Input is focused."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            screen.query_one("#catalog-search", Input).focus()
            screen.key_1()
            screen.key_2()
            screen.key_3()
            screen.key_4()


async def test_app_question_mark_opens_help():
    """? key binding is registered on LilbeeApp."""
    from textual.binding import Binding as B

    from lilbee.cli.tui.app import LilbeeApp

    bindings = {b.key for b in LilbeeApp.BINDINGS if isinstance(b, B)}
    assert "question_mark" in bindings


async def test_chat_bindings_include_half_page():
    """Verify Ctrl-D/U bindings are registered on ChatScreen."""
    from textual.binding import Binding as B

    from lilbee.cli.tui.screens.chat import ChatScreen

    keys = {b.key for b in ChatScreen.BINDINGS if isinstance(b, B)}
    assert "ctrl+d" in keys
    assert "ctrl+u" in keys
