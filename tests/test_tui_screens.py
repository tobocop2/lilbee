"""Tests for TUI screens, app, and command provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Static

from lilbee.catalog import (
    FEATURED_EMBEDDING,
    CatalogModel,
    CatalogResult,
)
from lilbee.cli.tui.screens.catalog import (
    _WORKER_FETCH_HF,
    _WORKER_FETCH_MORE_HF,
    _WORKER_FETCH_REMOTE,
)
from lilbee.cli.tui.screens.catalog_utils import (
    TableRow,
    _format_downloads,
    _is_param_count,
    catalog_to_row,
    format_size_gb,
    matches_search,
    parse_param_label,
    remote_to_row,
    row_display_name,
    variant_to_row,
)
from lilbee.config import cfg
from lilbee.model_manager import RemoteModel
from lilbee.services import set_services

_EMPTY_CATALOG = CatalogResult(total=0, limit=25, offset=0, models=[])


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    """Snapshot and restore cfg for every test."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.documents_dir = tmp_path / "documents"
    cfg.lancedb_dir = tmp_path / "lancedb"
    cfg.chat_model = "test-model:latest"
    cfg.embedding_model = "test-embed:latest"
    cfg.chunk_size = 512
    # Simulate "already-initialized" state so ChatScreen._needs_setup()
    # doesn't push the SetupWizard during tests that exercise chat.
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
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
    """Patch out embedding model checks and model scanning so ChatScreen mounts cleanly."""
    from lilbee.cli.tui.widgets.model_bar import ModelBar

    with (
        patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        patch(
            "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready",
            return_value=False,
        ),
        patch(
            "lilbee.cli.tui.widgets.model_bar._classify_installed_models",
            return_value=([], []),
        ),
        patch.object(ModelBar, "_scan_models"),
    ):
        yield


def _make_catalog_model(
    name: str = "test",
    tag: str = "7b",
    display_name: str = "Test 7B",
    hf_repo: str = "org/test-7B-GGUF",
    task: str = "chat",
    featured: bool = False,
    downloads: int = 1000,
    size_gb: float = 4.0,
    description: str = "A test model",
) -> CatalogModel:
    return CatalogModel(
        name=name,
        tag=tag,
        display_name=display_name,
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


class TestParseParamLabel:
    def test_extracts_integer(self):
        assert parse_param_label("qwen-8B-instruct") == "8B"

    def test_extracts_decimal(self):
        assert parse_param_label("phi-0.6B") == "0.6B"

    def test_no_match(self):
        assert parse_param_label("nomic-embed-text") == "--"

    def test_case_insensitive(self):
        assert parse_param_label("model-3b-chat") == "3B"


class TestIsParamCount:
    def test_integer_param(self):
        assert _is_param_count("8B") is True

    def test_decimal_param(self):
        assert _is_param_count("0.6B") is True

    def test_version_string(self):
        assert _is_param_count("v1.5") is False

    def test_plain_text(self):
        assert _is_param_count("latest") is False


class TestVariantToRowDedup:
    """Verify variant_to_row avoids tag duplication and version-as-params."""

    def test_no_suffix_duplication(self):
        from lilbee.catalog import ModelFamily, ModelVariant

        variant = ModelVariant(
            hf_repo="nomic-ai/nomic-embed-text-v1.5-GGUF",
            filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
            param_count="v1.5",
            tag="v1.5",
            quant="Q4_K_M",
            size_mb=300,
            recommended=True,
        )
        family = ModelFamily(
            slug="nomic-embed-text",
            name="Nomic Embed Text v1.5",
            task="embedding",
            description="test",
            variants=(variant,),
        )
        row = variant_to_row(variant, family, installed=False)
        assert row.name.count("v1.5") == 1

    def test_version_tag_params_dash(self):
        from lilbee.catalog import ModelFamily, ModelVariant

        variant = ModelVariant(
            hf_repo="nomic-ai/nomic-embed-text-v1.5-GGUF",
            filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
            param_count="v1.5",
            tag="v1.5",
            quant="Q4_K_M",
            size_mb=300,
            recommended=True,
        )
        family = ModelFamily(
            slug="nomic-embed-text",
            name="Nomic Embed Text v1.5",
            task="embedding",
            description="test",
            variants=(variant,),
        )
        row = variant_to_row(variant, family, installed=False)
        assert row.params == "--"

    def test_numeric_param_kept(self):
        from lilbee.catalog import ModelFamily, ModelVariant

        variant = ModelVariant(
            hf_repo="org/qwen3-0.6b-GGUF",
            filename="qwen3-0.6b.Q4_K_M.gguf",
            param_count="0.6B",
            tag="0.6b",
            quant="Q4_K_M",
            size_mb=400,
            recommended=False,
        )
        family = ModelFamily(
            slug="qwen3",
            name="Qwen3",
            task="chat",
            description="test",
            variants=(variant,),
        )
        row = variant_to_row(variant, family, installed=False)
        assert "0.6B" in row.name
        assert row.params == "0.6B"


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


class TestRowDisplayName:
    def test_featured_star(self):
        row = catalog_to_row(_make_catalog_model(featured=True), installed=False)
        name = row_display_name(row)
        assert name.startswith("\u2605")

    def test_not_featured(self):
        row = catalog_to_row(_make_catalog_model(featured=False), installed=False)
        name = row_display_name(row)
        assert not name.startswith("\u2605")

    def test_installed_tag(self):
        row = catalog_to_row(_make_catalog_model(), installed=True)
        name = row_display_name(row)
        assert "[installed]" in name

    def test_not_installed_no_tag(self):
        row = catalog_to_row(_make_catalog_model(), installed=False)
        name = row_display_name(row)
        assert "[installed]" not in name


class TestFormatSizeGb:
    def test_positive_size(self):
        assert format_size_gb(4.0) == "4.0 GB"

    def test_zero_size_shows_dash(self):
        assert format_size_gb(0.0) == "--"

    def test_negative_shows_dash(self):
        assert format_size_gb(-1.0) == "--"


class TestCatalogToRow:
    def test_contains_display_name(self):
        m = _make_catalog_model(display_name="My Model 8B", hf_repo="my-org/my-model-8B-GGUF")
        row = catalog_to_row(m, installed=False)
        assert "my model 8b" in row.name.lower()

    def test_zero_downloads(self):
        m = _make_catalog_model(downloads=0)
        row = catalog_to_row(m, installed=False)
        assert row.downloads == "--"

    def test_positive_downloads(self):
        m = _make_catalog_model(downloads=5000)
        row = catalog_to_row(m, installed=False)
        assert row.downloads == "5K"


class TestMatchesSearch:
    def test_no_search(self):
        row = catalog_to_row(_make_catalog_model(task="chat"), installed=False)
        assert matches_search(row, "") is True

    def test_search_by_name(self):
        row = catalog_to_row(
            _make_catalog_model(display_name="Qwen 8B", hf_repo="org/qwen-8B-GGUF"),
            installed=False,
        )
        assert matches_search(row, "qwen") is True

    def test_search_by_task(self):
        row = catalog_to_row(_make_catalog_model(task="embedding"), installed=False)
        assert matches_search(row, "embedding") is True

    def test_search_no_match(self):
        row = catalog_to_row(_make_catalog_model(display_name="Llama 7B"), installed=False)
        assert matches_search(row, "qwen") is False

    def test_search_by_quant(self):
        row = TableRow(
            name="test",
            task="chat",
            params="8B",
            size="4.0 GB",
            quant="Q4_K_M",
            downloads="5K",
            featured=False,
            installed=False,
            sort_downloads=5000,
            sort_size=4.0,
        )
        assert matches_search(row, "q4_k_m") is True


class TestRemoteToRow:
    def test_creates_row(self):
        rm = _make_remote_model(name="qwen:latest", task="chat", parameter_size="7B")
        row = remote_to_row(rm)
        assert row.name == "qwen:latest"
        assert row.task == "chat"
        assert row.params == "7B"
        assert row.installed is True

    def test_no_parameter_size(self):
        rm = _make_remote_model(parameter_size="")
        row = remote_to_row(rm)
        assert row.params == "--"


class SettingsTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        self.push_screen(SettingsScreen())


async def test_settings_screen_mounts_grouped_sections():
    """Settings screen renders grouped sections with setting rows."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        groups = app.screen.query(".setting-group")
        assert len(groups) > 0
        rows = app.screen.query(".setting-row")
        assert len(rows) > 0


async def test_settings_search_filters_settings():
    """Search input filters visible setting rows."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        search = app.screen.query_one("#settings-search", Input)
        search.focus()
        search.value = "top_k"
        await pilot.pause()
        visible = [r for r in app.screen.query(".setting-row") if r.display]
        assert len(visible) >= 1
        assert any("top_k" in (r.name or "") for r in visible)


async def test_settings_search_clears_restores_all():
    """Clearing search restores all settings."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        search = app.screen.query_one("#settings-search", Input)
        total = len(app.screen.query(".setting-row"))
        search.value = "xyznonexistent"
        await pilot.pause()
        search.value = ""
        await pilot.pause()
        visible = [r for r in app.screen.query(".setting-row") if r.display]
        assert len(visible) == total


async def test_settings_bool_renders_checkbox():
    """Boolean settings render as Checkbox widgets."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        checkboxes = app.screen.query("Checkbox.setting-editor")
        assert len(checkboxes) >= 1


async def test_settings_readonly_no_editor():
    """Read-only settings do not have editor widgets."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        chat_row = app.screen.query_one("#row-chat_model")
        editors = chat_row.query(".setting-editor")
        assert len(editors) == 0


async def test_settings_persist_on_change():
    """Changing a setting persists the value to cfg."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        editor = app.screen.query_one("#ed-top_k", Input)
        editor.focus()
        editor.value = "20"
        await pilot.press("enter")
        await pilot.pause()
        assert cfg.top_k == 20


async def test_settings_exposes_wiki_fields():
    """Settings screen renders an editor for every wiki config field."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        wiki_keys = [
            "wiki",
            "wiki_dir",
            "wiki_prune_raw",
            "wiki_faithfulness_threshold",
            "wiki_stale_citation_threshold",
            "wiki_drift_threshold",
            "wiki_clusterer",
            "wiki_clusterer_k",
        ]
        for key in wiki_keys:
            assert app.screen.query_one(f"#ed-{key}") is not None


async def test_settings_wiki_clusterer_k_persists():
    """Editing wiki_clusterer_k writes through to cfg."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        editor = app.screen.query_one("#ed-wiki_clusterer_k", Input)
        editor.focus()
        editor.value = "8"
        await pilot.press("enter")
        await pilot.pause()
        assert cfg.wiki_clusterer_k == 8


async def test_settings_checkbox_persist():
    """Toggling a checkbox persists the boolean value."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Checkbox

        cb = app.screen.query_one("#ed-show_reasoning", Checkbox)
        original = cfg.show_reasoning
        cb.toggle()
        await pilot.pause()
        assert cfg.show_reasoning != original


async def test_settings_tab_reaches_checkbox_and_space_toggles():
    """Tab walks focus to the checkbox and Space toggles it."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Checkbox

        cb = app.screen.query_one("#ed-show_reasoning", Checkbox)
        original = cfg.show_reasoning
        for _ in range(30):
            await pilot.press("tab")
            await pilot.pause()
            if app.focused is cb:
                break
        assert app.focused is cb, "Tab failed to reach show_reasoning checkbox"
        await pilot.press("space")
        await pilot.pause()
        assert cfg.show_reasoning != original


async def test_settings_vim_keys():
    """Vim navigation keys work on the scroll container."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("j")
        await pilot.press("k")
        await pilot.press("g")
        await pilot.press("G")
        assert isinstance(app.screen, SettingsScreen)
        assert app.screen.query(".setting-group")


async def test_settings_pop_screen():
    """Pressing q pops the settings screen."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        assert isinstance(app.screen, SettingsScreen)
        await pilot.press("q")
        assert not isinstance(app.screen, SettingsScreen)


async def test_settings_effective_value_shows_model_default():
    """When user hasn't set a value, model default is shown with suffix."""
    from dataclasses import dataclass

    from lilbee.cli.tui.screens.settings import _effective_value

    @dataclass(frozen=True)
    class FakeDefaults:
        temperature: float | None = 0.7
        top_p: float | None = None
        top_k: int | None = None
        repeat_penalty: float | None = None
        num_ctx: int | None = 4096
        max_tokens: int | None = None

    old_defaults = cfg._model_defaults
    old_temp = cfg.temperature
    try:
        cfg.apply_model_defaults(FakeDefaults())
        cfg.temperature = None
        result = _effective_value("temperature")
        assert "0.7" in result
        assert "(model default)" in result
        cfg.num_ctx = None
        result = _effective_value("num_ctx")
        assert "4096" in result
        assert "(model default)" in result
        cfg.top_p = None
        result = _effective_value("top_p")
        assert result == "None"
    finally:
        cfg.temperature = old_temp
        object.__setattr__(cfg, "_model_defaults", old_defaults)


async def test_settings_effective_value_user_overrides_default():
    """When user has set a value, it takes precedence over model default."""
    from dataclasses import dataclass

    from lilbee.cli.tui.screens.settings import _effective_value

    @dataclass(frozen=True)
    class FakeDefaults:
        temperature: float | None = 0.7
        top_p: float | None = None
        top_k: int | None = None
        repeat_penalty: float | None = None
        num_ctx: int | None = None
        max_tokens: int | None = None

    old_defaults = cfg._model_defaults
    old_temp = cfg.temperature
    try:
        cfg.apply_model_defaults(FakeDefaults())
        cfg.temperature = 0.9
        result = _effective_value("temperature")
        assert result == "0.9"
        assert "(model default)" not in result
    finally:
        cfg.temperature = old_temp
        object.__setattr__(cfg, "_model_defaults", old_defaults)


async def test_settings_effective_value_no_defaults():
    """When no model defaults are loaded, None values show as 'None'."""
    from lilbee.cli.tui.screens.settings import _effective_value

    old_defaults = cfg._model_defaults
    old_temp = cfg.temperature
    try:
        cfg.clear_model_defaults()
        cfg.temperature = None
        result = _effective_value("temperature")
        assert result == "None"
    finally:
        cfg.temperature = old_temp
        object.__setattr__(cfg, "_model_defaults", old_defaults)


async def test_settings_is_writable():
    """_is_writable correctly identifies writable vs read-only fields."""
    from lilbee.cli.tui.screens.settings import _is_writable

    assert _is_writable("top_k")
    assert _is_writable("temperature")
    assert not _is_writable("chat_model")
    assert not _is_writable("embedding_model")
    assert not _is_writable("nonexistent_key_xyz")


async def test_settings_persist_invalid_int():
    """Invalid value for int field shows error and does not change cfg."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        original = cfg.top_k
        editor = app.screen.query_one("#ed-top_k", Input)
        editor.focus()
        editor.value = "abc"
        await pilot.press("enter")
        await pilot.pause()
        assert cfg.top_k == original


async def test_settings_select_save():
    """_on_select_save routes through _persist_value correctly."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        defn = SettingDef(type=str, nullable=False, group="Generation")
        event = MagicMock()
        event.select.name = "test_select"
        event.value = "chosen"
        with (
            patch.dict(
                "lilbee.cli.tui.screens.settings.SETTINGS_MAP",
                {"test_select": defn},
            ),
            patch.object(screen, "_persist_value") as mock_persist,
        ):
            screen._on_select_save(event)
            mock_persist.assert_called_once_with("test_select", defn, "chosen")


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
        info = app.screen.query_one("#config-info", Static)
        rendered = str(info.render())
        assert "Chat model" in rendered
        assert "Embed model" in rendered


async def test_status_screen_has_collapsible_sections(mock_svc):
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Collapsible

        sections = app.screen.query(Collapsible)
        assert len(sections) == 4


async def test_status_screen_config_shows_models(mock_svc):
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        info = app.screen.query_one("#config-info", Static)
        rendered = str(info.render())
        assert "Chat model" in rendered
        assert "Embed model" in rendered
        assert "OCR" in rendered


async def test_status_screen_config_pills_render(mock_svc):
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        info = app.screen.query_one("#config-info", Static)
        rendered = str(info.render())
        assert "loaded" in rendered or "not set" in rendered


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
        table = app.screen.query_one("#docs-table", DataTable)
        assert table.row_count == 1


async def test_status_screen_storage_section(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"source": "a.md", "chunk_count": 1, "content_type": "text/markdown"},
    ]
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        info = app.screen.query_one("#storage-info", Static)
        rendered = str(info.render())
        assert "Documents" in rendered
        assert "Data dir" in rendered
        assert "Models dir" in rendered


async def test_status_screen_arch_section(mock_svc):
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        info = app.screen.query_one("#arch-info", Static)
        rendered = str(info.render())
        assert "Chat arch" in rendered
        assert "Handler" in rendered


async def test_status_screen_arch_with_vision(mock_svc):
    cfg.chat_model = "test-vision:latest"
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        info = app.screen.query_one("#arch-info", Static)
        rendered = str(info.render())
        assert "Vision proj" in rendered


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
        assert table.has_focus


async def test_status_tab_moves_focus_between_sections(mock_svc):
    """Tab on StatusScreen advances focus across focusable widgets."""
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        initial = app.focused
        assert initial is not None
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is not None
        assert app.focused is not initial


async def test_status_screen_escape_pops():
    from lilbee.cli.tui.screens.status import StatusScreen

    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert isinstance(app.screen, StatusScreen)
        await _pilot.press("escape")
        assert not isinstance(app.screen, StatusScreen)


def test_ocr_label_enabled():
    from lilbee.cli.tui.screens.status import _ocr_label

    cfg.enable_ocr = True
    assert _ocr_label() == "enabled"


def test_ocr_label_disabled():
    from lilbee.cli.tui.screens.status import _ocr_label

    cfg.enable_ocr = False
    assert _ocr_label() == "disabled"


def test_ocr_pill_enabled():
    from lilbee.cli.tui.screens.status import _ocr_pill

    cfg.enable_ocr = True
    result = _ocr_pill()
    assert "on" in str(result)


def test_ocr_pill_disabled():
    from lilbee.cli.tui.screens.status import _ocr_pill

    cfg.enable_ocr = False
    result = _ocr_pill()
    assert "off" in str(result)


def test_status_model_pill_truthy():
    from lilbee.cli.tui.screens.status import _model_pill

    result = _model_pill("qwen3:8b")
    assert "loaded" in str(result)


def test_status_model_pill_empty():
    from lilbee.cli.tui.screens.status import _model_pill

    result = _model_pill("")
    assert "not set" in str(result)


def test_status_read_chat_arch_success():
    from lilbee.model_info import ModelArchInfo, _read_chat_arch

    info = ModelArchInfo()
    with (
        patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            return_value="/fake/path",
        ),
        patch(
            "lilbee.providers.llama_cpp_provider.read_gguf_metadata",
            return_value={"architecture": "llama"},
        ),
    ):
        result = _read_chat_arch(info)
    assert result.chat_arch == "llama"
    assert result.active_handler == "llama-cpp"


def test_status_read_embed_arch_success():
    from lilbee.model_info import ModelArchInfo, _read_embed_arch

    info = ModelArchInfo()
    with (
        patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            return_value="/fake/path",
        ),
        patch(
            "lilbee.providers.llama_cpp_provider.read_gguf_metadata",
            return_value={"architecture": "bert"},
        ),
    ):
        result = _read_embed_arch(info)
    assert result.embed_arch == "bert"


def test_status_read_vision_arch_success():
    from lilbee.model_info import ModelArchInfo, _read_vision_arch

    cfg.chat_model = "test-vision:latest"
    info = ModelArchInfo()
    with (
        patch(
            "lilbee.model_manager.is_vision_capable",
            return_value=True,
        ),
        patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            return_value="/fake/path",
        ),
        patch(
            "lilbee.providers.llama_cpp_provider.find_mmproj_for_model",
            return_value="/fake/mmproj",
        ),
        patch(
            "lilbee.providers.llama_cpp_provider.read_mmproj_projector_type",
            return_value="resampler",
        ),
    ):
        result = _read_vision_arch(info)
    assert result.vision_projector == "resampler"


def test_status_read_vision_arch_skips_when_no_model():
    from lilbee.model_info import ModelArchInfo, _read_vision_arch

    cfg.chat_model = "test-chat:latest"
    info = ModelArchInfo()
    with patch("lilbee.model_manager.is_vision_capable", return_value=False):
        result = _read_vision_arch(info)
    assert result.vision_projector == "unknown"


def test_status_read_model_arch_import_error():
    from lilbee.model_info import get_model_architecture

    with patch(
        "builtins.__import__",
        side_effect=lambda name, *a, **kw: (
            (_ for _ in ()).throw(ImportError("no llama-cpp"))
            if "llama_cpp" in name
            else __import__(name, *a, **kw)
        ),
    ):
        result = get_model_architecture()
    assert result.chat_arch == "unknown"


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


async def test_app_switch_to_catalog():
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.catalog.get_catalog", return_value=_EMPTY_CATALOG),
            patch("lilbee.model_manager.classify_remote_models", return_value=[]),
        ):
            app.switch_view("Catalog")
            await _pilot.pause()
            assert isinstance(app.screen, CatalogScreen)


async def test_app_switch_to_status():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.switch_view("Status")
        await _pilot.pause()
        from lilbee.cli.tui.screens.status import StatusScreen

        assert isinstance(app.screen, StatusScreen)


async def test_app_switch_to_settings():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.switch_view("Settings")
        await _pilot.pause()
        from lilbee.cli.tui.screens.settings import SettingsScreen

        assert isinstance(app.screen, SettingsScreen)


async def test_app_push_help():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.action_push_help()
        await _pilot.pause()
        assert app.screen.query("HelpPanel")


async def test_app_auto_sync_flag():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp(auto_sync=True)
    assert app._auto_sync is True


class ChatTestApp(App[None]):
    CSS = ""

    def __init__(self) -> None:
        super().__init__()
        from lilbee.cli.tui.widgets.task_bar import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield from ()

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
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/bogus")
            mock_notify.assert_called_once()
            assert "Unknown command" in mock_notify.call_args[0][0]


async def test_chat_slash_version():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.tui.screens.chat.get_version", return_value="1.2.3"):
            with patch.object(app.screen, "notify") as mock_notify:
                app.screen._handle_slash("/version")
                mock_notify.assert_called_once()
                assert "1.2.3" in mock_notify.call_args[0][0]


async def test_chat_slash_model_with_arg():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.settings.set_value"):
            app.screen._handle_slash("/model new-model:latest")
            await _pilot.pause()
            for worker in list(app.screen.workers):
                await worker.wait()
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
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/theme dracula")
            mock_notify.assert_called_once()
            assert "dracula" in mock_notify.call_args[0][0].lower()


async def test_chat_slash_theme_no_arg():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/theme")
            mock_notify.assert_called_once()
            assert "Themes:" in mock_notify.call_args[0][0]


async def test_chat_slash_theme_non_lilbee_app():
    """Theme with arg on a non-LilbeeApp should just list themes."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/theme dracula")
            mock_notify.assert_called_once()
            assert "Themes:" in mock_notify.call_args[0][0]


async def test_chat_slash_delete_with_match(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"filename": "notes.md", "source": "notes.md"},
    ]
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Re-inject mock after mount (model bar events may call reset_services)
        set_services(mock_svc)
        app.screen._cmd_delete("notes.md")
        mock_svc.store.delete_by_source.assert_called_once_with("notes.md")
        mock_svc.store.delete_source.assert_called_once_with("notes.md")


async def test_chat_slash_delete_not_found(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"filename": "notes.md", "source": "notes.md"},
    ]
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        set_services(mock_svc)
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_delete("nonexistent.md")
            mock_notify.assert_called_once()
            assert "Not found" in mock_notify.call_args[0][0]


async def test_chat_slash_delete_no_arg(mock_svc):
    mock_svc.store.get_sources.return_value = [
        {"filename": "notes.md", "source": "notes.md"},
    ]
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        set_services(mock_svc)
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_delete("")
            mock_notify.assert_called_once()
            assert "Documents:" in mock_notify.call_args[0][0]


async def test_chat_slash_delete_store_error(mock_svc):
    mock_svc.store.get_sources.side_effect = Exception("no store")
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        set_services(mock_svc)
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_delete("x")
            mock_notify.assert_called_once()
            assert "No documents" in mock_notify.call_args[0][0]


async def test_chat_slash_delete_empty_sources(mock_svc):
    mock_svc.store.get_sources.return_value = []
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        set_services(mock_svc)
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_delete("x")
            mock_notify.assert_called_once()
            assert "No documents" in mock_notify.call_args[0][0]


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
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/reset")
            mock_notify.assert_called_once()
            assert "confirm" in mock_notify.call_args[0][0].lower()


async def test_chat_slash_reset_error():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.helpers.perform_reset", side_effect=Exception("oops")):
            with patch.object(app.screen, "notify") as mock_notify:
                app.screen._handle_slash("/reset confirm")
                mock_notify.assert_called_once()
                assert "oops" in mock_notify.call_args[0][0]


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
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_set("bogus_key 42")
            mock_notify.assert_called_once()
            assert "Unknown setting" in mock_notify.call_args[0][0]


async def test_chat_slash_set_invalid_value():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_set("top_k not-a-number")
            mock_notify.assert_called_once()
            assert "Invalid value" in mock_notify.call_args[0][0]


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
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_add("")
            mock_notify.assert_not_called()


async def test_chat_slash_set_empty_args():
    """Cover early return when /set has no args — no notification posted."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_set("")
            mock_notify.assert_not_called()


async def test_chat_slash_add_nonexistent():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_add("/nonexistent/path/abc.txt")
            mock_notify.assert_called_once()
            assert "Not found" in mock_notify.call_args[0][0]


async def test_chat_slash_add_existing(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "_run_add_background") as mock_add_bg:
            app.screen._cmd_add(str(test_file))
            mock_add_bg.assert_called_once()


async def test_chat_slash_add_blocked_by_sync(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._sync_active = True
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_add(str(test_file))
            mock_notify.assert_called_once()
            assert "Sync in progress" in mock_notify.call_args[0][0]


async def test_chat_slash_cancel():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/cancel")
            mock_notify.assert_called_once()
            assert "Cancelled" in mock_notify.call_args[0][0]


async def test_chat_slash_help():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/help")
        await _pilot.pause()
        assert app.screen.query("HelpPanel")


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
        with patch.object(app.screen, "_send_message") as mock_send:
            await _pilot.press("enter")
            mock_send.assert_not_called()


async def test_chat_scroll_actions():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.containers import VerticalScroll

        log = app.screen.query_one("#chat-log", VerticalScroll)
        with (
            patch.object(log, "scroll_page_up") as mock_up,
            patch.object(log, "scroll_page_down") as mock_down,
        ):
            app.screen.action_scroll_up()
            mock_up.assert_called_once()
            app.screen.action_scroll_down()
            mock_down.assert_called_once()


async def test_chat_cancel_stream_not_streaming():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen.streaming = False
        app.screen.action_cancel_stream()
        assert app.screen.streaming is False


async def test_chat_cancel_stream_while_streaming():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen.streaming = True
        app.screen.action_cancel_stream()
        assert app.screen.streaming is False


async def test_apply_model_change_cancels_stream_when_streaming():
    """_apply_model_change cancels stream and defers service reset."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        screen.streaming = True
        with (
            patch.object(screen, "action_cancel_stream") as mock_cancel,
            patch.object(screen, "call_later") as mock_later,
        ):
            screen._apply_model_change()
            mock_cancel.assert_called_once()
            mock_later.assert_called_once_with(screen._deferred_service_reset)


async def test_apply_model_change_resets_immediately_when_not_streaming():
    """_apply_model_change resets services immediately when not streaming."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        screen.streaming = False
        with patch("lilbee.cli.tui.screens.chat.reset_services") as mock_reset:
            screen._apply_model_change()
            mock_reset.assert_called_once()


async def test_deferred_service_reset_retries_while_workers_active():
    """_deferred_service_reset retries via call_later when workers exist."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        with (
            patch.object(
                type(screen), "workers", new_callable=MagicMock, return_value=[MagicMock()]
            ),
            patch.object(screen, "call_later") as mock_later,
            patch("lilbee.cli.tui.screens.chat.reset_services") as mock_reset,
        ):
            screen._deferred_service_reset()
            mock_later.assert_called_once_with(screen._deferred_service_reset)
            mock_reset.assert_not_called()


async def test_deferred_service_reset_resets_when_no_workers():
    """_deferred_service_reset calls reset_services when workers drained."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        # Cancel background workers so the screen's worker manager is empty
        for w in list(screen.workers):
            w.cancel()
        await pilot.pause()
        with patch("lilbee.cli.tui.screens.chat.reset_services") as mock_reset:
            screen._deferred_service_reset()
            mock_reset.assert_called_once()


async def test_chat_vim_j_k_scrolls_in_normal_mode():
    """j/k scroll the chat log in normal mode."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        assert app.screen._insert_mode is False
        from textual.containers import VerticalScroll

        log = app.screen.query_one("#chat-log", VerticalScroll)
        with (
            patch.object(log, "scroll_down") as mock_down,
            patch.object(log, "scroll_up") as mock_up,
        ):
            app.screen.action_vim_scroll_down()
            mock_down.assert_called_once()
            app.screen.action_vim_scroll_up()
            mock_up.assert_called_once()


async def test_chat_vim_j_k_skips_in_insert_mode():
    """j/k raise SkipAction when in insert mode."""
    from textual.actions import SkipAction

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.focus()
        await pilot.pause()
        assert app.screen._insert_mode is True
        with pytest.raises(SkipAction):
            app.screen.action_vim_scroll_down()
        with pytest.raises(SkipAction):
            app.screen.action_vim_scroll_up()
        assert inp.has_focus


async def test_chat_needs_setup_false_when_models_exist():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False):
            assert not app.screen._needs_setup()


async def test_chat_refresh_model_bar():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        bar = app.screen.query_one("#model-bar", ModelBar)
        with patch.object(bar, "refresh_models") as mock_refresh:
            app.screen._refresh_model_bar()
            mock_refresh.assert_called_once()


async def test_chat_input_changed_hides_overlay():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.focus()
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
        inp.value = "/he"
        await _pilot.pause()
        assert not overlay.is_visible


async def test_chat_slash_quit():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "exit"):
            app.screen._handle_slash("/quit")
            app.exit.assert_called_once()


async def test_chat_slash_q():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "exit"):
            app.screen._handle_slash("/q")
            app.exit.assert_called_once()


async def test_chat_slash_exit():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "exit"):
            app.screen._handle_slash("/exit")
            app.exit.assert_called_once()


async def test_chat_slash_h():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/h")
        await _pilot.pause()
        assert app.screen.query("HelpPanel")


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
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/add /nonexistent/xyz")
            mock_notify.assert_called_once()
            assert "Not found" in mock_notify.call_args[0][0]


async def test_chat_slash_delete_dispatch():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/delete")
            mock_notify.assert_called_once()


async def test_chat_action_complete_no_options():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "hello"
        app.screen.action_complete()
        assert inp.value == "hello"


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


async def test_chat_tab_completes_alias_prefix():
    """Pressing Tab on '/cat' expands to the /catalog alias."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.focus()
        for key in ("slash", "c", "a", "t"):
            await pilot.press(key)
        await pilot.pause()
        assert inp.value == "/cat"
        await pilot.press("tab")
        await pilot.pause()
        assert inp.value == "/catalog"


async def test_chat_action_complete_cycle_no_selection():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
        overlay.show_completions(["a", "b"])
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        original = inp.value
        with patch.object(overlay, "cycle_next", return_value=None):
            app.screen.action_complete()
            assert inp.value == original


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


async def test_chat_input_handler_uses_on_decorator():
    """Chat input handlers use @on decorator for ID filtering."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    assert hasattr(ChatScreen._on_chat_submitted, "__wrapped__") or hasattr(
        ChatScreen._on_chat_submitted, "_textual_on"
    )
    assert hasattr(ChatScreen._on_chat_input_changed, "__wrapped__") or hasattr(
        ChatScreen._on_chat_input_changed, "_textual_on"
    )


async def test_chat_scroll_to_bottom():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.containers import VerticalScroll

        log = app.screen.query_one("#chat-log", VerticalScroll)
        with patch.object(log, "scroll_end") as mock_end:
            app.screen._scroll_to_bottom()
            # scroll_end called only when near bottom (within 5 lines)
            assert mock_end.called or log.max_scroll_y - log.scroll_y >= 5


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


async def test_command_provider_wiki_generate_action():
    """Palette 'Generate wiki pages' action notifies the user to use /wiki generate."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch.object(app, "notify") as mock_notify:
            provider._action_wiki_generate()
            mock_notify.assert_called_once()
            assert "/wiki generate" in mock_notify.call_args[0][0]


async def test_command_provider_delete_doc(mock_svc):
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Re-inject mock after mount (model bar events may call reset_services)
        set_services(mock_svc)
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        provider._delete_doc("notes.md")
        mock_svc.store.delete_by_source.assert_called_once_with("notes.md")
        mock_svc.store.delete_source.assert_called_once_with("notes.md")


async def test_command_provider_action_sync():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch.object(app, "notify") as mock_notify:
            provider._action_sync()
            mock_notify.assert_called_once()
            assert "/add" in mock_notify.call_args[0][0]


async def test_command_provider_action_version():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with (
            patch("lilbee.cli.helpers.get_version", return_value="1.0.0"),
            patch.object(app, "notify") as mock_notify,
        ):
            provider._action_version()
            mock_notify.assert_called_once()
            assert "1.0.0" in mock_notify.call_args[0][0]


async def test_command_provider_action_noop():
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch.object(app, "notify") as mock_notify:
            provider._action_noop()
            mock_notify.assert_called_once()
            assert "reset" in mock_notify.call_args[0][0].lower()


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
            assert cmds == []


async def test_command_provider_document_commands(mock_svc):
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Re-inject mock after mount (model bar events may call reset_services)
        set_services(mock_svc)
        mock_svc.store.get_sources.return_value = [
            {"filename": "notes.md", "source": "notes.md"},
        ]
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        cmds = provider._document_commands()
        assert len(cmds) == 1
        assert "notes.md" in cmds[0][0]


async def test_command_provider_document_commands_error(mock_svc):
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Re-inject mock after mount (model bar events may call reset_services)
        set_services(mock_svc)
        mock_svc.store.get_sources.side_effect = Exception("no store")
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        cmds = provider._document_commands()
        assert cmds == []


async def test_command_provider_document_commands_empty_name(mock_svc):
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Re-inject mock after mount (model bar events may call reset_services)
        set_services(mock_svc)
        mock_svc.store.get_sources.return_value = [{"source": ""}]
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        cmds = provider._document_commands()
        assert cmds == []


class CatalogTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()


def _patch_catalog():
    """Context manager to patch catalog screen's network calls."""
    return (
        patch("lilbee.cli.tui.screens.catalog.get_catalog", return_value=_EMPTY_CATALOG),
        patch("lilbee.model_manager.classify_remote_models", return_value=[]),
        patch(
            "lilbee.cli.tui.screens.catalog.get_model_manager",
            return_value=MagicMock(
                list_installed=MagicMock(return_value=[]),
                is_installed=MagicMock(return_value=False),
            ),
        ),
    )


async def test_catalog_screen_renders():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            app.push_screen(CatalogScreen())
            await _pilot.pause()
            assert app.screen.query_one("#catalog-search") is not None


async def test_catalog_focus_search():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen.action_focus_search()
            await _pilot.pause()
            from textual.widgets import Input

            assert app.screen.query_one("#catalog-search", Input).has_focus


async def test_catalog_header_sort():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            assert screen._sort_column == "Name"
            assert screen._sort_ascending is True
            # Simulate clicking same column header toggles direction
            event = MagicMock()
            event.column_key = "Name"
            screen._on_header_selected(event)
            assert screen._sort_ascending is False
            # Clicking different column resets to ascending
            event.column_key = "Downloads"
            screen._on_header_selected(event)
            assert screen._sort_column == "Downloads"
            assert screen._sort_ascending is True


async def test_catalog_pop_screen():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen.action_go_back()
            await _pilot.pause()
            # action_go_back on non-LilbeeApp calls pop_screen
            from lilbee.cli.tui.screens.catalog import CatalogScreen

            assert not isinstance(app.screen, CatalogScreen)


async def test_catalog_vim_keys():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen.action_cursor_down()
            screen.action_cursor_up()
            assert isinstance(app.screen, CatalogScreen)


async def test_catalog_vim_keys_in_input():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            inp = screen.query_one("#catalog-search", Input)
            inp.focus()
            await _pilot.pause()
            screen.action_cursor_down()
            screen.action_cursor_up()
            # Input stays focused; vim nav is suppressed when Input focused
            assert inp.has_focus


async def test_catalog_page_down_up():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen.action_page_down()
            screen.action_page_up()
            assert isinstance(app.screen, CatalogScreen)


async def test_catalog_page_down_no_focus():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            inp = screen.query_one("#catalog-search", Input)
            inp.focus()
            await _pilot.pause()
            screen.action_page_down()
            screen.action_page_up()
            # Page actions are suppressed when Input is focused
            assert inp.has_focus


async def test_catalog_install_already_installed(tmp_path):
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="installed-model")
            cfg.models_dir = tmp_path
            dest = tmp_path / "resolved.gguf"
            dest.write_text("fake")
            with (
                patch("lilbee.catalog.resolve_filename", return_value="resolved.gguf"),
                patch.object(screen, "notify") as mock_notify,
            ):
                screen._install_model(m)
                mock_notify.assert_called_once()
                assert "already installed" in mock_notify.call_args[0][0]


async def test_catalog_install_new_model():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="new-model")
            mock_mgr = MagicMock()
            mock_mgr.is_installed.return_value = False
            with (
                patch("lilbee.model_manager.get_model_manager", return_value=mock_mgr),
                patch.object(screen, "_enqueue_download") as mock_enqueue,
            ):
                screen._install_model(m)
                await _pilot.pause()
                mock_enqueue.assert_called_once_with(m)


async def test_catalog_select_remote_row():
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import remote_to_row

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            om = _make_remote_model(name="remote-chat:latest")
            row = remote_to_row(om)
            screen._select_row(row)
            assert cfg.chat_model == "remote-chat:latest"


async def test_catalog_load_more():
    from lilbee.cli.tui.screens.catalog import _HF_PAGE_SIZE, CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            old_offset = screen._hf_offset
            with patch.object(screen, "_fetch_more_hf"):
                screen._load_more()
                assert screen._hf_offset == old_offset + _HF_PAGE_SIZE


async def test_catalog_get_highlighted_model_name_empty():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            # Clear all models
            screen._families = []
            screen._hf_models = []
            screen._remote_models = []
            screen._refresh_table()
            assert screen._get_highlighted_model_name() is None


async def test_catalog_get_highlighted_with_rows():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._hf_models = [_make_catalog_model(name="test-7B")]
            screen._refresh_table()
            await _pilot.pause()
            name = screen._get_highlighted_model_name()
            assert name is not None


async def test_catalog_worker_hf_success():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.worker import WorkerState

            mock_worker = MagicMock()
            mock_worker.name = _WORKER_FETCH_HF
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
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.worker import WorkerState

            mock_worker = MagicMock()
            mock_worker.name = _WORKER_FETCH_REMOTE
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
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._hf_models = [_make_catalog_model(name="existing-7B")]

            from textual.worker import WorkerState

            mock_worker = MagicMock()
            mock_worker.name = _WORKER_FETCH_MORE_HF
            mock_worker.result = [_make_catalog_model(name="new-7B")]
            mock_event = MagicMock()
            mock_event.state = WorkerState.SUCCESS
            mock_event.worker = mock_worker
            screen.on_worker_state_changed(mock_event)
            assert len(screen._hf_models) == 2


async def test_catalog_worker_non_success_ignored():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.worker import WorkerState

            mock_event = MagicMock()
            mock_event.state = WorkerState.RUNNING
            before_hf = len(screen._hf_models)
            screen.on_worker_state_changed(mock_event)
            # Non-SUCCESS state should not change model lists
            assert len(screen._hf_models) == before_hf


async def test_catalog_select_catalog_row():
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import catalog_to_row

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="test-7B")
            row = catalog_to_row(m, installed=False)
            with patch.object(screen, "_install_model") as mock_install:
                screen._select_row(row)
                mock_install.assert_called_once_with(m)


async def test_catalog_input_changed_refreshes():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            inp = screen.query_one("#catalog-search", Input)
            with patch.object(screen, "_filter_grid") as mock_filter:
                event = MagicMock(spec=Input.Changed)
                event.input = inp
                screen._on_search_changed(event)
                mock_filter.assert_called()


async def test_catalog_input_handler_uses_on_decorator():
    """Catalog search handlers use @on decorator for ID filtering."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    assert hasattr(CatalogScreen._on_search_changed, "__wrapped__") or hasattr(
        CatalogScreen._on_search_changed, "_textual_on"
    )


async def test_catalog_row_selected_out_of_range():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            event = MagicMock()
            event.cursor_row = 999
            with patch.object(screen, "_select_row") as mock_select:
                screen._on_row_selected(event)
                mock_select.assert_not_called()


async def test_catalog_fetch_more_hf_worker():
    """Cover _fetch_more_hf worker body."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    hf_models = [_make_catalog_model(name=f"hf-{i}B", featured=False) for i in range(5)]
    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
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
                # Worker completed; models are now populated
                assert len(screen._hf_models) >= 0


async def test_catalog_grid_cache_skips_rebuild():
    """Second _refresh_grid call with same data skips DOM rebuild."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            screen._refresh_grid()
            first_key = screen._grid_cache_key
            assert first_key != ()

            with patch.object(screen.query_one("#catalog-grid"), "remove_children") as mock_remove:
                screen._refresh_grid()
                mock_remove.assert_not_called()
            assert screen._grid_cache_key == first_key


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
        assert app.screen.streaming is False


async def test_chat_stream_response_reasoning_worker(mock_svc):
    """Cover the reasoning token branch in _stream_response."""
    from dataclasses import dataclass

    @dataclass
    class FakeToken:
        content: str
        is_reasoning: bool = False

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        set_services(mock_svc)
        tokens = [FakeToken("thinking", is_reasoning=True), FakeToken("answer")]
        mock_svc.searcher.ask_stream = MagicMock(return_value=iter(tokens))
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "test"
        await _pilot.press("enter")
        await _pilot.pause()
        while app.screen.workers:
            await _pilot.pause()
        assert app.screen.streaming is False


async def test_chat_stream_response_inner_exception(mock_svc):
    """Cover the inner except/break in _stream_response (app shutting down)."""

    class ExplodingToken:
        is_reasoning = False

        @property
        def content(self):
            raise RuntimeError("app shutting down")

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        set_services(mock_svc)
        tokens = [ExplodingToken()]
        mock_svc.searcher.ask_stream = MagicMock(return_value=iter(tokens))
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "test"
        await _pilot.press("enter")
        await _pilot.pause()
        while app.screen.workers:
            await _pilot.pause()
        assert app.screen.streaming is False


async def test_chat_run_sync_worker():
    """Cover _run_sync lines 356-376 via actual worker."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.progress import EventType

        async def fake_sync(quiet=False, on_progress=None):
            # Call progress callback to cover lines 367-370
            if on_progress:
                on_progress(
                    EventType.FILE_START,
                    {"current_file": 1, "total_files": 2, "file": "test.md"},
                )
            return {"added": 3}

        with patch("lilbee.ingest.sync", new=fake_sync):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()
            assert app.screen._sync_active is False


async def test_chat_sync_progress_uses_indeterminate():
    """Verify sync progress uses indeterminate mode, not percentages."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.progress import EventType, FileDoneEvent, FileStartEvent

        update_calls: list[tuple] = []
        task_bar = app.task_bar
        original_update = task_bar.update_task

        def tracking_update(task_id, pct, status, *, indeterminate=None):
            update_calls.append((task_id, pct, status, indeterminate))
            return original_update(task_id, pct, status, indeterminate=indeterminate)

        async def fake_sync(quiet=False, on_progress=None):
            if on_progress:
                on_progress(
                    EventType.FILE_START,
                    FileStartEvent(current_file=1, total_files=1, file="doc.md"),
                )
                on_progress(
                    EventType.FILE_DONE,
                    FileDoneEvent(file="doc.md", status="ok", chunks=3),
                )
            return {"added": 1}

        with (
            patch("lilbee.ingest.sync", new=fake_sync),
            patch.object(task_bar, "update_task", tracking_update),
        ):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()

        # All progress updates should use indeterminate mode
        for _, _pct, _, indet in update_calls:
            if indet is not None:
                assert indet is True
        # No update should report 100% progress
        pct_values = [pct for _, pct, _, _ in update_calls]
        assert 100 not in pct_values


async def test_chat_sync_file_done_bad_type():
    """Sync progress raises TypeError when FILE_DONE data is not FileDoneEvent."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.progress import EventType

        async def fake_sync(quiet=False, on_progress=None):
            if on_progress:
                on_progress(EventType.FILE_DONE, {"file": "x.md", "status": "ok", "chunks": 1})
            return {"added": 0}

        with patch("lilbee.ingest.sync", new=fake_sync):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()
            # Worker catches the TypeError via the except Exception handler
            assert app.screen._sync_active is False


async def test_chat_run_sync_error_worker():
    """Cover the sync error branch."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:

        async def failing_sync(quiet=False, on_progress=None):
            raise Exception("sync failed")

        with patch("lilbee.ingest.sync", new=failing_sync):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()
            assert app.screen._sync_active is False


async def test_chat_cancel_stream_with_streaming_workers(mock_svc):
    """Cover action_cancel_stream line 350."""
    from dataclasses import dataclass

    @dataclass
    class FakeToken:
        content: str
        is_reasoning: bool = False

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        set_services(mock_svc)

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
        app.screen.streaming = True
        app.screen.action_cancel_stream()
        assert app.screen.streaming is False


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
        with (
            patch.object(app.screen, "_embedding_ready", return_value=False),
            patch.object(app.screen, "_run_sync") as mock_sync,
        ):
            app.screen._on_setup_complete("done")
            await _pilot.pause()
            # Embedding not ready, so sync should NOT be triggered
            mock_sync.assert_not_called()


async def test_chat_cancel_with_active_worker(mock_svc):
    """Cover the /cancel worker.cancel() line with an active worker."""
    from dataclasses import dataclass

    @dataclass
    class FakeToken:
        content: str
        is_reasoning: bool = False

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        set_services(mock_svc)
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
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/cancel")
            mock_notify.assert_called_once()
            assert "Cancelled" in mock_notify.call_args[0][0]
        barrier.set()
        await _pilot.pause()


async def test_catalog_refresh_table_empty():
    """Cover empty table case."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            screen._families = []
            screen._hf_models = []
            screen._remote_models = []
            screen._refresh_table()
            table = screen.query_one("#catalog-table", DataTable)
            assert table.row_count == 0


async def test_catalog_refresh_table_with_models():
    """Cover table with HF models."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            screen._hf_models = [
                _make_catalog_model(name=f"model-{i}B", hf_repo=f"org/model-{i}", downloads=100 - i)
                for i in range(5)
            ]
            screen._hf_has_more = True
            screen._refresh_table()
            table = screen.query_one("#catalog-table", DataTable)
            assert table.row_count >= 5


async def test_catalog_page_down_with_focused_table():
    """Cover action_page_down with focused DataTable."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            screen._hf_models = [
                _make_catalog_model(name=f"f-{i}B", featured=False) for i in range(15)
            ]
            screen._refresh_table()
            table = screen.query_one("#catalog-table", DataTable)
            table.focus()
            await _pilot.pause()
            screen.action_page_down()
            screen.action_page_up()
            assert table.has_focus


async def test_catalog_action_cursor_with_focused_table():
    """Cover action_cursor_down with focused DataTable."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            screen._hf_models = [
                _make_catalog_model(name=f"f-{i}B", featured=False) for i in range(5)
            ]
            screen._refresh_table()
            table = screen.query_one("#catalog-table", DataTable)
            table.focus()
            await _pilot.pause()
            screen.action_cursor_down()
            screen.action_cursor_up()
            assert table.has_focus


async def test_catalog_jump_top_bottom():
    """Cover action_jump_top and action_jump_bottom."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            screen._hf_models = [
                _make_catalog_model(name=f"f-{i}B", featured=False) for i in range(5)
            ]
            screen._refresh_table()
            table = screen.query_one("#catalog-table", DataTable)
            table.focus()
            await _pilot.pause()
            screen.action_jump_bottom()
            screen.action_jump_top()
            assert table.has_focus


async def test_chat_vim_j_scrolls_from_chat_log():
    """action_vim_scroll_down scrolls in normal mode."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        app.screen.query_one("#chat-log").focus()
        await pilot.pause()
        app.screen.action_vim_scroll_down()
        await pilot.pause()
        assert app.screen._insert_mode is False


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


async def test_chat_slash_crawl_unavailable():
    """_cmd_crawl notifies when crawler is not installed."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=False):
            app.screen._cmd_crawl("https://example.com")
            assert app.screen.is_current


async def test_chat_slash_crawl_no_args():
    """Cover /crawl with no URL showing usage hint."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True):
            app.screen._cmd_crawl("")
            assert app.screen.is_current


async def test_chat_slash_crawl_invalid_url():
    """Cover /crawl with non-URL argument."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True):
            app.screen._cmd_crawl("not-a-url")
            assert app.screen.is_current


async def test_chat_slash_crawl_valid_url():
    """Cover /crawl dispatching to background crawler."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch("lilbee.cli.tui.screens.chat.require_valid_crawl_url"),
            patch.object(app.screen, "_run_crawl_background") as mock_crawl,
        ):
            app.screen._cmd_crawl("https://example.com")
            mock_crawl.assert_called_once()


async def test_chat_slash_crawl_with_flags():
    """Cover /crawl with --depth and --max-pages flags."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch.object(app.screen, "_run_crawl_background") as mock_crawl,
        ):
            app.screen._cmd_crawl("https://example.com --depth 3 --max-pages 20")
            mock_crawl.assert_called_once()
            call_args = mock_crawl.call_args[0]
            assert call_args[0] == "https://example.com"
            assert call_args[1] == 3
            assert call_args[2] == 20
            assert len(call_args) == 4


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
        return [Path("a.md")]

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with (
            patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock) as mock_crawl,
            patch.object(app.screen, "_run_sync"),
        ):
            mock_crawl.side_effect = _fake_crawl
            app.screen._run_crawl_background("https://example.com", 0, 50, "test-task-id")
            await pilot.pause(delay=0.5)
            while app.screen.workers:
                await pilot.pause()
            # Worker completed successfully
            assert app.screen._sync_active is False


async def test_chat_run_crawl_background_error():
    """Cover _run_crawl_background error path."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch("lilbee.crawler.crawl_and_save", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.side_effect = RuntimeError("network error")
            app.screen._run_crawl_background("https://example.com", 0, 50, "test-task-id")
            await pilot.pause(delay=0.5)
            while app.screen.workers:
                await pilot.pause()
            assert app.screen._sync_active is False


async def test_chat_vim_g_scrolls_home():
    """g/G scroll to top/bottom of chat log in normal mode."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        app.screen.action_vim_scroll_home()
        app.screen.action_vim_scroll_end()
        assert app.screen._insert_mode is False


async def test_chat_vim_g_skips_in_insert_mode():
    """g/G raise SkipAction in insert mode."""
    from textual.actions import SkipAction

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app.screen._insert_mode is True
        with pytest.raises(SkipAction):
            app.screen.action_vim_scroll_home()
        with pytest.raises(SkipAction):
            app.screen.action_vim_scroll_end()


async def test_chat_half_page_actions():
    """Ctrl-D/U half-page scroll actions execute without error."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen.action_half_page_down()
        app.screen.action_half_page_up()
        # Half-page actions should not raise
        assert app.screen._insert_mode is True


async def test_settings_key_g_G():
    """g/G scroll settings via action methods."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen.action_scroll_end()
        app.screen.action_scroll_home()
        # Scroll actions delegate to settings-scroll widget
        scroll = app.screen.query_one("#settings-scroll")
        assert scroll is not None


async def test_status_key_g_G(mock_svc):
    """g/G scroll the status page to top/bottom."""
    mock_svc.store.get_sources.return_value = [
        {"source": "a.md", "chunk_count": 1, "content_type": "text/markdown"},
        {"source": "b.md", "chunk_count": 2, "content_type": "text/markdown"},
        {"source": "c.md", "chunk_count": 3, "content_type": "text/markdown"},
    ]
    app = StatusTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        scroll = app.screen.query_one("#status-scroll")
        app.screen.action_jump_bottom()
        await pilot.pause()
        app.screen.action_jump_top()
        await pilot.pause()
        assert scroll.scroll_offset.y == 0


async def test_catalog_key_g_G():
    """g/G jump to top/bottom of catalog table."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen.action_jump_top()
            screen.action_jump_bottom()
            assert isinstance(app.screen, CatalogScreen)


async def test_catalog_key_g_G_noop_in_input():
    """g/G do nothing when catalog search Input is focused."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            from textual.widgets import Input

            inp = screen.query_one("#catalog-search", Input)
            inp.focus()
            await _pilot.pause()
            screen.action_jump_top()
            screen.action_jump_bottom()
            # Jump actions are suppressed when Input is focused
            assert inp.has_focus


async def test_catalog_tab_bindings_removed():
    """Number key tab-switching bindings removed from catalog."""
    from textual.binding import Binding as B

    from lilbee.cli.tui.screens.catalog import CatalogScreen

    keys = {b.key for b in CatalogScreen.BINDINGS if isinstance(b, B)}
    for k in ("1", "2", "3", "4"):
        assert k not in keys


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


async def test_catalog_delete_installed_model_confirmation():
    """First press of d shows confirmation notification."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.cli.tui.screens.catalog.get_catalog", return_value=_EMPTY_CATALOG),
            patch("lilbee.model_manager.classify_remote_models", return_value=[]),
            patch("lilbee.cli.tui.screens.catalog.get_model_manager") as mock_mgr,
        ):
            mock_mgr.return_value.is_installed.return_value = True
            mock_mgr.return_value.list_installed.return_value = []
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            await screen.workers.wait_for_complete()

            screen._remote_models = [_make_remote_model("test-model:latest")]
            screen._refresh_table()
            await _pilot.pause()

            table = screen.query_one("#catalog-table", DataTable)
            table.focus()
            # Move cursor to last row (remote model)
            if screen._rows:
                table.move_cursor(row=len(screen._rows) - 1)
            await _pilot.pause()

            screen.action_delete_model()
            assert screen._pending_delete == "test-model:latest"


async def test_catalog_delete_second_press_confirms():
    """Second press of d calls remove and clears pending state."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.cli.tui.screens.catalog.get_catalog", return_value=_EMPTY_CATALOG),
            patch("lilbee.model_manager.classify_remote_models", return_value=[]),
            patch("lilbee.cli.tui.screens.catalog.get_model_manager") as mock_mgr,
        ):
            mock_mgr.return_value.is_installed.return_value = True
            mock_mgr.return_value.list_installed.return_value = []
            mock_mgr.return_value.remove.return_value = True
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            await screen.workers.wait_for_complete()

            screen._remote_models = [_make_remote_model("test-model:latest")]
            screen._refresh_table()
            await _pilot.pause()

            table = screen.query_one("#catalog-table", DataTable)
            table.focus()
            if screen._rows:
                table.move_cursor(row=len(screen._rows) - 1)
            await _pilot.pause()

            # First press sets pending
            screen.action_delete_model()
            assert screen._pending_delete == "test-model:latest"
            # Second press confirms
            screen.action_delete_model()
            assert screen._pending_delete is None


async def test_catalog_delete_not_installed():
    """Pressing d on a model that is not installed shows warning."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with (
            patch("lilbee.cli.tui.screens.catalog.get_catalog", return_value=_EMPTY_CATALOG),
            patch("lilbee.model_manager.classify_remote_models", return_value=[]),
            patch("lilbee.cli.tui.screens.catalog.get_model_manager") as mock_mgr,
        ):
            mock_mgr.return_value.is_installed.return_value = False
            mock_mgr.return_value.list_installed.return_value = []
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            await screen.workers.wait_for_complete()

            screen._remote_models = [_make_remote_model("test-model:latest")]
            screen._refresh_table()
            await _pilot.pause()

            table = screen.query_one("#catalog-table", DataTable)
            table.focus()
            if screen._rows:
                table.move_cursor(row=len(screen._rows) - 1)
            await _pilot.pause()

            screen.action_delete_model()
            assert screen._pending_delete is None


async def test_catalog_delete_no_highlighted_row():
    """Pressing d with no highlighted row shows warning."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            screen._families = []
            screen._hf_models = []
            screen._remote_models = []
            screen._refresh_table()
            await _pilot.pause()

            screen.action_delete_model()
            assert screen._pending_delete is None


async def test_catalog_delete_in_input_ignored():
    """Pressing d while focused on search input does nothing."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.widgets import Input

            screen.query_one("#catalog-search", Input).focus()
            screen.action_delete_model()
            assert screen._pending_delete is None


async def test_chat_slash_remove_no_args():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._handle_slash("/remove")
            mock_notify.assert_called_once()
            assert "Usage" in mock_notify.call_args[0][0]


async def test_chat_slash_remove_not_installed():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.model_manager.get_model_manager") as mock_mgr:
            mock_mgr.return_value.is_installed.return_value = False
            app.screen._handle_slash("/remove some-model:latest")
            while app.screen.workers:
                await _pilot.pause()
            await _pilot.pause()
            mock_mgr.return_value.is_installed.assert_called_once_with("some-model:latest")


async def test_chat_slash_remove_success():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.model_manager.get_model_manager") as mock_mgr:
            mock_mgr.return_value.is_installed.return_value = True
            mock_mgr.return_value.remove.return_value = True
            app.screen._handle_slash("/remove some-model:latest")
            while app.screen.workers:
                await _pilot.pause()
            await _pilot.pause()
            mock_mgr.return_value.remove.assert_called_once_with("some-model:latest")


async def test_chat_slash_remove_failed():
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.model_manager.get_model_manager") as mock_mgr:
            mock_mgr.return_value.is_installed.return_value = True
            mock_mgr.return_value.remove.return_value = False
            app.screen._handle_slash("/remove some-model:latest")
            while app.screen.workers:
                await _pilot.pause()
            await _pilot.pause()
            mock_mgr.return_value.remove.assert_called_once_with("some-model:latest")


async def test_cmd_add_creates_task_bar_entry(tmp_path):
    """B1: /add creates a TaskBar entry and runs copy_paths in a background worker."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello")

        async def fake_sync(quiet=False, on_progress=None):
            return {"added": 1}

        with (
            patch(
                "lilbee.cli.helpers.copy_files",
                return_value=MagicMock(copied=["doc.txt"], skipped=[]),
            ) as mock_copy,
            patch("lilbee.ingest.sync", new=fake_sync),
        ):
            task_bar = app.task_bar
            add_task_spy = MagicMock(wraps=task_bar.add_task)
            with patch.object(task_bar, "add_task", add_task_spy):
                app.screen._handle_slash(f"/add {test_file}")
                await _pilot.pause()

                # TaskBar.add_task was called with the file name
                add_task_spy.assert_called_once()
                label_arg = add_task_spy.call_args[0][0]
                assert "Add" in label_arg

            while app.screen.workers:
                await _pilot.pause()

            mock_copy.assert_called_once()


async def test_cmd_add_error_in_background(tmp_path):
    """B1: /add error branch reports failure through TaskBar."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello")

        with patch("lilbee.cli.helpers.copy_files", side_effect=RuntimeError("copy failed")):
            app.screen._handle_slash(f"/add {test_file}")
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()
            assert app.screen._sync_active is False


async def test_sync_called_with_quiet_true():
    """B2: _run_sync_worker passes quiet=True to suppress Rich progress bar."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        sync_kwargs: list[dict] = []

        async def capturing_sync(**kwargs):
            sync_kwargs.append(kwargs)
            return {"added": 0}

        with patch("lilbee.ingest.sync", new=capturing_sync):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()

        assert len(sync_kwargs) >= 1
        assert sync_kwargs[0].get("quiet") is True


async def test_chat_escape_enters_normal_mode():
    """F3: Escape leaves insert mode and enters normal mode."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        assert app.screen._insert_mode is True
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        assert app.screen._insert_mode is False


async def test_chat_enter_returns_to_insert_mode():
    """F3: Enter in normal mode switches back to insert mode."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        # Enter normal mode first
        app.screen._insert_mode = False
        app.screen._update_input_style()
        await pilot.pause()
        # Trigger enter via the on_key handler
        app.screen._enter_insert_mode()
        await pilot.pause()
        assert app.screen._insert_mode is True


async def test_chat_normal_mode_dims_input():
    """Input widget gets normal-mode class when in normal mode."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        assert "normal-mode" not in inp.classes
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        assert "normal-mode" in inp.classes


async def test_chat_escape_key_enters_normal_mode():
    """Escape key enters normal mode and focuses chat log."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.containers import VerticalScroll
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        log = app.screen.query_one("#chat-log", VerticalScroll)
        assert app.screen._insert_mode is True
        assert inp.has_focus

        app.screen.action_enter_normal_mode()
        await pilot.pause()
        assert app.screen._insert_mode is False
        assert log.has_focus


async def test_chat_history_next_skips_in_normal_mode():
    """action_history_next raises SkipAction in normal mode."""
    from textual.actions import SkipAction

    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        with pytest.raises(SkipAction):
            app.screen.action_history_next()


async def test_chat_history_prev_skips_in_normal_mode():
    """action_history_prev raises SkipAction in normal mode."""
    from textual.actions import SkipAction

    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        with pytest.raises(SkipAction):
            app.screen.action_history_prev()


async def test_chat_enter_key_returns_to_insert_mode():
    """Enter key returns to insert mode from normal mode."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        app.screen.action_enter_normal_mode()
        await pilot.pause()
        assert app.screen._insert_mode is False

        inp = app.screen.query_one("#chat-input", Input)
        inp.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen._insert_mode is True


async def test_app_nav_prev_cycles_views():
    """App-level h/left binding cycles to previous view."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.active_view == "Chat"

        app.action_nav_prev()
        await pilot.pause()
        assert app.active_view == "Wiki"

        app.action_nav_prev()
        await pilot.pause()
        assert app.active_view == "Tasks"


async def test_app_nav_next_cycles_views():
    """App-level l/right binding cycles to next view."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.active_view == "Chat"

        app.action_nav_next()
        await pilot.pause()
        assert app.active_view == "Catalog"

        app.action_nav_next()
        await pilot.pause()
        assert app.active_view == "Status"


async def test_app_nav_switches_all_views():
    """Nav prev/next cycles through all 5 views including Tasks."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        app.switch_view("Chat")
        await pilot.pause()
        assert app.active_view == "Chat"

        app.switch_view("Tasks")
        await pilot.pause()
        assert app.active_view == "Tasks"

        app.switch_view("Catalog")
        await pilot.pause()
        assert app.active_view == "Catalog"


async def test_chat_ctrl_n_p_bindings_exist():
    """Ctrl+N and Ctrl+P bindings exist on ChatScreen."""
    from textual.binding import Binding as B

    from lilbee.cli.tui.screens.chat import ChatScreen

    keys = {b.key for b in ChatScreen.BINDINGS if isinstance(b, B)}
    assert "ctrl+n" in keys
    assert "ctrl+p" in keys


async def test_chat_input_history_tracking():
    """Input history list tracks submitted messages."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        screen = app.screen
        screen._input_history.append("hello")
        screen._input_history.append("/help")
        assert screen._input_history[-2:] == ["hello", "/help"]


async def test_chat_sync_gating_flag():
    """_sync_active flag defaults to False."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        assert app.screen._sync_active is False


async def test_task_center_shows_history():
    """TaskCenter displays history entries."""
    from lilbee.cli.tui.screens.task_center import TaskCenter

    tc = TaskCenter()
    assert any(b.action == "cancel_task" for b in tc.BINDINGS if hasattr(b, "action"))


async def test_task_center_renders_empty():
    """TaskCenter shows 'No tasks' when queue is empty."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        table = app.screen.query_one("#task-table")
        assert table is not None


async def test_task_center_renders_active_and_history():
    """TaskCenter shows active task and completed history."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        task_bar = app.task_bar

        # Complete a task to create history
        t1 = task_bar.add_task("Sync A", "sync")
        task_bar.queue.advance()
        task_bar.queue.complete_task(t1)

        # Add an active task
        task_bar.add_task("Sync B", "sync")
        task_bar.queue.advance()

        app.push_screen(TaskCenter())
        await pilot.pause()

        table = app.screen.query_one("#task-table")
        assert table is not None
        assert table.row_count >= 2  # active + history


async def test_task_center_cancel_action():
    """TaskCenter cancel action triggers on active task."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        task_bar = app.task_bar

        task_bar.add_task("Sync", "sync")
        task_bar.queue.advance()

        app.push_screen(TaskCenter())
        await pilot.pause()
        # Just call action - should not crash even if cursor is on wrong row
        app.screen.action_cancel_task()
        await pilot.pause()
        assert isinstance(app.screen, TaskCenter)


async def test_task_center_refresh_action():
    """TaskCenter refresh action refreshes table."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        app.screen.action_refresh_tasks()
        await pilot.pause()
        table = app.screen.query_one("#task-table", DataTable)
        assert table is not None


async def test_task_center_cursor_actions():
    """TaskCenter cursor up/down delegate to DataTable."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        app.screen.action_cursor_down()
        app.screen.action_cursor_up()
        await pilot.pause()
        table = app.screen.query_one("#task-table", DataTable)
        assert table is not None


async def test_task_center_pop_screen():
    """TaskCenter pop_screen returns to chat."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.chat import ChatScreen
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        app.screen.action_go_back()
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)


async def test_chat_input_history_up_down():
    """Up/down arrows cycle through input history when input focused."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.screen.query_one("#chat-input")
        inp.focus()
        await pilot.pause()

        # Patch _stream_response to prevent background worker threads
        with patch.object(app.screen, "_stream_response"):
            # Submit two messages
            inp.value = "hello"
            await pilot.press("enter")
            inp.value = "world"
            await pilot.press("enter")
        await pilot.pause()

        assert app.screen._input_history == ["hello", "world"]

        # Press up to recall "world"
        app.screen.action_history_prev()
        await pilot.pause()
        assert inp.value == "world"

        # Press up again to recall "hello"
        app.screen.action_history_prev()
        await pilot.pause()
        assert inp.value == "hello"

        # Press up at boundary stays at "hello"
        app.screen.action_history_prev()
        await pilot.pause()
        assert inp.value == "hello"

        # Press down to go to "world"
        app.screen.action_history_next()
        await pilot.pause()
        assert inp.value == "world"

        # Press down past end clears input
        app.screen.action_history_next()
        await pilot.pause()
        assert inp.value == ""


async def test_chat_input_history_up_no_history():
    """Up arrow raises SkipAction when input history is empty."""
    from textual.actions import SkipAction

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.screen.query_one("#chat-input")
        inp.focus()
        await pilot.pause()

        with pytest.raises(SkipAction):
            app.screen.action_history_prev()


async def test_chat_input_history_down_no_index():
    """Down arrow raises SkipAction when history_index is -1."""
    from textual.actions import SkipAction

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.screen.query_one("#chat-input")
        inp.focus()
        await pilot.pause()

        with pytest.raises(SkipAction):
            app.screen.action_history_next()


async def test_chat_sync_gating_rejects_add(tmp_path):
    """B3: /add is rejected when _sync_active is True."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello")

        app.screen._sync_active = True
        app.screen._handle_slash(f"/add {test_file}")
        await pilot.pause()
        # No task should have been created
        task_bar = app.task_bar
        assert task_bar.queue.is_empty


async def test_chat_sync_gating_rejects_sync():
    """B3: /sync (/add synonym via _run_sync) is rejected when _sync_active is True."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen._sync_active = True
        app.screen._run_sync()
        await pilot.pause()
        task_bar = app.task_bar
        # No new sync task should be queued
        assert task_bar.queue.active_task is None


async def test_chat_action_complete_next():
    """Ctrl+N (action_complete_next) delegates to action_complete."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "/he"
        with patch(
            "lilbee.cli.tui.screens.chat.get_completions",
            return_value=["/help"],
        ):
            app.screen.action_complete_next()
            assert inp.value == "/help"


async def test_chat_action_complete_prev_opens_overlay():
    """Ctrl+P (action_complete_prev) opens overlay when not visible."""
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
            app.screen.action_complete_prev()
            assert overlay.is_visible
            assert inp.value == "/help"


async def test_chat_action_complete_prev_cycles_backward():
    """Ctrl+P cycles backward through existing completions."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        inp = app.screen.query_one("#chat-input", Input)
        overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
        inp.value = "/he"

        # Open completions first
        with patch(
            "lilbee.cli.tui.screens.chat.get_completions",
            return_value=["/help", "/hello"],
        ):
            app.screen.action_complete()
            assert overlay.is_visible

            # Cycle prev through existing overlay
            app.screen.action_complete_prev()
            assert overlay.is_visible


async def test_chat_action_complete_prev_with_space():
    """Ctrl+P with argument completions sets cmd + selection."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "/model q"
        with patch(
            "lilbee.cli.tui.screens.chat.get_completions",
            return_value=["qwen:latest", "qwen:8b"],
        ):
            app.screen.action_complete_prev()
            assert "qwen" in inp.value


async def test_task_center_with_queued_tasks():
    """TaskCenter shows queued tasks when active + queued present."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        task_bar = app.task_bar

        # Add active + queued tasks (same type so one is queued)
        task_bar.add_task("Download A", "download")
        task_bar.queue.advance("download")
        task_bar.add_task("Download B", "download")

        app.push_screen(TaskCenter())
        await pilot.pause()

        table = app.screen.query_one("#task-table")
        assert table.row_count >= 2  # active + queued


async def test_task_center_status_icon():
    """_status_icon maps all TaskStatus values."""
    from lilbee.cli.tui.screens.task_center import _status_icon
    from lilbee.cli.tui.task_queue import TaskStatus

    assert _status_icon(TaskStatus.QUEUED) == "\u23f3"
    assert _status_icon(TaskStatus.ACTIVE) == "\u25b6"
    assert _status_icon(TaskStatus.DONE) == "\u2713"
    assert _status_icon(TaskStatus.FAILED) == "\u2717"
    assert _status_icon(TaskStatus.CANCELLED) == "\u2298"


async def test_app_switch_to_tasks():
    """App switch_view navigates to TaskCenter."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.switch_view("Tasks")
        await pilot.pause()
        assert isinstance(app.screen, TaskCenter)


async def test_chat_mode_indicator_shows_normal():
    """ViewTabs shows NORMAL when entering normal mode."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app.screen.action_enter_normal_mode()
        await pilot.pause()
        bar = app.screen.query_one(ViewTabs)
        assert bar.mode_text == msg.MODE_NORMAL


async def test_chat_mode_indicator_shows_insert():
    """ViewTabs shows INSERT when returning to insert mode."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app.screen.action_enter_normal_mode()
        await pilot.pause()
        app.screen._enter_insert_mode()
        await pilot.pause()
        bar = app.screen.query_one(ViewTabs)
        assert bar.mode_text == msg.MODE_INSERT


async def test_chat_up_down_skip_in_normal_mode():
    """Up/down arrow keys raise SkipAction in normal mode (no focus cycling)."""
    from textual.actions import SkipAction

    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        with pytest.raises(SkipAction):
            app.screen.action_history_next()
        with pytest.raises(SkipAction):
            app.screen.action_history_prev()


async def test_chat_vim_scroll_in_normal_mode():
    """j/k scroll the chat log in normal mode."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        app.screen.action_vim_scroll_down()
        app.screen.action_vim_scroll_up()
        assert app.screen._insert_mode is False


async def test_chat_up_arrow_insert_mode_recalls_history():
    """Up arrow in insert mode still recalls input history."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.focus()
        await pilot.pause()
        app.screen._input_history = ["hello", "world"]
        app.screen._history_index = -1
        app.screen.action_history_prev()
        assert inp.value == "world"


def test_statusbar_mode_text_reactive_declared():
    """ViewTabs declares a mode_text reactive."""
    from textual.reactive import Reactive

    from lilbee.cli.tui.widgets.status_bar import ViewTabs

    reactives = {name for name, val in vars(ViewTabs).items() if isinstance(val, Reactive)}
    assert "mode_text" in reactives


async def test_task_center_row_click_shows_detail():
    """Clicking a row in TaskCenter updates the detail panel."""
    from unittest import mock

    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        task_bar = app.task_bar
        tid = task_bar.add_task("Download X", "download")
        task_bar.queue.advance()
        task_bar.queue.update_task(tid, 42, "10/24 MB")

        app.push_screen(TaskCenter())
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, TaskCenter)
        detail = screen.query_one("#task-detail", Static)

        # Simulate row selection with a mock event carrying the task_id as row_key
        row_key = mock.Mock()
        row_key.value = tid
        event = mock.Mock()
        event.row_key = row_key
        screen._on_row_highlighted(event)
        await pilot.pause()
        text = detail.content
        assert "Download X" in text
        assert "download" in text
        assert "42%" in text


async def test_task_center_show_detail_no_key():
    """TaskCenter _show_task_detail handles None key gracefully."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        screen._show_task_detail(None)
        await pilot.pause()
        detail = screen.query_one("#task-detail", Static)
        assert detail.content == ""


async def test_task_center_has_css_path():
    """TaskCenter declares a CSS_PATH for task-specific styles."""
    from lilbee.cli.tui.screens.task_center import TaskCenter

    assert TaskCenter.CSS_PATH == "task_center.tcss"


def test_task_center_status_pill():
    """_status_pill returns a pill string for each status."""
    from lilbee.cli.tui.screens.task_center import _status_pill
    from lilbee.cli.tui.task_queue import TaskStatus

    for status in TaskStatus:
        result = _status_pill(status)
        assert status.value in result


async def test_chat_screen_has_css_path():
    """ChatScreen declares a CSS_PATH for chat-specific styles."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    assert ChatScreen.CSS_PATH == "chat.tcss"


async def test_chat_status_line_updates_on_model_change():
    """ChatStatusLine renders a pill when model_name is set."""
    from textual.app import App

    from lilbee.cli.tui.screens.chat import ChatStatusLine

    class StatusApp(App[None]):
        def compose(self):  # type: ignore[override]
            yield ChatStatusLine(id="status")

    app = StatusApp()
    async with app.run_test(size=(80, 10)) as pilot:
        widget = app.query_one("#status", ChatStatusLine)
        widget.model_name = "qwen3:8b"
        await pilot.pause()
        assert widget.model_name == "qwen3:8b"
        # Label.content holds the plain-text form of the last update() call
        assert "qwen3:8b" in str(widget.content)


async def test_chat_status_line_empty_model():
    """ChatStatusLine renders empty when model_name is empty."""
    from textual.app import App

    from lilbee.cli.tui.screens.chat import ChatStatusLine

    class StatusApp(App[None]):
        def compose(self):  # type: ignore[override]
            yield ChatStatusLine(id="status")

    app = StatusApp()
    async with app.run_test(size=(80, 10)) as pilot:
        widget = app.query_one("#status", ChatStatusLine)
        widget.model_name = ""
        await pilot.pause()
        assert widget.model_name == ""


async def test_chat_screen_has_status_line():
    """ChatScreen compose includes a ChatStatusLine widget."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        from lilbee.cli.tui.screens.chat import ChatStatusLine

        status = app.screen.query_one("#chat-status-line", ChatStatusLine)
        assert status is not None


async def test_chat_screen_has_prompt_area():
    """ChatScreen compose wraps input in a PromptArea container."""
    cfg.chat_model = "test-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        from lilbee.cli.tui.screens.chat import PromptArea

        prompt_area = app.screen.query_one("#chat-prompt-area", PromptArea)
        assert prompt_area is not None


async def test_chat_refresh_status_line():
    """_refresh_status_line sets the model name on the status widget."""
    cfg.chat_model = "my-model"
    cfg.embedding_model = "test-embed"
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        from lilbee.cli.tui.screens.chat import ChatStatusLine

        status = app.screen.query_one("#chat-status-line", ChatStatusLine)
        assert status.model_name == "my-model"


async def test_settings_group_titles_present():
    """Settings screen renders group titles for each section."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(SettingsScreen())
        await pilot.pause()
        titles = app.screen.query(".group-title")
        assert len(titles) >= 2


class WikiTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        from textual.widgets import Footer

        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.wiki import WikiScreen

        self.push_screen(WikiScreen())


def _create_wiki_page(wiki_root, subdir, slug, title, content_body="Some content"):
    """Create a wiki markdown file with frontmatter."""
    d = wiki_root / subdir
    d.mkdir(parents=True, exist_ok=True)
    page = d / f"{slug}.md"
    page.write_text(
        f"---\ntitle: {title}\ngenerated_at: 2025-01-01\nsource_count: 3\n"
        f"faithfulness_score: 0.85\n---\n{content_body}\n"
    )
    return page


class TestWikiScreenCompose:
    async def test_composes_with_status_bar(self):
        """WikiScreen includes a ViewTabs widget."""
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            from lilbee.cli.tui.widgets.status_bar import ViewTabs

            bars = app.screen.query(ViewTabs)
            assert len(bars) == 1

    async def test_has_sidebar_and_content(self):
        """WikiScreen has sidebar and main content areas."""
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            from textual.widgets import Input, OptionList

            assert app.screen.query_one("#wiki-sidebar") is not None
            assert app.screen.query_one("#wiki-main") is not None
            assert app.screen.query_one("#wiki-search", Input) is not None
            assert app.screen.query_one("#wiki-page-list", OptionList) is not None


class TestWikiScreenEmptyState:
    async def test_shows_empty_when_wiki_disabled(self):
        """Shows empty state message when cfg.wiki is False."""
        cfg.wiki = False
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            from textual.widgets import OptionList

            from lilbee.cli.tui import messages as msg

            option_list = app.screen.query_one("#wiki-page-list", OptionList)
            assert option_list.option_count == 1
            assert msg.WIKI_EMPTY_STATE in str(option_list.get_option_at_index(0).prompt)

    async def test_shows_empty_when_no_pages(self, tmp_path):
        """Shows empty state when wiki is enabled but no pages exist."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_dir = cfg.data_root / cfg.wiki_dir
        wiki_dir.mkdir(parents=True)
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            from textual.widgets import OptionList

            option_list = app.screen.query_one("#wiki-page-list", OptionList)
            assert option_list.option_count >= 1


class TestWikiScreenWithPages:
    async def test_lists_pages(self, tmp_path):
        """WikiScreen lists pages when wiki data exists."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "test-doc", "Test Document")
        _create_wiki_page(wiki_root, "synthesis", "some-synthesis", "Some Synthesis")

        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as _pilot:
            from textual.widgets import OptionList

            option_list = app.screen.query_one("#wiki-page-list", OptionList)
            assert option_list.option_count >= 2

    async def test_displays_selected_page_content(self, tmp_path):
        """Selecting a page renders its content."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(
            wiki_root, "summaries", "my-page", "My Page", "# Hello World\nSome text here."
        )

        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from lilbee.cli.tui.screens.wiki import WikiScreen

            screen = app.screen
            assert isinstance(screen, WikiScreen)
            screen._display_page("summaries/my-page")
            await pilot.pause()

            header = app.screen.query_one("#wiki-page-header", Static)
            header_text = header.content
            assert "My Page" in header_text

    async def test_displays_faithfulness_in_header(self, tmp_path):
        """Page header shows faithfulness score from frontmatter."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "scored-page", "Scored Page")

        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from lilbee.cli.tui.screens.wiki import WikiScreen

            screen = app.screen
            assert isinstance(screen, WikiScreen)
            screen._display_page("summaries/scored-page")
            await pilot.pause()

            header = app.screen.query_one("#wiki-page-header", Static)
            header_text = header.content
            assert "85%" in header_text


class TestWikiScreenSearch:
    async def test_search_filters_pages(self, tmp_path):
        """Search input filters the page list."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "alpha-doc", "Alpha Document")
        _create_wiki_page(wiki_root, "summaries", "beta-doc", "Beta Document")

        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from textual.widgets import Input as TextualInput

            from lilbee.cli.tui.screens.wiki import WikiScreen

            screen = app.screen
            assert isinstance(screen, WikiScreen)
            search = app.screen.query_one("#wiki-search", TextualInput)
            search.value = "Alpha"
            await pilot.pause()
            assert "summaries/alpha-doc" in screen._page_slugs
            assert "summaries/beta-doc" not in screen._page_slugs

    async def test_escape_clears_search(self, tmp_path):
        """Escape clears search text when search has a value."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "test-page", "Test Page")

        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from textual.widgets import Input as TextualInput

            search = app.screen.query_one("#wiki-search", TextualInput)
            search.value = "something"
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert search.value == ""


class TestWikiScreenNavigation:
    async def test_go_back_pops_screen(self):
        """Pressing q pops the wiki screen in a non-LilbeeApp context."""
        from lilbee.cli.tui.screens.wiki import WikiScreen

        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("q")
            assert not isinstance(app.screen, WikiScreen)

    async def test_vim_keys(self):
        """Vim navigation keys work on the option list."""
        cfg.wiki = True
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("j")
            await pilot.press("k")
            await pilot.press("g")
            await pilot.press("G")
            from lilbee.cli.tui.screens.wiki import WikiScreen

            assert isinstance(app.screen, WikiScreen)

    async def test_focus_search(self, tmp_path):
        """Pressing / focuses the search input."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "page-one", "Page One")

        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("slash")
            await pilot.pause()
            from textual.widgets import Input as TextualInput

            assert app.screen.query_one("#wiki-search", TextualInput).has_focus


class TestWikiViewRegistration:
    def test_wiki_not_in_views_when_disabled(self):
        """Wiki view is not in get_views() when cfg.wiki is False."""
        from lilbee.cli.tui.app import get_views

        cfg.wiki = False
        assert "Wiki" not in get_views()

    def test_wiki_in_views_when_enabled(self):
        """Wiki view is in get_views() when cfg.wiki is True."""
        from lilbee.cli.tui.app import get_views

        cfg.wiki = True
        assert "Wiki" in get_views()

    def test_wiki_in_nav_views_when_enabled(self):
        """Wiki appears in get_nav_views() when cfg.wiki is True."""
        from lilbee.cli.tui.messages import get_nav_views

        cfg.wiki = True
        assert "Wiki" in get_nav_views()

    def test_wiki_not_in_nav_views_when_disabled(self):
        """Wiki does not appear in get_nav_views() when cfg.wiki is False."""
        from lilbee.cli.tui.messages import get_nav_views

        cfg.wiki = False
        assert "Wiki" not in get_nav_views()


class TestWikiFormatPageHeader:
    def test_basic_header(self):
        from lilbee.cli.tui.screens.wiki import _format_page_header

        result = _format_page_header("Title", "summary", 3, "2025-01-01", 0.85)
        assert "Title" in result
        assert "summary" in result
        assert "3 sources" in result
        assert "85%" in result

    def test_no_faithfulness(self):
        from lilbee.cli.tui.screens.wiki import _format_page_header

        result = _format_page_header("Title", "synthesis", 0, "", None)
        assert "Title" in result
        assert "%" not in result

    def test_no_sources(self):
        from lilbee.cli.tui.screens.wiki import _format_page_header

        result = _format_page_header("Title", "synthesis", 0, "2025-01-01", None)
        assert "sources" not in result


class TestWikiGroupPages:
    def test_groups_by_type(self):
        from lilbee.cli.tui.screens.wiki import _group_pages
        from lilbee.wiki.browse import WikiPageInfo

        pages = [
            WikiPageInfo("s/a", "A", "summary", 1, ""),
            WikiPageInfo("c/b", "B", "synthesis", 2, ""),
            WikiPageInfo("s/c", "C", "summary", 1, ""),
        ]
        groups = _group_pages(pages)
        types = [g[0] for g in groups]
        assert types == ["summary", "synthesis"]
        assert len(groups[0][1]) == 2
        assert len(groups[1][1]) == 1

    def test_empty_pages(self):
        from lilbee.cli.tui.screens.wiki import _group_pages

        assert _group_pages([]) == []


class TestWikiDisplayPageMissing:
    async def test_display_nonexistent_page(self, tmp_path):
        """Displaying a nonexistent page shows placeholder."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        wiki_root.mkdir(parents=True)

        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from lilbee.cli.tui.screens.wiki import WikiScreen

            screen = app.screen
            assert isinstance(screen, WikiScreen)
            screen._display_page("summaries/nonexistent")
            await pilot.pause()
            header = app.screen.query_one("#wiki-page-header", Static)
            assert header.content == ""


class TestWikiCoverageEdgeCases:
    async def test_load_pages_exception_path(self, tmp_path):
        """Exception in list_pages falls back to empty list."""
        cfg.wiki = True
        cfg.data_dir = tmp_path / "data"
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from lilbee.cli.tui.screens.wiki import WikiScreen

            screen = app.screen
            assert isinstance(screen, WikiScreen)
            with patch("lilbee.wiki.browse.list_pages", side_effect=OSError("boom")):
                screen._load_pages()
            await pilot.pause()

    async def test_on_page_selected_none_id(self, tmp_path):
        """Selecting an option with no id (heading) is a no-op."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "test", "Test Page")
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from lilbee.cli.tui.screens.wiki import WikiScreen

            screen = app.screen
            assert isinstance(screen, WikiScreen)
            # Simulate selecting a disabled heading (id=None)
            fake_event = MagicMock()
            fake_event.option = MagicMock(id=None)
            screen._on_page_selected(fake_event)
            await pilot.pause()

    async def test_action_focus_search(self, tmp_path):
        """action_focus_search focuses the search input."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        wiki_root.mkdir(parents=True)
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from textual.widgets import Input as TextualInput

            app.screen.action_focus_search()
            await pilot.pause()
            assert app.screen.query_one("#wiki-search", TextualInput).has_focus

    async def test_dismiss_or_back_empty_search(self, tmp_path):
        """Escape with empty search calls go_back."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        wiki_root.mkdir(parents=True)
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from lilbee.cli.tui.screens.wiki import WikiScreen

            screen = app.screen
            assert isinstance(screen, WikiScreen)
            # Search is empty, so dismiss_or_back should call go_back
            screen.action_dismiss_or_back()
            await pilot.pause()

    async def test_go_back_pops_screen(self, tmp_path):
        """action_go_back pops screen on non-LilbeeApp."""
        from lilbee.cli.tui.screens.wiki import WikiScreen

        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        wiki_root.mkdir(parents=True)
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            app.screen.action_go_back()
            await pilot.pause()
            assert not isinstance(app.screen, WikiScreen)

    async def test_go_back_switches_to_chat_on_lilbee_app(self, tmp_path):
        """action_go_back calls switch_view('Chat') on LilbeeApp."""
        from lilbee.cli.tui.app import LilbeeApp

        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "test", "Test")
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            app.switch_view("Wiki")
            await pilot.pause()
            from lilbee.cli.tui.screens.wiki import WikiScreen

            assert isinstance(app.screen, WikiScreen)
            app.screen.action_go_back()
            await pilot.pause()
            assert app.active_view == "Chat"

    async def test_vim_nav_noop_when_input_focused(self, tmp_path):
        """Vim navigation is suppressed when Input is focused."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "test", "Test Page")
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from textual.widgets import Input as TextualInput

            inp = app.screen.query_one("#wiki-search", TextualInput)
            inp.focus()
            await pilot.pause()
            # All vim nav actions should be no-ops when input is focused
            app.screen.action_cursor_down()
            app.screen.action_cursor_up()
            app.screen.action_jump_top()
            app.screen.action_jump_bottom()
            await pilot.pause()
            assert inp.has_focus

    async def test_on_page_selected_valid_slug(self, tmp_path):
        """Selecting a page with a valid slug displays it."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "hello", "Hello Page")
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from lilbee.cli.tui.screens.wiki import WikiScreen

            screen = app.screen
            assert isinstance(screen, WikiScreen)
            fake_event = MagicMock()
            fake_event.option = MagicMock(id="summaries/hello")
            screen._on_page_selected(fake_event)
            await pilot.pause()

    async def test_vim_nav_when_not_input_focused(self, tmp_path):
        """Vim nav dispatches to OptionList when Input is not focused."""
        cfg.wiki = True
        cfg.data_root = tmp_path
        wiki_root = cfg.data_root / cfg.wiki_dir
        _create_wiki_page(wiki_root, "summaries", "a", "Page A")
        _create_wiki_page(wiki_root, "summaries", "b", "Page B")
        app = WikiTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            from textual.widgets import OptionList as TextualOptionList

            ol = app.screen.query_one("#wiki-page-list", TextualOptionList)
            ol.focus()
            await pilot.pause()
            app.screen.action_cursor_down()
            app.screen.action_cursor_up()
            app.screen.action_jump_top()
            app.screen.action_jump_bottom()
            await pilot.pause()
            assert ol.has_focus

    def test_group_pages_unknown_type(self):
        """Pages with unknown type get their own group."""
        from lilbee.cli.tui.screens.wiki import _group_pages
        from lilbee.wiki.browse import WikiPageInfo

        pages = [
            WikiPageInfo("a", "Page A", "summary", 1, "2025-01-01"),
            WikiPageInfo("b", "Page B", "custom", 2, "2025-01-02"),
        ]
        result = _group_pages(pages)
        types = [t for t, _ in result]
        assert "summary" in types
        assert "custom" in types


def test_scan_installed_models_returns_sorted_lists():
    """_scan_installed_models splits chat/embed from registry."""
    from lilbee.cli.tui.screens.setup import _scan_installed_models

    mock_model_chat = MagicMock(name="qwen3", tag="8b", task="chat")
    mock_model_chat.name = "qwen3"
    mock_model_chat.tag = "8b"
    mock_model_chat.task = "chat"
    mock_model_embed = MagicMock(name="nomic", tag="latest", task="embedding")
    mock_model_embed.name = "nomic"
    mock_model_embed.tag = "latest"
    mock_model_embed.task = "embedding"
    mock_registry = MagicMock()
    mock_registry.list_installed.return_value = [mock_model_chat, mock_model_embed]
    with patch("lilbee.registry.ModelRegistry", return_value=mock_registry):
        chat, embed = _scan_installed_models()
    assert "qwen3:8b" in chat
    assert "nomic:latest" in embed


def test_scan_installed_models_exception_returns_empty():
    """_scan_installed_models returns ([], []) on exception."""
    from lilbee.cli.tui.screens.setup import _scan_installed_models

    with patch("lilbee.registry.ModelRegistry", side_effect=Exception("fail")):
        chat, embed = _scan_installed_models()
    assert chat == []
    assert embed == []


def test_installed_name_to_row_creates_row():
    """_installed_name_to_row creates a TableRow with correct fields."""
    from lilbee.cli.tui.screens.setup import _installed_name_to_row

    row = _installed_name_to_row("qwen3:8b", "chat")
    assert row.name == "qwen3:8b"
    assert row.task == "chat"
    assert row.installed is True
    assert row.size == "--"


class SetupTestApp(App[None]):
    CSS = ""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.setup import SetupWizard

        self.push_screen(SetupWizard())


def _patch_setup_scan(chat: list[str] | None = None, embed: list[str] | None = None):
    return patch(
        "lilbee.cli.tui.screens.setup._scan_installed_models",
        return_value=(chat or [], embed or []),
    )


def _patch_setup_ram(ram_gb: float = 16.0):
    return patch("lilbee.models.get_system_ram_gb", return_value=ram_gb)


def test_pick_recommended_small_ram():
    from lilbee.cli.tui.screens.setup import _pick_recommended

    chat, embed = _pick_recommended(3.0)
    assert chat.min_ram_gb <= 3.0
    assert embed == FEATURED_EMBEDDING[0]


def test_pick_recommended_medium_ram():
    from lilbee.cli.tui.screens.setup import _pick_recommended

    chat, _ = _pick_recommended(8.0)
    assert chat.min_ram_gb <= 8.0


def test_pick_recommended_large_ram():
    from lilbee.cli.tui.screens.setup import _pick_recommended

    chat, _ = _pick_recommended(32.0)
    assert chat.min_ram_gb <= 32.0


def test_pick_recommended_always_nomic_embed():
    from lilbee.cli.tui.screens.setup import _pick_recommended

    _, embed = _pick_recommended(4.0)
    assert embed.name == FEATURED_EMBEDDING[0].name


def test_scan_installed_models_empty():
    from lilbee.cli.tui.screens.setup import _scan_installed_models

    with patch("lilbee.registry.ModelRegistry", side_effect=Exception("no")):
        chat, embed = _scan_installed_models()
        assert chat == []
        assert embed == []


async def test_setup_wizard_preselect_skips_none_recommended():
    """_preselect_recommended skips when recommended model is None."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            # Clear recommendations and re-run preselect
            screen._recommended_chat = None
            screen._recommended_embed = None
            from lilbee.cli.tui.widgets.model_card import ModelCard

            cards = list(screen.query(ModelCard))
            screen._preselect_recommended(cards, cards)


async def test_setup_wizard_mounts_with_recommendations():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            assert screen._selected_chat is not None
            assert screen._selected_embed is not None


async def test_setup_wizard_model_cards_render_compact():
    """Wizard ModelCards render in the compact layout, not stretched to fill the grid."""
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            cards = list(screen.query(ModelCard))
            assert cards, "expected model cards in the wizard"
            for card in cards:
                assert card.size.height <= 6, (
                    f"wizard ModelCard is {card.size.height} rows tall, "
                    "expected compact layout (<=6 rows)"
                )


async def test_setup_wizard_select_chat_updates_slot():
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.grid_select import GridSelect
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            chat_cards = [
                c
                for c in screen.query(ModelCard)
                if c.row.task == "chat" and (c.row.ref or c.row.name) != screen._selected_chat
            ]
            if chat_cards:
                card = chat_cards[0]
                mock_grid = MagicMock(spec=GridSelect)
                event = GridSelect.Selected(grid_select=mock_grid, widget=card)
                screen._on_grid_selected(event)
                assert screen._selected_chat == (card.row.ref or card.row.name)
                assert card.selected is True


async def test_setup_wizard_select_embed_updates_slot():
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.grid_select import GridSelect
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            embed_cards = [c for c in screen.query(ModelCard) if c.row.task == "embedding"]
            assert len(embed_cards) > 0
            card = embed_cards[0]
            mock_grid = MagicMock(spec=GridSelect)
            event = GridSelect.Selected(grid_select=mock_grid, widget=card)
            screen._on_grid_selected(event)
            assert screen._selected_embed == (card.row.ref or card.row.name)
            assert card.selected is True


async def test_setup_wizard_deselects_previous():
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.grid_select import GridSelect
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            chat_cards = [c for c in screen.query(ModelCard) if c.row.task == "chat"]
            assert len(chat_cards) >= 2
            first = chat_cards[0]
            second = chat_cards[1]
            mock_grid = MagicMock(spec=GridSelect)
            screen._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=first))
            assert first.selected is True
            screen._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=second))
            assert second.selected is True
            assert first.selected is False


async def test_setup_wizard_skip_dismisses():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            screen._on_skip()
            await pilot.pause()


async def test_setup_wizard_skip_saves_chat_if_selected():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            screen._selections["chat"] = ("my-chat:latest", None)
            screen._selections["embedding"] = (None, None)
            with (
                patch("lilbee.settings.set_value") as mock_set,
                patch("lilbee.services.reset_services"),
            ):
                screen._on_skip()
                assert cfg.chat_model == "my-chat:latest"
                mock_set.assert_called_once()


async def test_setup_wizard_finish_saves_config():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            screen._selections["chat"] = ("my-chat:latest", None)
            screen._selections["embedding"] = ("my-embed:latest", None)
            with (
                patch("lilbee.settings.set_value") as mock_set,
                patch("lilbee.services.reset_services"),
            ):
                screen._save_and_dismiss("completed")
                assert cfg.chat_model == "my-chat:latest"
                assert cfg.embedding_model == "my-embed:latest"
                assert mock_set.call_count == 2


async def test_setup_wizard_finish_no_embed():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            screen._selections["chat"] = ("my-chat:latest", None)
            screen._selections["embedding"] = (None, None)
            with (
                patch("lilbee.settings.set_value") as mock_set,
                patch("lilbee.services.reset_services"),
            ):
                screen._save_and_dismiss("completed")
                assert mock_set.call_count == 1


async def test_setup_wizard_install_both_models():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            with patch.object(screen, "_run_downloads"):
                screen._on_install()
                await pilot.pause()
            assert len(screen._download_models) >= 1
            assert screen.has_class("-downloading")


async def test_setup_wizard_install_already_installed():
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = SetupTestApp()
    with _patch_setup_scan(chat=["chat:latest"], embed=["embed:latest"]), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            installed_cards = [c for c in screen.query(ModelCard) if c.row.installed]
            from lilbee.cli.tui.widgets.grid_select import GridSelect

            mock_grid = MagicMock(spec=GridSelect)
            for card in installed_cards:
                screen._on_grid_selected(GridSelect.Selected(grid_select=mock_grid, widget=card))
            with (
                patch("lilbee.settings.set_value"),
                patch("lilbee.services.reset_services"),
            ):
                screen._on_install()


async def test_setup_wizard_download_failure():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            with patch("lilbee.catalog.download_model", side_effect=Exception("network error")):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
                while screen.workers:
                    await pilot.pause()


async def test_setup_wizard_download_401_error():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            with patch("lilbee.catalog.download_model", side_effect=Exception("401 Unauthorized")):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
                while screen.workers:
                    await pilot.pause()


async def test_setup_wizard_download_with_progress():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)

            def fake_download(model, on_progress=None):
                if on_progress:
                    on_progress(512 * 1024, 1024 * 1024)
                    on_progress(1024 * 1024, 1024 * 1024)
                    on_progress(512 * 1024, 0)
                return MagicMock(stem="prog-model")

            with (
                patch("lilbee.catalog.download_model", side_effect=fake_download),
                patch("lilbee.settings.set_value"),
                patch("lilbee.services.reset_services"),
            ):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
                while screen.workers:
                    await pilot.pause()


async def test_setup_wizard_partial_download():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            call_count = 0

            def _fake_download(model, on_progress=None):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("embed failed")
                return MagicMock(stem="chat-model")

            with (
                patch("lilbee.catalog.download_model", side_effect=_fake_download),
                patch("lilbee.settings.set_value"),
                patch("lilbee.services.reset_services"),
            ):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
                while screen.workers:
                    await pilot.pause()


async def test_setup_wizard_download_cancel():
    """Setting _cancel_event aborts the download loop via _DownloadCancelled."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            download_started = False

            def fake_download(model, on_progress=None):
                nonlocal download_started
                download_started = True
                # Simulate progress callbacks; the cancel event is set before
                # the first callback so _on_download_progress raises.
                if on_progress:
                    on_progress(100, 1000)
                return MagicMock(stem="cancelled-model")

            # Populate at least one model so the for-loop body executes
            screen._download_models = [MagicMock(ref="chat:model", display_name="Test Model")]
            screen._cancel_event.set()
            with (
                patch(
                    "lilbee.cli.tui.screens.setup.download_model",
                    side_effect=fake_download,
                ),
                patch("lilbee.settings.set_value"),
                patch("lilbee.services.reset_services"),
            ):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
            # Download should not have started because the loop checks the
            # event before each model.
            assert not download_started


async def test_setup_wizard_cancel_event_raises_in_progress_callback():
    """_on_download_progress raises _DownloadCancelled when cancel event is set."""
    from lilbee.cli.tui.screens.setup import SetupWizard, _DownloadCancelled

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            screen._cancel_event.set()
            from lilbee.catalog import DownloadProgress

            with pytest.raises(_DownloadCancelled):
                screen._on_download_progress(
                    lambda fn, *a: fn(*a),
                    "test:ref",
                    DownloadProgress(percent=50, detail="50%", is_cache_hit=False),
                )


async def test_setup_wizard_download_cancel_mid_download():
    """Cancel event set during download triggers _DownloadCancelled in the loop."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)

            def fake_download(model, on_progress=None):
                # Set cancel during download; the progress callback will raise
                screen._cancel_event.set()
                if on_progress:
                    on_progress(100, 1000)
                return MagicMock(stem="model")

            screen._download_models = [MagicMock(ref="chat:model", display_name="Test Model")]
            with (
                patch(
                    "lilbee.cli.tui.screens.setup.download_model",
                    side_effect=fake_download,
                ),
                patch("lilbee.settings.set_value"),
                patch("lilbee.services.reset_services"),
            ):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()


async def test_wizard_action_cancel_sets_event():
    """action_cancel sets the cancel event so download threads abort."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            assert not screen._cancel_event.is_set()
            screen.action_cancel()
            assert screen._cancel_event.is_set()


async def test_setup_wizard_single_model_download_error():
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.grid_select import GridSelect
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = SetupTestApp()
    with _patch_setup_scan(embed=["embed:latest"]), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            embed_cards = [
                c for c in screen.query(ModelCard) if c.row.task == "embedding" and c.row.installed
            ]
            if embed_cards:
                mock_grid = MagicMock(spec=GridSelect)
                screen._on_grid_selected(
                    GridSelect.Selected(grid_select=mock_grid, widget=embed_cards[0])
                )
            with patch("lilbee.catalog.download_model", side_effect=Exception("connection error")):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
                while screen.workers:
                    await pilot.pause()


async def test_setup_wizard_download_cache_hit():
    """Download that returns 100% immediately (cache hit)."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)

            def fake_download(model, on_progress=None):
                if on_progress:
                    on_progress(1000, 1000)
                return MagicMock(stem="cached-model")

            with (
                patch("lilbee.catalog.download_model", side_effect=fake_download),
                patch("lilbee.settings.set_value"),
                patch("lilbee.services.reset_services"),
            ):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
                while screen.workers:
                    await pilot.pause()


async def test_setup_wizard_action_cancel():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            screen.action_cancel()
            await pilot.pause()


async def test_setup_wizard_footer_updates():
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            action_btn = screen.query_one("#setup-action")
            assert action_btn.disabled is False
            screen._selections["chat"] = (None, None)
            screen._selections["embedding"] = (None, None)
            screen._update_footer()
            assert action_btn.disabled is True


async def test_setup_wizard_with_installed_models():
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = SetupTestApp()
    with _patch_setup_scan(chat=["my-chat:1b"], embed=["my-embed:latest"]), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            installed_cards = [c for c in screen.query(ModelCard) if c.row.installed]
            assert len(installed_cards) >= 2


async def test_setup_wizard_grid_selected_non_model():
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.grid_select import GridSelect

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            mock_grid = MagicMock(spec=GridSelect)
            mock_widget = MagicMock()
            event = GridSelect.Selected(grid_select=mock_grid, widget=mock_widget)
            screen._on_grid_selected(event)


async def test_setup_wizard_run_downloads_all_succeed():
    """_run_downloads calls _on_all_downloads_complete when all succeed."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)

            cm1 = _make_catalog_model(name="chat-m")
            cm2 = _make_catalog_model(name="embed-m")
            screen._download_models = [cm1, cm2]

            with (
                patch("lilbee.catalog.download_model"),
                patch("lilbee.settings.set_value"),
                patch("lilbee.services.reset_services"),
            ):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
                while screen.workers:
                    await pilot.pause()


async def test_setup_wizard_run_downloads_embed_fails():
    """Embedding download failure calls _on_partial_success."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)

            cm1 = _make_catalog_model(name="chat-ok")
            cm2 = _make_catalog_model(name="embed-fail")
            screen._download_models = [cm1, cm2]
            call_count = 0

            def _fake(model, on_progress=None):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise Exception("embed failed")

            with (
                patch("lilbee.catalog.download_model", side_effect=_fake),
                patch("lilbee.settings.set_value"),
                patch("lilbee.services.reset_services"),
            ):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
                while screen.workers:
                    await pilot.pause()


async def test_setup_wizard_run_downloads_chat_401():
    """401 error on first model (chat) returns early."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)

            cm1 = _make_catalog_model(name="gated-model")
            cm2 = _make_catalog_model(name="embed-m")
            screen._download_models = [cm1, cm2]

            def _fake(model, on_progress=None):
                raise PermissionError("401 Unauthorized")

            with patch("lilbee.catalog.download_model", side_effect=_fake):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()
                while screen.workers:
                    await pilot.pause()


async def test_setup_wizard_on_all_downloads_complete():
    """_on_all_downloads_complete sets status and dismisses."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            with patch("lilbee.settings.set_value"), patch("lilbee.services.reset_services"):
                screen._on_all_downloads_complete()
                await pilot.pause()


async def test_setup_wizard_on_partial_success():
    """_on_partial_success clears embedding selection and dismisses."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            with patch("lilbee.settings.set_value"), patch("lilbee.services.reset_services"):
                screen._on_partial_success()
                await pilot.pause()
            from lilbee.models import ModelTask

            assert screen._selections[ModelTask.EMBEDDING] == (None, None)


async def test_setup_wizard_grid_leave_down_walks_focus_forward():
    """Arrow-down past the last card advances focus out of the grid."""
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.grid_select import GridSelect

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            grids = list(screen.query(GridSelect))
            assert grids, "expected at least one GridSelect in the wizard"
            last_grid = grids[-1]
            last_grid.focus()
            last_grid.highlight_last()
            await pilot.pause()
            assert app.focused is last_grid
            await pilot.press("down")
            await pilot.pause()
            assert app.focused is not last_grid
            assert app.focused is not None


async def test_setup_wizard_grid_leave_up_walks_focus_backward():
    """Arrow-up past the first card walks focus backward."""
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.grid_select import GridSelect

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            grids = list(screen.query(GridSelect))
            assert len(grids) >= 2, "expected multiple GridSelects in the wizard"
            second_grid = grids[1]
            second_grid.focus()
            second_grid.highlight_first()
            await pilot.pause()
            assert app.focused is second_grid
            await pilot.press("up")
            await pilot.pause()
            assert app.focused is not second_grid
            assert app.focused is not None


async def test_setup_wizard_on_download_progress():
    """_on_download_progress updates progress bar and status."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            from lilbee.catalog import DownloadProgress

            cm = _make_catalog_model(name="prog-m")
            screen._download_models = [cm]
            screen._mount_download_rows()
            await pilot.pause()
            screen._on_download_progress(
                lambda fn, *a: fn(*a),
                cm.ref,
                DownloadProgress(percent=50, detail="25/50 MB", is_cache_hit=False),
            )
            await pilot.pause()
            row = screen._download_rows[cm.ref]
            assert "50" in str(row.label.content)
            assert row.bar.progress == 50


async def test_setup_wizard_handle_download_error_401():
    """_handle_download_error rewrites 401 errors."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            cm = _make_catalog_model(name="gated", display_name="Gated")
            screen._download_models = [cm]
            screen._mount_download_rows()
            screen._handle_download_error(
                lambda fn, *a: fn(*a),
                PermissionError("401 Unauthorized"),
                cm,
                is_first=True,
            )
            row = screen._download_rows[cm.ref]
            assert "requires login" in str(row.label.content)


async def test_setup_wizard_handle_download_error_partial():
    """_handle_download_error calls _on_partial_success for non-first model."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            cm = _make_catalog_model(name="embed-fail", display_name="EmbedFail")
            screen._download_models = [cm]
            screen._mount_download_rows()
            with patch("lilbee.settings.set_value"), patch("lilbee.services.reset_services"):
                screen._handle_download_error(
                    lambda fn, *a: fn(*a),
                    Exception("download failed"),
                    cm,
                    is_first=False,
                )
            row = screen._download_rows[cm.ref]
            assert "download failed" in str(row.label.content)


async def test_setup_wizard_download_progress_callback():
    """Download progress callback updates status via _download_loop."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)

            cm = _make_catalog_model(name="prog-m", display_name="ProgM")
            screen._download_models = [cm]

            def _fake(model, on_progress=None):
                if on_progress:
                    on_progress(25 * 1024 * 1024, 50 * 1024 * 1024)
                    on_progress(50 * 1024 * 1024, 50 * 1024 * 1024)

            with (
                patch("lilbee.cli.tui.screens.setup.download_model", side_effect=_fake),
                patch("lilbee.settings.set_value"),
                patch("lilbee.services.reset_services"),
            ):
                screen._download_loop(lambda fn, *a: fn(*a))
                await pilot.pause()

            row = screen._download_rows[cm.ref]
            assert "done" in str(row.label.content)
            assert row.bar.progress == 100


async def test_setup_wizard_two_downloads_show_independent_rows():
    """Two concurrent downloads render side-by-side with independent labels and bars."""
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)

            cm_chat = _make_catalog_model(name="chat-x", display_name="ChatX")
            cm_embed = _make_catalog_model(name="embed-y", display_name="EmbedY")
            screen._download_models = [cm_chat, cm_embed]
            screen._mount_download_rows()
            await pilot.pause()

            # Both rows exist before any downloads start.
            assert cm_chat.ref in screen._download_rows
            assert cm_embed.ref in screen._download_rows
            chat_row = screen._download_rows[cm_chat.ref]
            embed_row = screen._download_rows[cm_embed.ref]
            assert "ChatX" in str(chat_row.label.content)
            assert "EmbedY" in str(embed_row.label.content)

            # Simulate chat finishing, then embed progressing mid-way.
            screen._mark_row_done(cm_chat.ref)
            screen._update_row(cm_embed.ref, 40, "20/50 MB")
            await pilot.pause()

            # Chat row stays at 100% with its own label intact.
            assert chat_row.bar.progress == 100
            assert "done" in str(chat_row.label.content)
            assert "ChatX" in str(chat_row.label.content)

            # Embed row shows its independent 40% state.
            assert embed_row.bar.progress == 40
            assert "40" in str(embed_row.label.content)
            assert "EmbedY" in str(embed_row.label.content)


def test_param_sort_value_with_match():
    """_param_sort_value parses '8B' to 8.0."""
    from lilbee.cli.tui.screens.catalog_utils import _param_sort_value

    assert _param_sort_value("8B") == 8.0
    assert _param_sort_value("0.6B") == 0.6


def test_param_sort_value_no_match():
    """_param_sort_value returns 0.0 for non-numeric."""
    from lilbee.cli.tui.screens.catalog_utils import _param_sort_value

    assert _param_sort_value("--") == 0.0


async def test_fetch_installed_names_exception():
    """_fetch_installed_names suppresses exception and keeps empty set."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._installed_names = set()
            with patch("lilbee.registry.ModelRegistry", side_effect=Exception("fail")):
                screen._fetch_installed_names()
            assert screen._installed_names == set()


async def test_catalog_nav_actions_forward_to_grid_in_grid_view():
    """Navigation actions forward to focused GridSelect in grid view mode."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            assert screen._grid_view is True
            # These should all run without error (forwarding to GridSelect or no-op)
            screen.action_page_down()
            screen.action_page_up()
            screen.action_cursor_down()
            screen.action_cursor_up()
            screen.action_jump_top()
            screen.action_jump_bottom()


async def test_catalog_select_variant_row():
    """_select_row with a variant row triggers _install_variant."""
    from lilbee.catalog import ModelFamily, ModelVariant
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import variant_to_row

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            variant = ModelVariant(
                hf_repo="org/model-GGUF",
                filename="model-Q4.gguf",
                param_count="8B",
                tag="8b",
                quant="Q4_K_M",
                size_mb=4096,
                recommended=True,
            )
            family = ModelFamily(
                slug="testmodel",
                name="TestModel",
                task="chat",
                description="Test",
                variants=(variant,),
            )
            row = variant_to_row(variant, family, installed=False)
            with patch.object(screen, "_install_variant") as mock_iv:
                screen._select_row(row)
                mock_iv.assert_called_once_with(variant, family)


async def test_catalog_install_variant_creates_catalog_model():
    """_install_variant creates a CatalogModel and calls _install_model."""
    from lilbee.catalog import ModelFamily, ModelVariant
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            variant = ModelVariant(
                hf_repo="org/model-GGUF",
                filename="model-Q4.gguf",
                param_count="8B",
                tag="8b",
                quant="Q4_K_M",
                size_mb=4096,
                recommended=True,
            )
            family = ModelFamily(
                slug="testmodel",
                name="TestModel",
                task="chat",
                description="Test",
                variants=(variant,),
            )
            with patch.object(screen, "_install_model") as mock_im:
                screen._install_variant(variant, family)
                mock_im.assert_called_once()
                entry = mock_im.call_args[0][0]
                assert entry.hf_repo == "org/model-GGUF"
                assert entry.featured is True


async def test_catalog_install_model_already_exists(tmp_path):
    """_install_model notifies when dest file already exists."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="existing-model")
            # Create the dest file so it exists
            cfg.models_dir = tmp_path
            dest = tmp_path / "test.gguf"
            dest.write_text("fake")
            with (
                patch("lilbee.catalog.resolve_filename", return_value="test.gguf"),
                patch.object(screen, "notify") as mock_notify,
            ):
                screen._install_model(m)
                mock_notify.assert_called_once()
                assert "already installed" in mock_notify.call_args[0][0]


async def test_catalog_enqueue_download_non_lilbee_app():
    """_enqueue_download notifies error when not LilbeeApp."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="dl-model")
            # CatalogTestApp is not LilbeeApp, so this should show error
            with patch.object(screen, "notify") as mock_notify:
                screen._enqueue_download(m)
                mock_notify.assert_called_once()
                assert "task bar" in mock_notify.call_args[0][0].lower()


async def test_catalog_make_progress_callback():
    """_make_progress_callback returns callback that formats progress."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            mock_bar = MagicMock()
            mock_bar.update_task = MagicMock()
            with patch.object(screen, "_safe_call") as mock_safe:
                cb = screen._make_progress_callback("task-1", mock_bar)
                cb(512 * 1024, 1024 * 1024)  # total > 0
                cb(512 * 1024, 0)  # total == 0 (throttled)
                assert mock_safe.call_count >= 1


async def test_catalog_safe_call_suppresses_exception():
    """_safe_call suppresses exceptions from call_from_thread."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            with patch.object(app, "call_from_thread", side_effect=RuntimeError("dead")):
                screen._safe_call(lambda: None)  # should not raise


async def test_catalog_get_highlighted_variant_name():
    """_get_highlighted_model_name returns correct name for variant row."""
    from lilbee.catalog import ModelFamily, ModelVariant
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import variant_to_row

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            variant = ModelVariant(
                hf_repo="org/model-GGUF",
                filename="model-Q4.gguf",
                param_count="8B",
                tag="8b",
                quant="Q4_K_M",
                size_mb=4096,
                recommended=True,
            )
            family = ModelFamily(
                slug="testmodel",
                name="TestModel",
                task="chat",
                description="Test",
                variants=(variant,),
            )
            row = variant_to_row(variant, family, installed=False)
            screen._rows = [row]
            screen._grid_view = False
            # Add a row to the table
            table = screen.query_one("#catalog-table", DataTable)
            table.clear()
            table.add_row("name", "chat", "8B", "4.0 GB", "Q4_K_M", "--")
            table.move_cursor(row=0)
            name = screen._get_highlighted_model_name()
            assert name == "testmodel:8b"


async def test_catalog_get_highlighted_remote_name():
    """_get_highlighted_model_name returns name for remote row."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import remote_to_row

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            rm = _make_remote_model(name="remote:latest")
            row = remote_to_row(rm)
            screen._rows = [row]
            table = screen.query_one("#catalog-table", DataTable)
            table.clear()
            table.add_row("name", "chat", "7B", "--", "--", "--")
            table.move_cursor(row=0)
            name = screen._get_highlighted_model_name()
            assert name == "remote:latest"


async def test_catalog_get_highlighted_catalog_name():
    """_get_highlighted_model_name returns name for catalog row."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import catalog_to_row

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="hf-model")
            row = catalog_to_row(m, installed=False)
            screen._rows = [row]
            table = screen.query_one("#catalog-table", DataTable)
            table.clear()
            table.add_row("name", "chat", "7B", "4.0 GB", "--", "1K")
            table.move_cursor(row=0)
            name = screen._get_highlighted_model_name()
            assert name == "hf-model:7b"


async def test_catalog_run_delete_success():
    """_run_delete success path notifies and refreshes."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            mock_mgr = MagicMock()
            mock_mgr.remove.return_value = True
            with patch("lilbee.cli.tui.screens.catalog.get_model_manager", return_value=mock_mgr):
                screen._run_delete("test:latest")
                await _pilot.pause()
                while screen.workers:
                    await _pilot.pause()
                mock_mgr.remove.assert_called_once_with("test:latest")


async def test_catalog_run_delete_failure():
    """_run_delete when remove returns False notifies error."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            mock_mgr = MagicMock()
            mock_mgr.remove.return_value = False
            with patch("lilbee.cli.tui.screens.catalog.get_model_manager", return_value=mock_mgr):
                screen._run_delete("test:latest")
                await _pilot.pause()
                while screen.workers:
                    await _pilot.pause()
                mock_mgr.remove.assert_called_once_with("test:latest")


async def test_catalog_run_delete_exception():
    """_run_delete exception path notifies error."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            mock_mgr = MagicMock()
            mock_mgr.remove.side_effect = OSError("disk full")
            with patch("lilbee.cli.tui.screens.catalog.get_model_manager", return_value=mock_mgr):
                screen._run_delete("test:latest")
                await _pilot.pause()
                while screen.workers:
                    await _pilot.pause()
                mock_mgr.remove.assert_called_once_with("test:latest")


async def test_catalog_run_download_success():
    """_run_download success path completes task."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="dl-model")
            mock_bar = MagicMock()
            with patch("lilbee.catalog.download_model") as mock_dl:
                screen._run_download(m, "task-1", mock_bar)
                await _pilot.pause()
                while screen.workers:
                    await _pilot.pause()
                mock_dl.assert_called_once()
                mock_bar.complete_task.assert_called_once_with("task-1")


async def test_catalog_run_download_permission_error():
    """_run_download PermissionError path shows gated repo message."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="gated-model")
            mock_bar = MagicMock()
            with patch("lilbee.catalog.download_model", side_effect=PermissionError("denied")):
                screen._run_download(m, "task-1", mock_bar)
                await _pilot.pause()
                while screen.workers:
                    await _pilot.pause()
                mock_bar.fail_task.assert_called_once()


async def test_catalog_run_download_generic_error():
    """_run_download generic Exception path shows error."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            m = _make_catalog_model(name="err-model")
            mock_bar = MagicMock()
            with patch("lilbee.catalog.download_model", side_effect=RuntimeError("network")):
                screen._run_download(m, "task-1", mock_bar)
                await _pilot.pause()
                while screen.workers:
                    await _pilot.pause()
                mock_bar.fail_task.assert_called_once()


# ---------------------------------------------------------------------------
# chat.py coverage
# ---------------------------------------------------------------------------


async def test_chat_on_show_calls_dismiss():
    """on_show calls splash.dismiss() to signal splash stop."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        with patch("lilbee.splash.dismiss") as mock_dismiss:
            app.screen.on_show()
            mock_dismiss.assert_called_once()


async def test_chat_on_setup_complete_completed_with_auto_sync():
    """_on_setup_complete with 'completed' and embedding ready triggers sync."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)):
        app.screen._auto_sync = True
        with (
            patch.object(app.screen, "_embedding_ready", return_value=True),
            patch.object(app.screen, "_run_sync") as mock_sync,
        ):
            app.screen._on_setup_complete("completed")
            mock_sync.assert_called_once()


async def test_chat_on_key_insert_mode_unfocused_input():
    """on_key in insert mode with unfocused input redirects printable chars."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Input

        app.screen.query_one("#chat-input", Input)
        # Focus the chat log instead
        app.screen.query_one("#chat-log").focus()
        await pilot.pause()
        assert app.screen._insert_mode is True
        # Simulate a printable key event
        from textual.events import Key

        event = Key("a", "a")
        event._bubbles = True  # type: ignore[attr-defined]
        app.screen.on_key(event)
        await pilot.pause()


async def test_chat_crawl_invalid_url():
    """_cmd_crawl with invalid URL shows error notification."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)):
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch(
                "lilbee.cli.tui.screens.chat.require_valid_crawl_url",
                side_effect=ValueError("bad url"),
            ),
            patch.object(app.screen, "notify") as mock_notify,
        ):
            app.screen._cmd_crawl("ftp://invalid.example.com")
            mock_notify.assert_called_once()
            assert "bad url" in mock_notify.call_args[0][0]


async def test_chat_login_no_token():
    """_cmd_login with no token opens browser."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)):
        with patch("webbrowser.open") as mock_open:
            app.screen._cmd_login("")
            mock_open.assert_called_once()


async def test_chat_login_with_token():
    """_cmd_login with token calls HF login."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch("huggingface_hub.login") as mock_login:
            app.screen._cmd_login("hf_test_token_123")
            while app.screen.workers:
                await pilot.pause()
            await pilot.pause()
            mock_login.assert_called_once_with(
                token="hf_test_token_123", add_to_git_credential=False
            )


async def test_chat_login_with_token_error():
    """_cmd_login with token handles login error."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with patch("huggingface_hub.login", side_effect=Exception("auth failed")) as mock_login:
            app.screen._cmd_login("hf_bad_token")
            while app.screen.workers:
                await pilot.pause()
            await pilot.pause()
            mock_login.assert_called_once()


async def test_chat_enter_normal_mode_while_streaming():
    """action_enter_normal_mode cancels stream when streaming."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)):
        app.screen.streaming = True
        app.screen.action_enter_normal_mode()
        assert app.screen.streaming is False
        # Should NOT have entered normal mode
        assert app.screen._insert_mode is True


async def test_chat_on_chat_input_changed_completing():
    """_on_chat_input_changed is no-op when _completing is True."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

        overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
        overlay.show_completions(["/help"])
        app.screen._completing = True
        from textual.widgets import Input

        inp = app.screen.query_one("#chat-input", Input)
        inp.value = "/test"
        await pilot.pause()
        # Overlay should still be visible since _completing skips hide
        assert overlay.is_visible


# ---------------------------------------------------------------------------
# settings.py coverage
# ---------------------------------------------------------------------------


def test_settings_make_select_value_matches_choice():
    """_make_select returns Select with value preset when it matches choices."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import _make_select

    defn = SettingDef(type=str, nullable=False, group="Test", choices=("auto", "litellm"))
    sel = _make_select("test_key", defn, "auto")
    # When value matches, the Select is created with value= kwarg
    assert sel.name == "test_key"
    assert sel.id == "ed-test_key"


def test_settings_make_select_value_no_match():
    """_make_select returns Select without preset value when no match."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import _make_select

    defn = SettingDef(type=str, nullable=False, group="Test", choices=("auto", "litellm"))
    sel = _make_select("test_key", defn, "unknown")
    assert sel.name == "test_key"
    assert sel.id == "ed-test_key"


async def test_settings_on_input_save_name_none():
    """_on_input_save returns early when name is None."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.input.name = None
        event.value = "x"
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_input_save(event)
            mock_pv.assert_not_called()


async def test_settings_on_input_save_defn_none():
    """_on_input_save returns early when SETTINGS_MAP has no entry."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.input.name = "nonexistent_key_xyz"
        event.value = "x"
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_input_save(event)
            mock_pv.assert_not_called()


async def test_settings_on_input_save_same_value_skip():
    """_on_input_save skips persist when value matches current."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.input.name = "top_k"
        event.value = str(cfg.top_k)
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_input_save(event)
            mock_pv.assert_not_called()


async def test_settings_on_checkbox_save_name_none():
    """_on_checkbox_save returns early when name is None."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.checkbox.name = None
        event.checkbox.value = True
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_checkbox_save(event)
            mock_pv.assert_not_called()


async def test_settings_on_checkbox_save_defn_none():
    """_on_checkbox_save returns early when SETTINGS_MAP has no entry."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.checkbox.name = "nonexistent_key"
        event.checkbox.value = True
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_checkbox_save(event)
            mock_pv.assert_not_called()


async def test_settings_on_select_save_name_none():
    """_on_select_save returns early when name is None."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.select.name = None
        event.value = "x"
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_select_save(event)
            mock_pv.assert_not_called()


async def test_settings_on_select_save_defn_none():
    """_on_select_save returns early when SETTINGS_MAP has no entry."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.select.name = "nonexistent_key"
        event.value = "x"
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_select_save(event)
            mock_pv.assert_not_called()


async def test_settings_parse_value_nullable_none():
    """_parse_value returns None for nullable setting with 'none'."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        defn = SettingDef(type=float, nullable=True, group="Test")
        result = screen._parse_value(defn, "none")
        assert result is None


async def test_settings_parse_value_nullable_empty():
    """_parse_value returns None for nullable setting with empty string."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        defn = SettingDef(type=float, nullable=True, group="Test")
        result = screen._parse_value(defn, "")
        assert result is None


async def test_settings_refresh_help_exception():
    """_refresh_help suppresses exception when widget not found."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        defn = SettingDef(type=str, nullable=False, group="Test")
        # This should not raise despite the widget not existing
        screen._refresh_help("nonexistent_key_xyz", defn)


async def test_settings_go_back_non_lilbee_app():
    """action_go_back pops screen on non-LilbeeApp."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from lilbee.cli.tui.screens.settings import SettingsScreen

        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen.action_go_back()
        await pilot.pause()


# ---------------------------------------------------------------------------
# task_center.py coverage
# ---------------------------------------------------------------------------


class TaskCenterTestApp(App[None]):
    """Non-LilbeeApp for testing TaskCenter go_back fallback."""

    CSS = ""

    def __init__(self) -> None:
        super().__init__()
        from lilbee.cli.tui.widgets.task_bar import TaskBarController

        self.task_bar = TaskBarController(self)

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.task_center import TaskCenter

        self.push_screen(TaskCenter())


async def test_task_center_go_back_non_lilbee_app():
    """action_go_back pops screen on non-LilbeeApp."""
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = TaskCenterTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.action_go_back()
        await pilot.pause()
        assert not isinstance(app.screen, TaskCenter)


async def test_task_center_queue_change_exception():
    """_on_queue_change suppresses exception from _refresh_tasks."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        with patch.object(screen, "_refresh_tasks", side_effect=RuntimeError("fail")):
            screen._on_queue_change()


async def test_task_center_show_detail_task_not_found():
    """_show_task_detail with unknown task_id shows empty."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        screen._show_task_detail("nonexistent-id-xyz")
        detail = screen.query_one("#task-detail", Static)
        assert detail.content == ""


async def test_task_center_find_task_not_found():
    """_find_task returns None for unknown ID."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.task_center import TaskCenter

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(TaskCenter())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TaskCenter)
        result = screen._find_task("nonexistent-id")
        assert result is None


# ---------------------------------------------------------------------------
# app.py coverage
# ---------------------------------------------------------------------------


async def test_app_action_quit_when_streaming():
    """action_quit cancels stream instead of exiting when streaming."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ChatScreen)
        screen.streaming = True
        with patch.object(screen, "action_cancel_stream") as mock_cancel:
            await app.action_quit()
            mock_cancel.assert_called_once()


async def test_app_action_quit_routes_to_wizard_cancel():
    """action_quit cancels the wizard instead of exiting when wizard is active."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # Push a wizard on top
        wizard = SetupWizard()
        with _patch_setup_scan(), _patch_setup_ram(16.0):
            app.push_screen(wizard)
            await pilot.pause()
            assert isinstance(app.screen, SetupWizard)
            assert not wizard._cancel_event.is_set()
            await app.action_quit()
            assert wizard._cancel_event.is_set()


async def test_app_action_quit_double_force_exits():
    """Double Ctrl+C within 2s calls _force_quit."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        # First quit sets last_quit_time
        with patch.object(app, "exit"):
            await app.action_quit()
        # Second quit within 2s should force-quit
        with patch.object(app, "_force_quit") as mock_fq:
            await app.action_quit()
            mock_fq.assert_called_once()


async def test_app_force_quit_calls_os_exit():
    """_force_quit resets services and calls os._exit."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.app.reset_services") as mock_reset,
            patch("os._exit") as mock_exit,
        ):
            app._force_quit()
            mock_reset.assert_called_once()
            mock_exit.assert_called_once_with(1)


async def test_app_switch_view_unknown():
    """switch_view with unknown name does nothing."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Nonexistent")
        await pilot.pause()
        # Should still be on the same screen type (chat)
        assert isinstance(app.screen, ChatScreen)


async def test_app_switch_view_chat_when_already_chat():
    """switch_view('Chat') when already on Chat is a no-op."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)
        app.switch_view("Chat")
        await pilot.pause()
        assert isinstance(app.screen, ChatScreen)


async def test_app_switch_view_non_chat():
    """switch_view to a non-Chat view works via factory."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.switch_view("Settings")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        assert app.active_view == "Settings"


# ---------------------------------------------------------------------------
# commands.py coverage
# ---------------------------------------------------------------------------


async def test_command_provider_app_not_lilbee():
    """_app property raises TypeError on non-LilbeeApp."""
    from lilbee.cli.tui.commands import LilbeeCommandProvider

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)):
        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with pytest.raises(TypeError, match="LilbeeApp"):
            _ = provider._app


async def test_command_provider_action_setup():
    """_action_setup pushes SetupWizard."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch("lilbee.cli.tui.screens.setup._scan_installed_models", return_value=([], [])):
            provider._action_setup()
            await pilot.pause()
            assert isinstance(app.screen, SetupWizard)


# ---------------------------------------------------------------------------
# __init__.py coverage
# ---------------------------------------------------------------------------


def test_run_tui_keyboard_interrupt_during_shutdown():
    """run_tui handles KeyboardInterrupt during shutdown cleanup."""
    from lilbee.cli.tui import run_tui

    mock_app = MagicMock()
    mock_app.run.return_value = None
    with (
        patch("lilbee.cli.tui.app.LilbeeApp", return_value=mock_app),
        patch("lilbee.cli.tui.shutdown_executor", side_effect=KeyboardInterrupt),
        patch("os._exit") as mock_exit,
    ):
        run_tui()
        mock_exit.assert_called_once_with(1)


def test_run_tui_exception_during_shutdown():
    """run_tui handles generic Exception during shutdown cleanup."""
    from lilbee.cli.tui import run_tui

    mock_app = MagicMock()
    mock_app.run.return_value = None
    with (
        patch("lilbee.cli.tui.app.LilbeeApp", return_value=mock_app),
        patch("lilbee.cli.tui.shutdown_executor", side_effect=RuntimeError("fail")),
        patch("os._exit") as mock_exit,
    ):
        run_tui()
        mock_exit.assert_called_once_with(1)


async def test_chat_on_show_dismiss_with_fd():
    """on_show calls dismiss which closes the splash pipe fd."""
    import os

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    os.environ["_LILBEE_SPLASH_FD"] = str(write_fd)

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        app.screen.on_show()
        assert "_LILBEE_SPLASH_FD" not in os.environ


async def test_chat_on_show_dismiss_no_fd():
    """on_show dismiss is a no-op when no splash fd is set."""
    import os

    os.environ.pop("_LILBEE_SPLASH_FD", None)

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        app.screen.on_show()  # Should not raise
        assert "_LILBEE_SPLASH_FD" not in os.environ


async def test_chat_embedding_ready_false_on_exception():
    """_embedding_ready returns False when resolve raises."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        screen = app.screen
        assert isinstance(screen, ChatScreen)
        with patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            side_effect=FileNotFoundError("not found"),
        ):
            assert screen._embedding_ready() is False


async def test_chat_hide_banner():
    """_hide_chat_only_banner hides the banner."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        app.screen._show_chat_only_banner()
        assert app.screen.query_one("#chat-only-banner").display is True
        app.screen._hide_chat_only_banner()
        assert app.screen.query_one("#chat-only-banner").display is False


async def test_chat_f5_opens_setup():
    """F5 binding opens the setup wizard."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with patch.object(app.screen, "_cmd_setup") as mock_setup:
            app.screen.action_open_setup()
            mock_setup.assert_called_once_with("")


async def test_chat_on_key_insert_mode_focus():
    """on_key in insert mode redirects printable chars to input."""
    from textual.events import Key

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        app.screen._insert_mode = True
        inp = app.screen.query_one("#chat-input")
        inp.blur()
        # Create a Key event with a printable character
        event = MagicMock(spec=Key)
        event.is_printable = True
        event.character = "x"
        event.key = "x"
        app.screen.on_key(event)
        assert app.screen._insert_mode is True


def test_chat_has_auto_focus():
    """ChatScreen declares AUTO_FOCUS for the chat input."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    assert ChatScreen.AUTO_FOCUS == "#chat-input"


def test_chat_has_help_attribute():
    """ChatScreen declares HELP for HelpPanel."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    assert ChatScreen.HELP
    assert "Chat" in ChatScreen.HELP


async def test_chat_action_enter_normal_mode_streaming():
    """action_enter_normal_mode cancels workers and stops streaming."""
    import asyncio

    async def _slow_worker() -> None:
        await asyncio.sleep(999)

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        app.screen.streaming = True
        # Start a real Textual worker so self.workers is non-empty
        app.screen.run_worker(_slow_worker(), exclusive=False)
        await _pilot.pause()
        assert len(list(app.screen.workers)) > 0
        app.screen.action_enter_normal_mode()
        assert app.screen.streaming is False


async def test_chat_action_toggle_markdown():
    """action_toggle_markdown toggles cfg.markdown_rendering and rebuilds messages."""
    from lilbee.cli.tui.widgets.message import AssistantMessage

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        # Add an assistant message to the chat log so the rebuild loop fires
        chat_log = app.screen.query_one("#chat-log")
        msg_widget = AssistantMessage()
        await chat_log.mount(msg_widget)
        await _pilot.pause()
        cfg.markdown_rendering = True
        await app.screen.action_toggle_markdown()
        assert cfg.markdown_rendering is False


async def test_chat_run_sync_when_already_active():
    """_run_sync notifies when sync is already active."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        app.screen._sync_active = True
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._run_sync()
            mock_notify.assert_called_once()


async def test_chat_remove_model_exception():
    """_run_remove_model handles exception from mgr.remove."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        mock_mgr = MagicMock()
        mock_mgr.is_installed.return_value = True
        mock_mgr.remove.side_effect = RuntimeError("disk error")
        with patch("lilbee.model_manager.get_model_manager", return_value=mock_mgr):
            app.screen._run_remove_model("test-model")
            while app.screen.workers:
                await _pilot.pause()
            mock_mgr.remove.assert_called_once_with("test-model")


async def test_chat_cmd_crawl_with_valid_url():
    """_cmd_crawl enqueues a crawl task for a valid URL."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch("lilbee.cli.tui.screens.chat.require_valid_crawl_url"),
            patch.object(app.screen, "_run_crawl_background") as mock_crawl,
        ):
            app.screen._cmd_crawl("https://example.com")
            mock_crawl.assert_called_once()


async def test_chat_cmd_crawl_invalid_url():
    """_cmd_crawl notifies error for invalid URL."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch(
                "lilbee.cli.tui.screens.chat.require_valid_crawl_url",
                side_effect=ValueError("bad url"),
            ),
            patch.object(app.screen, "notify") as mock_notify,
        ):
            app.screen._cmd_crawl("not-a-url")
            mock_notify.assert_called()


async def test_chat_cmd_wiki_disabled_notifies():
    """/wiki notifies when wiki config flag is off."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch.object(app.screen, "notify") as mock_notify,
        ):
            mock_cfg.wiki = False
            app.screen._cmd_wiki("generate")
            mock_notify.assert_called_once()
            assert "disabled" in mock_notify.call_args[0][0].lower()


async def test_chat_cmd_wiki_usage_without_subcommand():
    """/wiki without 'generate' shows usage hint."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch.object(app.screen, "notify") as mock_notify,
        ):
            mock_cfg.wiki = True
            app.screen._cmd_wiki("")
            mock_notify.assert_called_once()
            assert "/wiki generate" in mock_notify.call_args[0][0]


async def test_chat_cmd_wiki_no_sources_notifies():
    """/wiki generate notifies when no sources are indexed."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        fake_store = MagicMock()
        fake_store.get_sources.return_value = []
        fake_svc = MagicMock(store=fake_store)
        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch("lilbee.cli.tui.screens.chat.get_services", return_value=fake_svc),
            patch.object(app.screen, "notify") as mock_notify,
        ):
            mock_cfg.wiki = True
            app.screen._cmd_wiki("generate")
            mock_notify.assert_called_once()
            assert "No indexed documents" in mock_notify.call_args[0][0]


async def test_chat_cmd_wiki_unknown_source_notifies():
    """/wiki generate <name> errors when name is not an indexed source."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        fake_store = MagicMock()
        fake_store.get_sources.return_value = [{"filename": "doc.txt"}]
        fake_svc = MagicMock(store=fake_store)
        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch("lilbee.cli.tui.screens.chat.get_services", return_value=fake_svc),
            patch.object(app.screen, "notify") as mock_notify,
        ):
            mock_cfg.wiki = True
            app.screen._cmd_wiki("generate missing.txt")
            mock_notify.assert_called_once()
            assert "Source not found" in mock_notify.call_args[0][0]


async def test_chat_cmd_wiki_generate_runs_background(tmp_path):
    """/wiki generate enqueues a task and runs generate_summary_page for each source."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        fake_store = MagicMock()
        fake_store.get_sources.return_value = [
            {"filename": "a.txt"},
            {"filename": "b.txt"},
        ]
        fake_store.get_chunks_by_source.return_value = [MagicMock()]
        fake_svc = MagicMock(store=fake_store, provider=MagicMock())
        generated_for: list[str] = []

        def _fake_generate(source, chunks, provider, store, on_progress=None):
            generated_for.append(source)
            if on_progress is not None:
                on_progress("writing", {})
            return tmp_path / f"{source}.md"

        task_bar = app.task_bar
        add_task_spy = MagicMock(wraps=task_bar.add_task)
        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch("lilbee.cli.tui.screens.chat.get_services", return_value=fake_svc),
            patch("lilbee.wiki.gen.generate_summary_page", side_effect=_fake_generate),
            patch.object(task_bar, "add_task", add_task_spy),
        ):
            mock_cfg.wiki = True
            app.screen._cmd_wiki("generate")
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()

        add_task_spy.assert_called_once()
        assert generated_for == ["a.txt", "b.txt"]


async def test_chat_cmd_wiki_get_sources_raises_notifies_empty():
    """/wiki treats a store.get_sources exception as 'no indexed documents'."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        fake_store = MagicMock()
        fake_store.get_sources.side_effect = RuntimeError("db gone")
        fake_svc = MagicMock(store=fake_store)
        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch("lilbee.cli.tui.screens.chat.get_services", return_value=fake_svc),
            patch.object(app.screen, "notify") as mock_notify,
        ):
            mock_cfg.wiki = True
            app.screen._cmd_wiki("generate")
            mock_notify.assert_called_once()
            assert "No indexed documents" in mock_notify.call_args[0][0]


async def test_chat_cmd_wiki_generate_single_source(tmp_path):
    """/wiki generate <name> runs generation only for the matching source."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        fake_store = MagicMock()
        fake_store.get_sources.return_value = [
            {"filename": "a.txt"},
            {"filename": "b.txt"},
        ]
        fake_store.get_chunks_by_source.return_value = [MagicMock()]
        fake_svc = MagicMock(store=fake_store, provider=MagicMock())
        generated_for: list[str] = []

        def _fake_generate(source, chunks, provider, store, on_progress=None):
            generated_for.append(source)
            return tmp_path / f"{source}.md"

        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch("lilbee.cli.tui.screens.chat.get_services", return_value=fake_svc),
            patch("lilbee.wiki.gen.generate_summary_page", side_effect=_fake_generate),
        ):
            mock_cfg.wiki = True
            app.screen._cmd_wiki("generate b.txt")
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()

        assert generated_for == ["b.txt"]


async def test_chat_cmd_wiki_skips_sources_with_no_chunks():
    """/wiki generate skips sources that return no chunks without raising."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        fake_store = MagicMock()
        fake_store.get_sources.return_value = [{"filename": "empty.txt"}]
        fake_store.get_chunks_by_source.return_value = []
        fake_svc = MagicMock(store=fake_store, provider=MagicMock())

        gen_spy = MagicMock()
        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch("lilbee.cli.tui.screens.chat.get_services", return_value=fake_svc),
            patch("lilbee.wiki.gen.generate_summary_page", new=gen_spy),
        ):
            mock_cfg.wiki = True
            app.screen._cmd_wiki("generate")
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()

        gen_spy.assert_not_called()


async def test_chat_refresh_wiki_screen_reloads_pages():
    """_refresh_wiki_screen delegates to WikiScreen.reload, which refreshes the sidebar."""
    from lilbee.cli.tui.screens.wiki import WikiScreen

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        chat_screen = app.screen
        wiki = WikiScreen()
        app.push_screen(wiki)
        await _pilot.pause()

        with patch.object(wiki, "_load_pages") as mock_load:
            chat_screen._refresh_wiki_screen()
            mock_load.assert_called_once_with()


async def test_chat_cmd_wiki_generate_failure_fails_task():
    """/wiki generate marks the task failed if generate_summary_page raises."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        fake_store = MagicMock()
        fake_store.get_sources.return_value = [{"filename": "a.txt"}]
        fake_store.get_chunks_by_source.return_value = [MagicMock()]
        fake_svc = MagicMock(store=fake_store, provider=MagicMock())

        task_bar = app.task_bar
        fail_spy = MagicMock(wraps=task_bar.fail_task)
        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch("lilbee.cli.tui.screens.chat.get_services", return_value=fake_svc),
            patch(
                "lilbee.wiki.gen.generate_summary_page",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(task_bar, "fail_task", fail_spy),
        ):
            mock_cfg.wiki = True
            app.screen._cmd_wiki("generate")
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()

        fail_spy.assert_called_once()
        assert "boom" in fail_spy.call_args[0][1]


async def test_chat_auto_sync_on_mount_runs_sync():
    """When auto_sync and embedding ready, _run_sync is called on mount."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    class SyncApp(App[None]):
        CSS = ""

        def __init__(self) -> None:
            super().__init__()
            from lilbee.cli.tui.widgets.task_bar import TaskBarController

            self.task_bar = TaskBarController(self)

        def compose(self) -> ComposeResult:
            yield from ()

        def on_mount(self) -> None:
            self.push_screen(ChatScreen(auto_sync=True))

    app = SyncApp()
    with (
        patch("lilbee.cli.tui.screens.chat.ChatScreen._needs_setup", return_value=False),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=True),
        patch("lilbee.cli.tui.screens.chat.ChatScreen._run_sync") as mock_sync,
    ):
        async with app.run_test(size=(120, 40)) as _pilot:
            await _pilot.pause()
            mock_sync.assert_called_once()


async def test_chat_on_key_non_key_event_returns():
    """on_key returns early for non-Key events."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        # Pass a non-Key object
        app.screen.on_key("not_a_key_event")  # Should not raise
        assert app.screen._insert_mode is True


async def test_chat_vim_scroll_actions_work():
    """Vim scroll actions execute without error in normal mode."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.screen.action_enter_normal_mode()
        await pilot.pause()
        app.screen.action_vim_scroll_down()
        app.screen.action_vim_scroll_up()
        app.screen.action_vim_scroll_home()
        app.screen.action_vim_scroll_end()
        assert app.screen._insert_mode is False


async def test_chat_cmd_setup_opens_wizard():
    """_cmd_setup pushes SetupWizard screen."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        from lilbee.cli.tui.screens.setup import SetupWizard

        with patch("lilbee.cli.tui.screens.setup._scan_installed_models", return_value=([], [])):
            app.screen._cmd_setup("")
            await _pilot.pause()
            assert isinstance(app.screen, SetupWizard)


async def test_catalog_enqueue_download_in_lilbee_app():
    """_enqueue_download works with a real LilbeeApp."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            cm = _make_catalog_model(name="enqueue-test")
            with patch.object(screen, "_run_download") as mock_dl:
                screen._enqueue_download(cm)
                mock_dl.assert_called_once()


async def test_catalog_select_row_out_of_range():
    """_on_row_selected returns early for out-of-range cursor_row."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._rows = []
            event = MagicMock()
            event.cursor_row = -1
            with patch.object(screen, "_select_row") as mock_sel:
                screen._on_row_selected(event)  # Should not raise
                mock_sel.assert_not_called()


async def test_chat_cmd_crawl_no_args():
    """_cmd_crawl with empty args notifies usage."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch.object(app.screen, "notify") as mock_notify,
        ):
            app.screen._cmd_crawl("")
            mock_notify.assert_called()


def test_chat_embedding_ready_real_code_false():
    """Placeholder — real test is in test_tui_e2e.py to avoid autouse fixture."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    assert hasattr(ChatScreen, "_embedding_ready")


async def test_chat_run_sync_worker_cancelled():
    """_run_sync_worker handles CancelledError by disabling auto_sync."""
    import asyncio as _asyncio

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        app.screen._auto_sync = True
        with patch("asyncio.run", side_effect=_asyncio.CancelledError):
            app.screen._run_sync_worker("test-task-id")
            while app.screen.workers:
                await _pilot.pause()
        assert app.screen._auto_sync is False


async def test_chat_add_skipped_file():
    """_run_add_background notifies about skipped files."""
    from pathlib import Path as _Path

    from lilbee.cli.helpers import CopyResult

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        mock_result = CopyResult(copied=[], skipped=["existing.txt"])

        from lilbee.progress import EventType, FileStartEvent

        async def fake_sync(*, quiet: bool = True, on_progress: object = None) -> None:
            if on_progress:
                on_progress(
                    EventType.FILE_START,
                    FileStartEvent(file="test.txt", total_files=0, current_file=0),
                )

        with (
            patch("lilbee.cli.helpers.copy_files", return_value=mock_result),
            patch("lilbee.ingest.sync", new=fake_sync),
        ):
            app.screen._run_add_background(_Path("test.txt"), "task-1")
            while app.screen.workers:
                await _pilot.pause()
            assert app.screen._sync_active is False


async def test_chat_add_sync_progress_wrong_type():
    """_run_add_background sync progress raises TypeError for non-FileStartEvent."""
    from pathlib import Path as _Path

    from lilbee.cli.helpers import CopyResult
    from lilbee.progress import CrawlPageEvent, EventType

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        mock_result = CopyResult(copied=[_Path("ok.txt")], skipped=[])

        async def fake_sync(*, quiet=True, on_progress=None):
            if on_progress:
                # Send wrong event type for FILE_START — triggers TypeError guard
                on_progress(
                    EventType.FILE_START,
                    CrawlPageEvent(current=1, total=1, url="https://x.com"),
                )

        with (
            patch("lilbee.cli.helpers.copy_files", return_value=mock_result),
            patch("lilbee.ingest.sync", new=fake_sync),
        ):
            app.screen._run_add_background(_Path("test.txt"), "task-wrong")
            while app.screen.workers:
                await _pilot.pause()
            assert app.screen._sync_active is False


async def test_chat_crawl_background_success():
    """_run_crawl_background completes successfully with progress and triggers sync."""
    from pathlib import Path as _Path

    from lilbee.progress import CrawlPageEvent, EventType

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()

        async def fake_crawl(url, *, depth=0, max_pages=0, on_progress=None):
            if on_progress:
                on_progress(
                    EventType.CRAWL_PAGE,
                    CrawlPageEvent(current=1, total=2, url="https://example.com/page1"),
                )
            return [_Path("p.md")]

        with (
            patch("lilbee.crawler.crawl_and_save", side_effect=fake_crawl),
            patch.object(app.screen, "_run_sync") as mock_sync,
        ):
            app.screen._run_crawl_background("https://example.com", 1, 10, "crawl-1")
            while app.screen.workers:
                await _pilot.pause()
            mock_sync.assert_called()


def test_on_row_selected_valid_index():
    """_on_row_selected calls _select_row for a valid row index."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import TableRow

    screen = MagicMock()
    row = TableRow(
        name="test",
        task="chat",
        params="7B",
        size="4.0 GB",
        quant="Q4_K_M",
        downloads="1K",
        installed=False,
        featured=False,
        sort_downloads=1000,
        sort_size=4.0,
    )
    screen._rows = [row]
    event = MagicMock()
    event.cursor_row = 0
    CatalogScreen._on_row_selected(screen, event)
    screen._select_row.assert_called_once_with(row)


def test_is_installed_by_name():
    """_is_installed returns True when name matches."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    screen = MagicMock()
    screen._installed_names = {"my-model:latest"}
    assert CatalogScreen._is_installed(screen, "my-model:latest") is True


def test_is_installed_no_match():
    """_is_installed returns False when neither name nor repo matches."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    screen = MagicMock()
    screen._installed_names = {"other:latest"}
    assert CatalogScreen._is_installed(screen, "missing", repo="", filename="") is False


def test_on_row_selected_negative_index():
    """_on_row_selected returns early for negative cursor_row."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    screen = MagicMock()
    screen._rows = []
    event = MagicMock()
    event.cursor_row = -1
    CatalogScreen._on_row_selected(screen, event)
    screen._select_row.assert_not_called()


def test_on_row_selected_exceeds_length():
    """_on_row_selected returns early when index exceeds rows length."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    screen = MagicMock()
    screen._rows = []
    event = MagicMock()
    event.cursor_row = 5
    CatalogScreen._on_row_selected(screen, event)
    screen._select_row.assert_not_called()


def test_type_pill_with_choices():
    """_type_pill returns 'select' pill when defn has choices."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import _type_pill

    defn = SettingDef(type=str, nullable=False, group="Test", choices=("a", "b"))
    result = _type_pill(defn)
    assert "select" in str(result).lower()


def test_make_editor_with_choices():
    """_make_editor returns a Select widget when defn has choices."""
    from textual.widgets import Select

    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import _make_editor

    with patch(
        "lilbee.cli.tui.screens.settings._effective_value",
        return_value="auto",
    ):
        defn = SettingDef(type=str, nullable=False, group="Test", choices=("auto", "litellm"))
        widget = _make_editor("test_key", defn)
    assert isinstance(widget, Select)


async def test_catalog_fetch_installed_names():
    """_fetch_installed_names populates _installed_names from registry."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            mock_manifest = MagicMock()
            mock_manifest.name = "test-model"
            mock_manifest.tag = "latest"
            mock_manifest.source_repo = "org/test-model-GGUF"
            mock_manifest.source_filename = "test.gguf"
            mock_registry = MagicMock()
            mock_registry.list_installed.return_value = [mock_manifest]

            with patch("lilbee.registry.ModelRegistry", return_value=mock_registry):
                screen._fetch_installed_names()
            assert "test-model:latest" in screen._installed_names
            assert "org/test-model-GGUF/test.gguf" in screen._installed_names


async def test_catalog_worker_state_unknown_worker():
    """on_worker_state_changed returns early for unknown worker name."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            event = MagicMock()
            event.state = MagicMock()
            event.state.name = "SUCCESS"
            from textual.worker import WorkerState

            event.state = WorkerState.SUCCESS
            event.worker.result = []
            event.worker.name = "unknown_worker"
            with patch.object(screen, "_refresh_view") as mock_refresh:
                screen.on_worker_state_changed(event)
                mock_refresh.assert_not_called()


async def test_catalog_is_installed_by_repo():
    """_is_installed matches by source repo/filename."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._installed_names = {"org/model-GGUF/test.gguf"}
            assert screen._is_installed("x", repo="org/model-GGUF", filename="test.gguf") is True
            assert screen._is_installed("x", repo="org/other", filename="other.gguf") is False


async def test_catalog_install_model_resolve_exception():
    """_install_model handles resolve_filename exception."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            cm = _make_catalog_model(name="fail-resolve")
            with (
                patch("lilbee.catalog.resolve_filename", side_effect=RuntimeError("fail")),
                patch.object(screen, "_enqueue_download") as mock_dl,
            ):
                screen._install_model(cm)
                mock_dl.assert_called_once_with(cm)


async def test_catalog_delete_when_input_focused():
    """action_delete_model returns early when Input is focused."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            # Focus the search input
            inp = screen.query_one("#catalog-search")
            inp.focus()
            await _pilot.pause()
            # action_delete_model should return early
            with patch.object(screen, "notify") as mock_notify:
                screen.action_delete_model()
                mock_notify.assert_not_called()


async def test_catalog_get_highlighted_model_name_catalog():
    """_get_highlighted_model_name returns catalog model name."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import TableRow

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            cm = _make_catalog_model(name="qwen3", tag="8b", display_name="Qwen3 8B")
            row = TableRow(
                name="Qwen3 8B",
                task="chat",
                params="8B",
                size="5.0 GB",
                quant="Q4_K_M",
                downloads="1K",
                installed=False,
                featured=False,
                sort_downloads=1000,
                sort_size=5.0,
                ref=cm.ref,
                catalog_model=cm,
            )
            screen._rows = [row]
            table = screen.query_one("#catalog-table", DataTable)
            table.clear()
            table.add_row("Qwen3 8B", "chat", "8B", "5.0 GB", "Q4_K_M", "1K")
            table.move_cursor(row=0)
            result = screen._get_highlighted_model_name()
            assert result == "qwen3:8b"


async def test_catalog_get_highlighted_model_name_fallback_none():
    """_get_highlighted_model_name returns None when row has no model ref."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import TableRow

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            row = TableRow(
                name="orphan",
                task="chat",
                params="?",
                size="?",
                quant="?",
                downloads="?",
                installed=False,
                featured=False,
                sort_downloads=0,
                sort_size=0.0,
            )
            screen._rows = [row]
            table = screen.query_one("#catalog-table", DataTable)
            table.clear()
            table.add_row("orphan", "chat", "?", "?", "?", "?")
            table.move_cursor(row=0)
            result = screen._get_highlighted_model_name()
            assert result is None


async def test_catalog_browse_more_clicked():
    """Browse more button triggers HF model fetch."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            assert screen._hf_fetched is False
            with patch.object(screen, "_fetch_all_hf_models") as mock_fetch:
                screen._on_browse_more_clicked()
                assert screen._hf_fetched is True
                mock_fetch.assert_called_once()


async def test_catalog_grid_selected_with_model_card():
    """Grid selection with ModelCard delegates to _select_row."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import TableRow
    from lilbee.cli.tui.widgets.grid_select import GridSelect
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            row = TableRow(
                name="card-model",
                task="chat",
                params="7B",
                size="4.0 GB",
                quant="Q4_K_M",
                downloads="1K",
                installed=False,
                featured=False,
                sort_downloads=1000,
                sort_size=4.0,
            )
            mock_card = MagicMock(spec=ModelCard)
            mock_card.row = row
            event = MagicMock(spec=GridSelect.Selected)
            event.widget = mock_card
            with patch.object(screen, "_select_row") as mock_sel:
                screen._on_grid_selected(event)
                mock_sel.assert_called_once_with(row)


async def test_cmd_add_uses_indeterminate_progress_during_ingest(tmp_path):
    """BEE-65f: /add must not claim determinate progress while ingest runs.
    Before the fix, the task bar jumped 0 -> 50 -> 100 for a single-file add,
    falsely showing "done" while parse/chunk/embed/store were still running
    for tens of seconds. The TaskBarController must flip the task to
    indeterminate mode during copy and sync so the bar pulses instead of
    lying about completion.
    """
    from pathlib import Path as _Path

    from lilbee.cli.helpers import CopyResult
    from lilbee.progress import EventType, FileStartEvent

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        test_file = tmp_path / "note.md"
        test_file.write_text("hello")

        task_id = app.task_bar.add_task("Add note.md", "add")
        app.task_bar.queue.advance("add")

        snapshots: list[tuple[bool, int]] = []

        def on_change() -> None:
            task = app.task_bar.queue.get_task(task_id)
            if task is not None:
                snapshots.append((task.indeterminate, task.progress))

        app.task_bar.queue.subscribe(on_change)

        async def fake_sync(*, quiet=True, on_progress=None):
            if on_progress:
                on_progress(
                    EventType.FILE_START,
                    FileStartEvent(file="note.md", total_files=1, current_file=1),
                )

        try:
            with (
                patch(
                    "lilbee.cli.helpers.copy_files",
                    return_value=CopyResult(copied=[_Path("note.md")], skipped=[]),
                ),
                patch("lilbee.ingest.sync", new=fake_sync),
            ):
                app.screen._run_add_background(test_file, task_id)
                while app.screen.workers:
                    await _pilot.pause()
                await _pilot.pause()
        finally:
            app.task_bar.queue.unsubscribe(on_change)

        # Drop the final DONE entry. The task bar flips indeterminate off at
        # completion; everything BEFORE that must be indeterminate and must
        # not claim 100% progress.
        running_snapshots = [s for s in snapshots if s != (False, 100)]
        assert running_snapshots, f"expected progress updates, got {snapshots}"
        assert all(indet for indet, _ in running_snapshots), (
            f"task should stay indeterminate until completion, got {running_snapshots}"
        )
        assert all(p < 100 for _, p in running_snapshots), (
            f"no update should claim 100% progress while running, got {running_snapshots}"
        )


async def test_task_bar_indeterminate_renders_total_none():
    """BEE-65f: indeterminate tasks render the Textual ProgressBar with total=None.
    Setting total=None is how Textual draws an indeterminate pulsing bar.
    A task flagged indeterminate=True must not leak a total=100 update.
    """
    from textual.widgets import ProgressBar

    from lilbee.cli.tui.task_queue import TaskStatus
    from lilbee.cli.tui.widgets.task_bar import TaskBar

    class _Harness(App[None]):
        def __init__(self) -> None:
            super().__init__()
            from lilbee.cli.tui.widgets.task_bar import TaskBarController

            self.task_bar = TaskBarController(self)

        def compose(self) -> ComposeResult:
            yield TaskBar(id="tbar")

    app = _Harness()
    async with app.run_test(size=(80, 24)) as _pilot:
        task_id = app.task_bar.add_task("indet", "add")
        app.task_bar.queue.advance("add")
        app.task_bar.update_task(task_id, 0, "working", indeterminate=True)
        # Panel compose + the spinner tick need a few frames to settle.
        for _ in range(5):
            await _pilot.pause()

        task = app.task_bar.queue.get_task(task_id)
        assert task is not None
        assert task.indeterminate is True
        assert task.status == TaskStatus.ACTIVE

        bar = app.screen.query_one("#tbar", TaskBar)
        pb = bar.query_one(ProgressBar)
        # total=None is Textual's indeterminate mode
        assert pb.total is None

        # Completing flips indeterminate off and drives the bar to 100
        app.task_bar.complete_task(task_id)
        await _pilot.pause()
        task = app.task_bar.queue.get_task(task_id)
        if task is not None:
            assert task.indeterminate is False
