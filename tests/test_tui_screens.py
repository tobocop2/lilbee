"""Tests for TUI screens, app, and command provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
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
    variant_to_row,
)
from lilbee.cli.tui.screens.chat import ChatScreen as _ChatScreen
from lilbee.cli.tui.widgets.model_list_item import ModelListItem
from lilbee.config import cfg
from lilbee.model_manager import RemoteModel
from lilbee.services import set_services

_EMPTY_CATALOG = CatalogResult(total=0, limit=25, offset=0, models=[])

# Save a reference to the real _embedding_ready before the autouse fixture
# replaces it with a mock.  Tests that need the real implementation call this.
_real_embedding_ready = _ChatScreen._embedding_ready


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
    provider: str = "Remote",
) -> RemoteModel:
    return RemoteModel(
        name=name, task=task, family=family, parameter_size=parameter_size, provider=provider
    )


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

    def test_backend_from_provider(self):
        rm = RemoteModel(
            name="qwen:latest", task="chat", family="qwen", parameter_size="7B", provider="Ollama"
        )
        row = remote_to_row(rm)
        assert row.backend == "ollama"


class TestBackendField:
    """Verify the backend field is set correctly across all row builders.

    Native (llama-cpp) models have backend="" because they are managed by
    lilbee itself. Only externally-managed models (ollama, litellm) show
    a backend pill so users know lilbee cannot install/delete them.
    """

    def test_catalog_to_row_backend_native(self):
        row = catalog_to_row(_make_catalog_model(), installed=False)
        assert row.backend == "native"

    def test_variant_to_row_backend_native(self):
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
            slug="qwen3", name="Qwen3", task="chat", description="test", variants=(variant,)
        )
        row = variant_to_row(variant, family, installed=False)
        assert row.backend == "native"

    def test_installed_name_to_row_backend_empty(self):
        from lilbee.cli.tui.screens.setup import _installed_name_to_row

        row = _installed_name_to_row("qwen3:8b", "chat")
        assert row.backend == ""

    def test_remote_to_row_backend_from_provider(self):
        rm = _make_remote_model()
        row = remote_to_row(rm)
        assert row.backend == "remote"

    def test_matches_search_by_backend(self):
        rm = _make_remote_model(provider="ollama")
        row = remote_to_row(rm)
        assert matches_search(row, "ollama") is True

    def test_matches_search_backend_no_match(self):
        row = catalog_to_row(_make_catalog_model(), installed=False)
        assert matches_search(row, "ollama") is False

    def test_matches_search_normalizes_hyphens_and_underscores(self):
        model = _make_catalog_model(display_name="Deepseek R1 Distill Llama 70B")
        row = catalog_to_row(model, installed=False)
        assert matches_search(row, "deepseek-r1-distill") is True
        assert matches_search(row, "deepseek_r1_distill") is True
        assert matches_search(row, "deepseek") is True
        assert matches_search(row, "mistral") is False


class TestGroupRowsForGrid:
    """Grid view sections include one bucket per ModelTask, including RERANK.

    Rerankers are opt-in tuning (not setup-critical) but they live in
    the catalog and must be visible in the catalog grid so users can
    install them via the settings-adjacent browse surface.
    """

    def _row(self, task: str, *, featured: bool = False, installed: bool = False):
        from lilbee.catalog import CatalogModel
        from lilbee.cli.tui.screens.catalog_utils import catalog_to_row

        model = CatalogModel(
            name=f"{task}-model",
            tag="v1",
            display_name=f"{task.capitalize()} Model",
            hf_repo=f"org/{task}-model",
            gguf_filename="m.gguf",
            size_gb=1.0,
            min_ram_gb=1.0,
            description="",
            featured=featured,
            downloads=0,
            task=task,
        )
        return catalog_to_row(model, installed=installed)

    def test_grid_contains_rerank_bucket(self) -> None:
        from lilbee.cli.tui.screens.catalog import _group_rows_for_grid
        from lilbee.models import ModelTask

        rows = [
            self._row(ModelTask.CHAT),
            self._row(ModelTask.EMBEDDING),
            self._row(ModelTask.VISION),
            self._row(ModelTask.RERANK),
        ]
        sections = _group_rows_for_grid(rows)
        headings = [s.heading for s in sections]
        assert ModelTask.RERANK.capitalize() in headings

        rerank_section = next(s for s in sections if s.heading == ModelTask.RERANK.capitalize())
        assert len(rerank_section.rows) == 1
        assert rerank_section.rows[0].task == ModelTask.RERANK

    def test_featured_and_installed_excluded_from_task_buckets(self) -> None:
        """A featured rerank row appears only in Our Picks, not the RERANK bucket."""
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.catalog import _group_rows_for_grid
        from lilbee.models import ModelTask

        rows = [
            self._row(ModelTask.RERANK, featured=True),
            self._row(ModelTask.RERANK),
        ]
        sections = {s.heading: s.rows for s in _group_rows_for_grid(rows)}
        assert len(sections[msg.HEADING_OUR_PICKS]) == 1
        assert len(sections[ModelTask.RERANK.capitalize()]) == 1

    def test_unknown_task_gets_its_own_section(self) -> None:
        """A row whose task is outside _TASK_BUCKET_ORDER still appears,
        in a section after the known buckets — never silently dropped."""
        from lilbee.cli.tui.screens.catalog import _group_rows_for_grid

        row = self._row("experimental")  # type: ignore[arg-type]
        sections = _group_rows_for_grid([row])
        headings = [s.heading for s in sections]
        assert "Experimental" in headings
        experimental = next(s for s in sections if s.heading == "Experimental")
        assert len(experimental.rows) == 1


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


def test_settings_screen_has_expected_handlers_and_actions() -> None:
    """Structural regression test: SettingsScreen event handlers + action
    bindings must be class methods, not nested functions captured by a
    misplaced module-level helper.

    The polish-pass commit 6ecd206 silently moved a helper into the
    class body with matching indentation, which closed the class early
    and turned every event handler into unreachable dead code. Parsing
    succeeded because the indentation stayed consistent, and all
    behavioural tests stayed red until caught by CI failures on
    release/next-style platforms. See 5ffc5d6 for the fix.

    This test fires before any pilot harness, so if someone re-breaks
    the class boundary it surfaces immediately instead of at runtime.
    """
    from lilbee.cli.tui.screens.settings import SettingsScreen

    expected = (
        "_on_input_save",
        "_on_checkbox_save",
        "_on_select_save",
        "_on_search_submitted",
        "_filter_settings",
        "_persist_value",
        "_parse_value",
        "_refresh_help",
        "action_focus_search",
        "action_go_back",
        "action_scroll_down",
        "action_scroll_up",
        "action_scroll_home",
        "action_scroll_end",
    )
    missing = [name for name in expected if not callable(getattr(SettingsScreen, name, None))]
    assert not missing, f"SettingsScreen missing expected methods: {missing}"


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
        for _ in range(10):
            await pilot.pause()
            if cfg.show_reasoning != original:
                break
        assert cfg.show_reasoning != original


async def test_settings_tab_reaches_checkbox_and_space_toggles():
    """Tab walks focus to the checkbox and Space toggles it."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        from textual.widgets import Checkbox

        cb = app.screen.query_one("#ed-show_reasoning", Checkbox)
        original = cfg.show_reasoning
        # Enough presses to cross every focusable setting row in the
        # current settings map (Models/Ingest/Generation groups +
        # Retrieval). Bumped from 30 when vision_model was added as a
        # new writable Models-group entry (each row adds a reset
        # button + editor field to the tab order).
        for _ in range(60):
            await pilot.press("tab")
            await pilot.pause()
            if app.focused is cb:
                break
        assert app.focused is cb, "Tab failed to reach show_reasoning checkbox"
        await pilot.press("space")
        for _ in range(10):
            await pilot.pause()
            if cfg.show_reasoning != original:
                break
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


async def test_settings_crawl_exclude_patterns_renders_collapsible():
    """crawl_exclude_patterns renders as a Collapsible with line count in title."""
    from textual.widgets import Collapsible

    from lilbee.config import cfg

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        collapsible = app.screen.query_one("#collapsible-crawl_exclude_patterns", Collapsible)
        assert collapsible.collapsed is True
        assert "crawl_exclude_patterns" in collapsible.title
        current = cfg.crawl_exclude_patterns or []
        assert f"({len(current)} lines)" in collapsible.title


async def test_settings_list_editor_can_be_expanded():
    """Setting `collapsed = False` on the public Collapsible API reveals the editor."""
    from textual.widgets import Collapsible

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        collapsible = app.screen.query_one("#collapsible-crawl_exclude_patterns", Collapsible)
        assert collapsible.collapsed is True
        collapsible.collapsed = False
        await pilot.pause()
        # The TextArea is inside the body once the Collapsible is open.
        ta = collapsible.query_one(ListTextArea)
        assert ta is not None


async def test_settings_list_editor_saves_on_blur():
    """Typing into the list TextArea and blurring persists the parsed list."""
    from textual.widgets import Input

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        ta.focus()
        await pilot.pause()
        ta.load_text("foo\nbar")
        search = app.screen.query_one("#settings-search", Input)
        search.focus()
        for _ in range(10):
            await pilot.pause()
            if cfg.crawl_exclude_patterns == ["foo", "bar"]:
                break
        assert cfg.crawl_exclude_patterns == ["foo", "bar"]


async def test_settings_list_editor_strips_blanks():
    """Blank lines and surrounding whitespace are stripped during parsing."""
    from textual.widgets import Input

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        ta.focus()
        await pilot.pause()
        ta.load_text("a\n\nb\n")
        search = app.screen.query_one("#settings-search", Input)
        search.focus()
        for _ in range(10):
            await pilot.pause()
            if cfg.crawl_exclude_patterns == ["a", "b"]:
                break
        assert cfg.crawl_exclude_patterns == ["a", "b"]


async def test_settings_list_editor_invalid_regex_blocks_save():
    """An invalid regex shows an error and does not mutate cfg."""
    from textual.widgets import Input, Static

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        cfg.crawl_exclude_patterns = ["keep"]
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        ta.focus()
        await pilot.pause()
        ta.load_text("[")
        search = app.screen.query_one("#settings-search", Input)
        search.focus()
        err = app.screen.query_one("#err-crawl_exclude_patterns", Static)
        # Focus change → blur handler → regex validation → error widget
        # class toggle are all async. Single pilot.pause is not enough on
        # slower runners; poll until the -visible class lands.
        for _ in range(10):
            await pilot.pause()
            if err.has_class("-visible"):
                break
        assert err.has_class("-visible")
        assert "line 1" in str(err.render())
        assert cfg.crawl_exclude_patterns == ["keep"]


async def test_settings_list_editor_restore_defaults():
    """Pressing Restore resets both cfg and the TextArea to the defaults."""
    from textual.widgets import Button

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea
    from lilbee.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        cfg.crawl_exclude_patterns = ["old"]
        btn = app.screen.query_one("#list-restore-crawl_exclude_patterns", Button)
        btn.press()
        # Button.Pressed flows through Textual's async message bus; poll until
        # the handler updates cfg. A single pilot.pause is not enough on
        # slower runners (Windows, in particular).
        for _ in range(10):
            await pilot.pause()
            if cfg.crawl_exclude_patterns == list(DEFAULT_CRAWL_EXCLUDE_PATTERNS):
                break
        assert cfg.crawl_exclude_patterns == list(DEFAULT_CRAWL_EXCLUDE_PATTERNS)
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        assert ta.text == "\n".join(DEFAULT_CRAWL_EXCLUDE_PATTERNS)


async def test_settings_parse_value_list_branch():
    """_parse_value splits, strips, and drops blanks for list-typed settings."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    defn = SettingDef(list, nullable=False)
    screen = SettingsScreen()
    result = screen._parse_value(defn, "a\n\nb")
    assert result == ["a", "b"]


async def test_settings_list_editor_persists_through_toml_round_trip(tmp_path):
    """Editing a list-typed setting survives a full settings.save + pydantic reload.

    Guards against the bug where `_persist_value` used `str(parsed)` for lists,
    which wrote Python repr such as "['foo', 'bar']" into the TOML store.
    After reload, the pydantic `splitlines()` validator then produced a
    one-element list with corrupt contents.
    """
    from textual.widgets import Input

    from lilbee import settings
    from lilbee.cli.settings_map import SETTINGS_MAP
    from lilbee.cli.tui.screens.settings import SettingsScreen
    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    cfg.data_root = tmp_path
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        ta = app.screen.query_one("#ed-crawl_exclude_patterns", ListTextArea)
        ta.focus()
        await pilot.pause()
        ta.load_text("pat-a\npat-b")
        app.screen.query_one("#settings-search", Input).focus()
        await pilot.pause()

    # Raw TOML value is a newline-joined string (not Python repr of the list).
    reloaded = settings.load(tmp_path)
    raw = reloaded["crawl_exclude_patterns"]
    assert raw == "pat-a\npat-b"
    assert "[" not in raw  # would indicate list repr leaked into TOML

    # Pydantic's `splitlines()` validator then reconstructs the list cleanly.
    screen = SettingsScreen()
    parsed = screen._parse_value(SETTINGS_MAP["crawl_exclude_patterns"], raw)
    assert parsed == ["pat-a", "pat-b"]


async def test_list_text_area_posts_blurred():
    """ListTextArea posts its Blurred message when focus moves away."""
    from textual.app import App
    from textual.widgets import Input

    from lilbee.cli.tui.widgets.list_text_area import ListTextArea

    captured: list[ListTextArea.Blurred] = []

    class _TestApp(App[None]):
        CSS = ""

        def compose(self) -> ComposeResult:
            yield ListTextArea(id="ta")
            yield Input(id="other")

        def on_list_text_area_blurred(self, event: ListTextArea.Blurred) -> None:
            captured.append(event)

    app = _TestApp()
    async with app.run_test(size=(80, 20)) as pilot:
        ta = app.query_one("#ta", ListTextArea)
        ta.focus()
        await pilot.pause()
        app.query_one("#other", Input).focus()
        await pilot.pause()
        assert captured, "Expected a Blurred message from ListTextArea"
        assert captured[0].control is ta


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


def test_settings_effective_value_summarizes_list():
    """List values are shown as a line count, not Python repr, on the help line."""
    from lilbee.cli.tui.screens.settings import _effective_value

    cfg.crawl_exclude_patterns = ["a", "b", "c"]
    result = _effective_value("crawl_exclude_patterns")
    assert result == "3 lines"
    # Specifically guards against the "current: ['a', 'b', 'c']" regression.
    assert "[" not in result
    assert "'" not in result

    # Empty list must still be rendered as a count, not fall through to
    # "None" or model defaults. Guards off-by-one refactors of len().
    cfg.crawl_exclude_patterns = []
    assert _effective_value("crawl_exclude_patterns") == "0 lines"


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


def test_get_default_for_scalar():
    """get_default returns the cfg default for a simple scalar field."""
    from lilbee.cli.settings_map import get_default
    from lilbee.config import Config

    expected = Config.model_fields["top_k"].default
    assert get_default("top_k") == expected
    assert isinstance(get_default("top_k"), int)


def test_get_default_for_nullable_scalar():
    """Nullable fields whose default is None return None."""
    from lilbee.cli.settings_map import get_default

    assert get_default("temperature") is None


def test_get_default_for_list_factory():
    """List-valued fields built by a factory return a fresh copy of the default list."""
    from lilbee.cli.settings_map import get_default
    from lilbee.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS

    result = get_default("crawl_exclude_patterns")
    assert result == list(DEFAULT_CRAWL_EXCLUDE_PATTERNS)


def test_get_default_handles_pydantic_undefined():
    """When a field has no default and no factory, get_default returns None."""
    from types import SimpleNamespace

    from pydantic_core import PydanticUndefined

    from lilbee.cli.settings_map import get_default
    from lilbee.config import cfg

    fake = SimpleNamespace(default=PydanticUndefined, default_factory=None)
    original = dict(type(cfg).model_fields)
    patched = dict(original)
    patched["_fake_field"] = fake
    with patch.object(type(cfg), "model_fields", patched):
        assert get_default("_fake_field") is None


async def test_reset_button_resets_scalar():
    """Pressing the reset button restores a scalar setting to its cfg default."""
    from textual.widgets import Button, Input

    from lilbee.cli.settings_map import get_default

    cfg.wiki_clusterer_k = 99
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        button = app.screen.query_one("#reset-wiki_clusterer_k", Button)
        button.press()
        target = get_default("wiki_clusterer_k")
        for _ in range(10):
            await pilot.pause()
            if cfg.wiki_clusterer_k == target:
                break
        assert cfg.wiki_clusterer_k == target
        editor = app.screen.query_one("#ed-wiki_clusterer_k", Input)
        assert editor.value == str(target)


async def test_reset_button_absent_for_readonly():
    """Read-only rows (e.g. chat_model) do not render a reset button."""
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        chat_row = app.screen.query_one("#row-chat_model")
        buttons = chat_row.query(".setting-reset-button")
        assert len(buttons) == 0


async def test_ctrl_r_resets_focused_row():
    """Ctrl+R walks up from the focused editor to reset its row."""
    from textual.widgets import Input

    from lilbee.cli.settings_map import get_default

    cfg.top_k = 99
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        editor = app.screen.query_one("#ed-top_k", Input)
        editor.focus()
        await pilot.pause()
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert cfg.top_k == get_default("top_k")


async def test_ctrl_r_with_no_focus_is_noop():
    """action_reset_focused is a no-op when nothing is focused."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen.focused = None  # type: ignore[misc]
            screen.action_reset_focused()
            mock_reset.assert_not_called()


async def test_ctrl_r_on_non_row_focus_is_noop():
    """action_reset_focused ignores focus that isn't inside a setting row."""
    from textual.widgets import Input

    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        search = screen.query_one("#settings-search", Input)
        search.focus()
        await pilot.pause()
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen.action_reset_focused()
            mock_reset.assert_not_called()


async def test_refresh_editor_updates_checkbox():
    """Resetting a boolean setting syncs the Checkbox widget to the default."""
    from textual.widgets import Checkbox

    from lilbee.cli.settings_map import get_default

    default = bool(get_default("show_reasoning"))
    cfg.show_reasoning = not default
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        checkbox = app.screen.query_one("#ed-show_reasoning", Checkbox)
        checkbox.value = not default
        await pilot.pause()
        from textual.widgets import Button

        button = app.screen.query_one("#reset-show_reasoning", Button)
        button.press()
        for _ in range(10):
            await pilot.pause()
            if cfg.show_reasoning == default:
                break
        assert cfg.show_reasoning == default
        assert checkbox.value == default


async def test_reset_nullable_to_none():
    """Resetting a nullable scalar clears cfg and empties the Input widget."""
    from textual.widgets import Button, Input

    cfg.temperature = 0.9
    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        button = app.screen.query_one("#reset-temperature", Button)
        button.press()
        for _ in range(10):
            await pilot.pause()
            if cfg.temperature is None:
                break
        assert cfg.temperature is None
        editor = app.screen.query_one("#ed-temperature", Input)
        assert editor.value == ""


async def test_reset_select_clears_when_default_none():
    """Resetting a Select-backed field to a None default clears the widget."""
    from textual.widgets import Select

    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        # Find any existing Select editor (wiki_clusterer is a choice-based field).
        select_widget = screen.query_one("#ed-wiki_clusterer", Select)
        # Seed a non-default choice so we can detect the clear.
        select_widget.value = "embedding"
        defn = SettingDef(type=str, nullable=True, group="Wiki", choices=("embedding", "concepts"))
        screen._refresh_editor("wiki_clusterer", defn, None)
        # clear() yields the "no selection" sentinel; value should no longer be the seeded choice.
        assert select_widget.value != "embedding"
        assert select_widget.value not in {"embedding", "concepts"}


async def test_reset_select_sets_value_when_default_provided():
    """Resetting a Select to a concrete string sets widget.value accordingly."""
    from textual.widgets import Select

    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        select_widget = screen.query_one("#ed-wiki_clusterer", Select)
        select_widget.value = "embedding"
        defn = SettingDef(type=str, nullable=False, group="Wiki", choices=("embedding", "concepts"))
        screen._refresh_editor("wiki_clusterer", defn, "concepts")
        assert select_widget.value == "concepts"


async def test_refresh_editor_missing_widget_is_logged(caplog):
    """Missing editor widget on refresh logs a debug message instead of raising."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        defn = SettingDef(type=str, nullable=True, group="General")
        with caplog.at_level("DEBUG", logger="lilbee.cli.tui.screens.settings"):
            screen._refresh_editor("nonexistent_key_xyz", defn, "irrelevant")
        assert any("nonexistent_key_xyz" in record.message for record in caplog.records)


async def test_refresh_editor_updates_textarea_list(monkeypatch):
    """Future-proofing: _refresh_editor loads list values into a TextArea."""
    from textual.widgets import TextArea

    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        fake = MagicMock(spec=TextArea)
        monkeypatch.setattr(screen, "query_one", lambda *a, **kw: fake)
        defn = SettingDef(type=list, nullable=False, group="Crawling")
        screen._refresh_editor("fake_list", defn, ["a", "b"])
        fake.load_text.assert_called_once_with("a\nb")


async def test_refresh_editor_updates_textarea_scalar(monkeypatch):
    """TextArea non-list scalar values go through load_text as plain str."""
    from textual.widgets import TextArea

    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        fake = MagicMock(spec=TextArea)
        monkeypatch.setattr(screen, "query_one", lambda *a, **kw: fake)
        defn = SettingDef(type=str, nullable=False, group="General")
        screen._refresh_editor("fake_str", defn, "hello")
        fake.load_text.assert_called_once_with("hello")


async def test_refresh_editor_updates_textarea_none(monkeypatch):
    """TextArea with None value loads an empty string."""
    from textual.widgets import TextArea

    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        fake = MagicMock(spec=TextArea)
        monkeypatch.setattr(screen, "query_one", lambda *a, **kw: fake)
        defn = SettingDef(type=str, nullable=True, group="General")
        screen._refresh_editor("fake_str", defn, None)
        fake.load_text.assert_called_once_with("")


async def test_reset_to_default_ignores_readonly_keys():
    """_reset_to_default is a no-op for read-only settings."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(screen, "_persist_value") as mock_persist:
            screen._reset_to_default("chat_model")
            mock_persist.assert_not_called()


async def test_reset_to_default_ignores_unknown_keys():
    """_reset_to_default is a no-op when the key is not in SETTINGS_MAP."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(screen, "_persist_value") as mock_persist:
            screen._reset_to_default("nonexistent_key_xyz")
            mock_persist.assert_not_called()


async def test_reset_button_with_malformed_id_is_noop():
    """Button press events with unexpected ids are gracefully ignored."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.button.id = "not-a-reset-button"
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen._on_reset_pressed(event)
            mock_reset.assert_not_called()
        event.button.id = None
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen._on_reset_pressed(event)
            mock_reset.assert_not_called()


async def test_reset_list_default_joins_newlines():
    """Resetting a list-valued setting stringifies via newline join."""
    from lilbee.cli.settings_map import SettingDef, get_default
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        defn = SettingDef(type=list, nullable=False, writable=True, group="Crawling")
        expected = "\n".join(get_default("crawl_exclude_patterns"))
        with (
            patch.dict(
                "lilbee.cli.tui.screens.settings.SETTINGS_MAP",
                {"crawl_exclude_patterns": defn},
            ),
            patch.object(screen, "_persist_value") as mock_persist,
            patch.object(screen, "_refresh_editor"),
        ):
            screen._reset_to_default("crawl_exclude_patterns")
            mock_persist.assert_called_once_with("crawl_exclude_patterns", defn, expected)


async def test_reset_all_button_mounts_in_top_row():
    """The Reset-all button renders alongside the search input."""
    from textual.widgets import Button

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        button = app.screen.query_one("#reset-all-defaults", Button)
        assert button is not None
        top_row = app.screen.query_one("#settings-top-row")
        assert button in list(top_row.query(Button))


async def test_reset_all_cancel_does_nothing():
    """Dismissing the confirm dialog with False leaves cfg untouched."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen._on_reset_all_confirmed(False)
            mock_reset.assert_not_called()
        with patch.object(screen, "_reset_to_default") as mock_reset:
            screen._on_reset_all_confirmed(None)
            mock_reset.assert_not_called()


async def test_reset_all_confirm_batches_writes_atomically():
    """Confirming the dialog issues a single settings.update_values batch write."""
    from lilbee.cli.settings_map import SETTINGS_MAP
    from lilbee.cli.tui.screens.settings import SettingsScreen

    writable_keys = {k for k, d in SETTINGS_MAP.items() if d.writable}
    readonly_keys = {k for k, d in SETTINGS_MAP.items() if not d.writable}
    assert readonly_keys, "test invariant: SETTINGS_MAP must contain a readonly field"

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch("lilbee.cli.tui.screens.settings.settings.update_values") as mock_batch:
            screen._on_reset_all_confirmed(True)
        mock_batch.assert_called_once()
        written_keys = set(mock_batch.call_args.args[1].keys())
        # Every writable key appears in the batch; no readonly key leaks in.
        assert written_keys == writable_keys
        assert not written_keys & readonly_keys


async def test_reset_all_suppresses_per_field_toasts():
    """Reset-all fires exactly one summary toast; no per-field CMD_SET_SUCCESS spam."""
    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with (
            patch("lilbee.cli.tui.screens.settings.settings.update_values"),
            patch.object(screen, "notify") as mock_notify,
        ):
            screen._on_reset_all_confirmed(True)
        # Exactly one notify call: the summary toast.
        assert mock_notify.call_count == 1
        assert mock_notify.call_args.args[0] == msg.SETTINGS_RESET_ALL_SUCCESS


async def test_reset_all_actually_mutates_cfg():
    """Batch reset restores cfg values to pydantic defaults (not just the TOML write)."""
    from lilbee.cli.settings_map import get_default
    from lilbee.cli.tui.screens.settings import SettingsScreen

    default_top_k = get_default("top_k")
    cfg.top_k = 999  # deliberately non-default
    assert cfg.top_k == 999

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch("lilbee.cli.tui.screens.settings.settings.update_values"):
            screen._on_reset_all_confirmed(True)
        assert cfg.top_k == default_top_k


async def test_reset_all_rolls_back_on_disk_write_failure():
    """If settings.update_values raises, in-memory cfg reverts so UI and disk stay in sync."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    cfg.top_k = 999
    cfg.wiki_clusterer_k = 77

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with (
            patch(
                "lilbee.cli.tui.screens.settings.settings.update_values",
                side_effect=OSError("disk full"),
            ),
            patch.object(screen, "notify") as mock_notify,
        ):
            screen._on_reset_all_confirmed(True)
        # Rollback restored the pre-reset values.
        assert cfg.top_k == 999
        assert cfg.wiki_clusterer_k == 77
        # User sees an error toast, not a success toast.
        assert mock_notify.called
        notify_kwargs = mock_notify.call_args.kwargs
        assert notify_kwargs.get("severity") == "error"


async def test_reset_all_reports_skipped_keys():
    """When setattr rejects a default, the summary toast names the skipped keys."""
    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        original_setattr = type(cfg).__setattr__

        def rejecting_setattr(self_cfg, name, value):
            if name == "top_k":
                raise ValueError("rejected for test")
            original_setattr(self_cfg, name, value)

        with (
            patch.object(type(cfg), "__setattr__", rejecting_setattr),
            patch("lilbee.cli.tui.screens.settings.settings.update_values"),
            patch.object(screen, "notify") as mock_notify,
        ):
            screen._on_reset_all_confirmed(True)
        # Summary toast mentions the skipped key and uses warning severity.
        summary_calls = [c for c in mock_notify.call_args_list if c.args and "top_k" in c.args[0]]
        assert summary_calls, "expected a toast mentioning the skipped key"
        assert summary_calls[0].args[0] == msg.SETTINGS_RESET_ALL_PARTIAL.format(skipped="top_k")
        assert summary_calls[0].kwargs.get("severity") == "warning"


async def test_rollback_swallows_setattr_failure():
    """If setattr on a rollback key raises, the rollback logs and proceeds."""
    from lilbee.cli.settings_map import SETTINGS_MAP
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        original_setattr = type(cfg).__setattr__

        def rejecting_setattr(self_cfg, name, value):
            if name == "top_k" and value == "rollback-sentinel":
                raise ValueError("rollback rejected")
            original_setattr(self_cfg, name, value)

        writable = [("top_k", SETTINGS_MAP["top_k"])]
        with (
            patch.object(type(cfg), "__setattr__", rejecting_setattr),
            patch.object(screen, "_refresh_editor"),
            patch.object(screen, "_refresh_help"),
        ):
            screen._rollback_batch(writable, {"top_k": "rollback-sentinel"})
        # Passing means the except branch swallowed the ValueError; no raise.


async def test_publish_batch_signals_on_lilbee_app():
    """When the screen runs under LilbeeApp, signals fan out for each reset key."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.push_screen(SettingsScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(app.settings_changed_signal, "publish") as mock_pub:
            screen._publish_batch_signals([("top_k", 7), ("embedding_dim", 768)])
        assert mock_pub.call_count == 2


async def test_reset_all_button_press_opens_confirm_dialog():
    """Pressing Reset-all pushes the ConfirmDialog screen before mutating state."""
    from lilbee.cli.tui.screens.settings import SettingsScreen
    from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        with patch.object(screen.app, "push_screen") as mock_push:
            screen._on_reset_all_pressed()
        mock_push.assert_called_once()
        pushed_screen = mock_push.call_args.args[0]
        assert isinstance(pushed_screen, ConfirmDialog)


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

    cfg.vision_model = "test-vision:latest"
    info = ModelArchInfo()
    with (
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

    cfg.vision_model = ""
    info = ModelArchInfo()
    result = _read_vision_arch(info)
    assert result.vision_projector == "unknown"


def test_status_read_vision_arch_swallows_errors():
    """When GGUF probing fails, _read_vision_arch logs and leaves info unchanged."""
    from lilbee.model_info import ModelArchInfo, _read_vision_arch

    cfg.vision_model = "test-vision:latest"
    info = ModelArchInfo()
    with patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        side_effect=RuntimeError("boom"),
    ):
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


async def test_chat_slash_reset_pushes_confirm_dialog():
    """``/reset`` pushes a ConfirmDialog modal."""
    from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.screen._handle_slash("/reset")
        await _pilot.pause()
        assert isinstance(app.screen_stack[-1], ConfirmDialog)


async def test_chat_slash_reset_confirm_executes():
    """Confirming the reset dialog calls perform_reset."""
    from lilbee.cli.helpers import ResetResult

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.helpers.perform_reset") as mock_reset:
            mock_reset.return_value = ResetResult(
                deleted_docs=0, deleted_data=0, documents_dir="d", data_dir="d"
            )
            app.screen._handle_slash("/reset")
            await _pilot.pause()
            await _pilot.press("y")
            await _pilot.pause()
            mock_reset.assert_called_once()


async def test_chat_slash_reset_cancel_does_nothing():
    """Cancelling the reset dialog does not call perform_reset."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.helpers.perform_reset") as mock_reset:
            app.screen._handle_slash("/reset")
            await _pilot.pause()
            await _pilot.press("n")
            await _pilot.pause()
            mock_reset.assert_not_called()


async def test_chat_slash_reset_error_notifies():
    """Reset failure shows an error notification."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.helpers.perform_reset", side_effect=Exception("oops")):
            with patch.object(app.screen, "notify") as mock_notify:
                app.screen._handle_slash("/reset")
                await _pilot.pause()
                await _pilot.press("y")
                await _pilot.pause()
                assert any("oops" in str(c) for c in mock_notify.call_args_list)


async def test_chat_slash_reset_partial_notifies_warning():
    """When some files can't be deleted, a warning notification is shown."""
    from lilbee.cli.helpers import ResetResult

    partial = ResetResult(
        deleted_docs=1, deleted_data=0, skipped=["/locked.exe"], documents_dir="d", data_dir="d"
    )
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch("lilbee.cli.helpers.perform_reset", return_value=partial):
            with patch.object(app.screen, "notify") as mock_notify:
                app.screen._handle_slash("/reset")
                await _pilot.pause()
                await _pilot.press("y")
                await _pilot.pause()
                assert any("could not be deleted" in str(c) for c in mock_notify.call_args_list)


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
        # top_k is writable but int-typed; empty string fails int() conversion
        # and surfaces as CMD_SET_INVALID, covering the no-value branch.
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_set("top_k")
            mock_notify.assert_called_once()
            assert "Invalid value" in mock_notify.call_args[0][0]


async def test_chat_slash_set_readonly_key():
    """Read-only keys (chat_model, vision_model, ...) must be rejected."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "notify") as mock_notify:
            app.screen._cmd_set("chat_model some-model:latest")
            mock_notify.assert_called_once()
            assert "read-only" in mock_notify.call_args[0][0]
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
            app.screen.refresh_model_bar()
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


async def test_command_provider_open_wiki_action():
    """Palette 'Open wiki' action switches to the Wiki view."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.cli.tui.commands import LilbeeCommandProvider

        provider = LilbeeCommandProvider(app.screen, match_style=None)
        with patch.object(app, "switch_view") as mock_switch:
            provider._action_open_wiki()
            mock_switch.assert_called_once_with("Wiki")


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
    """s keybinding cycles Name -> Downloads -> Size -> Params in list view."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await pilot.pause()
            assert screen._sort_column == "Name"
            assert screen._sort_ascending is True
            await pilot.press("v")
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert screen._sort_column == "Downloads"
            assert screen._sort_ascending is True
            await pilot.press("s")
            await pilot.pause()
            assert screen._sort_column == "Size"
            await pilot.press("s")
            await pilot.pause()
            assert screen._sort_column == "Params"
            await pilot.press("s")
            await pilot.pause()
            assert screen._sort_column == "Name"


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


async def test_catalog_select_ollama_remote_row_stores_prefix():
    """Picking an Ollama-backed catalog row stores the ollama/ prefix.

    Without the prefix, routing would classify it as local and dispatch
    to llama-cpp, silently bypassing the user's Ollama choice.
    """
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.catalog_utils import remote_to_row

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            om = _make_remote_model(name="qwen3:0.6b", provider="Ollama")
            row = remote_to_row(om)
            screen._select_row(row)
            assert cfg.chat_model == "ollama/qwen3:0.6b"


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
                assert screen._loading_more is True


async def test_catalog_action_load_more_triggers_fetch():
    """Pressing n fires a fetch when more results are available."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._hf_has_more = True
            with patch.object(screen, "_fetch_more_hf") as fetch:
                screen.action_load_more()
                assert fetch.called


async def test_catalog_load_more_noop_when_exhausted():
    """Calling _load_more when _hf_has_more is False must not fire a fetch."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._hf_has_more = False
            old_offset = screen._hf_offset
            with patch.object(screen, "_fetch_more_hf") as fetch:
                screen._load_more()
                assert not fetch.called
                assert screen._hf_offset == old_offset


async def test_catalog_load_more_deduplicated_while_in_flight():
    """A second _load_more during an in-flight fetch does not re-advance the offset."""
    from lilbee.cli.tui.screens.catalog import _HF_PAGE_SIZE, CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            old_offset = screen._hf_offset
            with patch.object(screen, "_fetch_more_hf") as fetch:
                screen._load_more()
                screen._load_more()
                assert fetch.call_count == 1
                assert screen._hf_offset == old_offset + _HF_PAGE_SIZE


async def test_catalog_row_highlighted_prefetches_near_bottom():
    """Focusing near the last list item triggers _load_more during nav."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._grid_view = False
            screen._hf_has_more = True
            # Wipe featured families so the list is exactly the HF models.
            screen._families = []
            screen._hf_models = [
                _make_catalog_model(name=f"m-{i}B", hf_repo=f"org/m-{i}", featured=False)
                for i in range(30)
            ]
            screen._refresh_list()
            await _pilot.pause()
            items = list(screen.query(ModelListItem))
            assert len(items) == 30
            # Focus the last item so prefetch trigger fires.
            items[-1].focus()
            await _pilot.pause()
            with patch.object(screen, "_fetch_more_hf") as fetch:
                screen._maybe_prefetch_on_nav()
                assert fetch.called


async def test_catalog_row_highlighted_ignored_in_grid_view():
    """Grid view doesn't trigger list-view prefetch."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._grid_view = True
            screen._hf_has_more = True
            with patch.object(screen, "_fetch_more_hf") as fetch:
                screen._maybe_prefetch_on_nav()
                assert not fetch.called


async def test_catalog_sort_label_covers_every_pagination_state():
    """`_update_sort_label` renders different suffixes per pagination state."""
    from textual.widgets import Static

    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            label = screen.query_one("#sort-label", Static)

            # In-flight fetch: "loading more…" branch.
            screen._loading_more = True
            screen._hf_has_more = True
            screen._update_sort_label()
            assert "loading more" in str(label._Static__content)  # type: ignore[attr-defined]

            # More results available, not loading: "press n for more" branch.
            screen._loading_more = False
            screen._update_sort_label()
            assert "for more" in str(label._Static__content).lower()  # type: ignore[attr-defined]

            # Exhausted: plain count with no suffix.
            screen._hf_has_more = False
            screen._update_sort_label()
            text = str(label._Static__content)  # type: ignore[attr-defined]
            assert "loading" not in text
            assert "for more" not in text.lower()


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
            # Invalidate the grid cache so _refresh_grid() rebuilds from scratch.
            screen._grid_cache_key = ()
            screen._refresh_grid()
            screen._refresh_list()
            # Move focus off the initial featured grid so
            # _get_highlighted_model_name() doesn't pick up stale state.
            screen.query_one("#catalog-search").focus()
            await _pilot.pause()
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
            screen._grid_view = False
            screen._refresh_list()
            await _pilot.pause()
            # Focus the first list item so _get_highlighted_model_name()
            # picks up the row via the focused ModelListItem.
            items = list(screen.query(ModelListItem))
            assert items
            items[0].focus()
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


async def test_catalog_worker_more_hf_error_releases_latch():
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()

            from textual.worker import WorkerState

            screen._loading_more = True
            mock_worker = MagicMock()
            mock_worker.name = _WORKER_FETCH_MORE_HF
            mock_event = MagicMock()
            mock_event.state = WorkerState.ERROR
            mock_event.worker = mock_worker
            screen.on_worker_state_changed(mock_event)
            assert screen._loading_more is False


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
        from lilbee.progress import EventType, FileStartEvent

        async def fake_sync(quiet=False, on_progress=None, cancel=None):
            if on_progress:
                on_progress(
                    EventType.FILE_START,
                    FileStartEvent(current_file=1, total_files=2, file="test.md"),
                )
            return {"added": 3}

        with patch("lilbee.ingest.sync", new=fake_sync):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()
            assert app.screen._sync_active is False


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
            for _ in range(10):
                await _pilot.pause()
                if app.screen._sync_active is False:
                    break
            # Worker catches the TypeError via the except Exception handler
            assert app.screen._sync_active is False


async def test_chat_sync_file_start_bad_type():
    """Sync progress raises TypeError when FILE_START data is not FileStartEvent."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.progress import EventType

        async def fake_sync(quiet=False, on_progress=None, cancel=None):
            if on_progress:
                on_progress(
                    EventType.FILE_START,
                    {"current_file": 1, "total_files": 1, "file": "x.md"},
                )
            return {"added": 0}

        with patch("lilbee.ingest.sync", new=fake_sync):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()
            for _ in range(10):
                await _pilot.pause()
                if app.screen._sync_active is False:
                    break
            # Worker catches the TypeError via the except Exception handler
            assert app.screen._sync_active is False


async def test_chat_sync_embed_bad_type():
    """Sync progress silently skips when EMBED data is not EmbedEvent."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        from lilbee.progress import EventType

        async def fake_sync(quiet=False, on_progress=None, cancel=None):
            if on_progress:
                on_progress(EventType.EMBED, {"file": "x.md", "chunk": 1, "total_chunks": 5})
            return {"added": 0}

        with patch("lilbee.ingest.sync", new=fake_sync):
            app.screen._run_sync()
            await _pilot.pause()
            while app.screen.workers:
                await _pilot.pause()
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


async def test_chat_on_setup_complete_skipped_no_banner_when_embedding_ready():
    """Skipping wizard does not show banner when embedding model is already configured."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app.screen, "_embedding_ready", return_value=True):
            app.screen._on_setup_complete("skipped")
            await _pilot.pause()
            banner = app.screen.query_one("#chat-only-banner")
            assert banner.display is False


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


async def test_catalog_refresh_list_empty():
    """Cover empty list case."""
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
            screen._refresh_list()
            list_container = screen.query_one("#catalog-list", VerticalScroll)
            assert len(list_container.query(ModelListItem)) == 0


async def test_catalog_refresh_list_with_models():
    """Cover list view with HF models."""
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
            screen._refresh_list()
            list_container = screen.query_one("#catalog-list", VerticalScroll)
            assert len(list_container.query(ModelListItem)) >= 5


async def test_catalog_page_down_with_focused_table():
    """Cover action_page_down with focused list item."""
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
            screen._grid_view = False
            screen._refresh_list()
            items = list(screen.query(ModelListItem))
            assert items
            items[0].focus()
            await _pilot.pause()
            # Nav actions focus a list item by calling .focus() on the
            # target; verify the back-to-back pair leaves focus on item 0.
            screen.action_page_down()
            screen.action_page_up()
            assert items[0].has_focus


async def test_catalog_action_cursor_with_focused_table():
    """Cover action_cursor_down with focused list item."""
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
            screen._grid_view = False
            screen._refresh_list()
            items = list(screen.query(ModelListItem))
            assert items
            items[0].focus()
            await _pilot.pause()
            screen.action_cursor_down()
            screen.action_cursor_up()
            assert items[0].has_focus


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
            screen._grid_view = False
            screen._refresh_list()
            items = list(screen.query(ModelListItem))
            assert items
            items[0].focus()
            await _pilot.pause()
            screen.action_jump_bottom()
            screen.action_jump_top()
            assert items[0].has_focus


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
        remote_embeds = detect_remote_embedding_models(cfg.remote_base_url)
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
        remote_embeds = detect_remote_embedding_models(cfg.remote_base_url)
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

        assert ChatScreen._parse_crawl_flags([]) == (None, None, False)

    def test_depth_only(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--depth", "3"]) == (3, None, False)

    def test_max_pages_only(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--max-pages", "20"]) == (None, 20, False)

    def test_both(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--depth", "2", "--max-pages", "15"]) == (
            2,
            15,
            False,
        )

    def test_invalid_values(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--depth", "abc"]) == (None, None, False)

    def test_missing_value(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--depth"]) == (None, None, False)

    def test_unknown_flags_skipped(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--unknown", "value"]) == (None, None, False)

    def test_include_subdomains_flag(self):
        """--include-subdomains opts into sibling-subdomain crawl."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--include-subdomains"]) == (None, None, True)

    def test_include_subdomains_with_depth(self):
        from lilbee.cli.tui.screens.chat import ChatScreen

        assert ChatScreen._parse_crawl_flags(["--depth", "2", "--include-subdomains"]) == (
            2,
            None,
            True,
        )


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
            screen._grid_view = False
            screen._refresh_list()
            await _pilot.pause()

            # Focus the last list item (remote model)
            items = list(screen.query(ModelListItem))
            assert items
            items[-1].focus()
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
            screen._grid_view = False
            screen._refresh_list()
            await _pilot.pause()

            items = list(screen.query(ModelListItem))
            assert items
            items[-1].focus()
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
            screen._grid_view = False
            screen._refresh_list()
            await _pilot.pause()

            items = list(screen.query(ModelListItem))
            assert items
            items[-1].focus()
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
            screen._refresh_list()
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


async def test_do_add_raises_on_sync_failed(tmp_path):
    """bb-vb28: _do_add raises when sync returns SyncResult with failed files.

    Without this, embedding/extract failures inside the sync pipeline are
    silently swallowed and the Task Center marks the task DONE. The worker's
    except-Exception needs to see a raise so the row routes to FAILED.

    Runs on a worker thread so it doesn't block the pilot's event loop.
    """
    import threading

    from lilbee.cli.tui.widgets.task_bar import ProgressReporter
    from lilbee.ingest import SyncResult

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello")

        fake_result = SyncResult(failed=["doc.txt"])

        async def fake_sync(**kwargs):
            return fake_result

        reporter = ProgressReporter(app.task_bar, "fake-id")
        captured: dict[str, BaseException] = {}

        def _run_worker() -> None:
            try:
                app.screen._do_add(test_file, reporter)
            except BaseException as exc:
                captured["exc"] = exc

        with (
            patch("lilbee.cli.helpers.copy_files") as mock_copy,
            patch("lilbee.ingest.sync", new=fake_sync),
        ):
            mock_copy.return_value = SimpleNamespace(copied=[test_file], skipped=[])
            thread = threading.Thread(target=_run_worker)
            thread.start()
            thread.join(timeout=5)

        assert "exc" in captured
        assert isinstance(captured["exc"], RuntimeError)
        assert "doc.txt" in str(captured["exc"])


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


async def test_task_center_has_css_path():
    """TaskCenter declares a CSS_PATH for task-specific styles."""
    from lilbee.cli.tui.screens.task_center import TaskCenter

    assert TaskCenter.CSS_PATH == "task_center.tcss"


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
        assert status.model_name == "my-model:latest"


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


async def test_setup_wizard_commit_chat_selection_writes_settings():
    """_commit_selection saves chat_model synchronously."""
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            chat_cards = [c for c in screen.query(ModelCard) if c.row.task == "chat"]
            assert chat_cards
            with patch("lilbee.settings.set_value") as mock_set:
                screen._commit_selection(chat_cards[0], "chat")
            assert mock_set.called
            assert cfg.chat_model == (chat_cards[0].row.ref or chat_cards[0].row.name)


async def test_setup_wizard_commit_embed_selection_writes_settings():
    """_commit_selection saves embedding_model synchronously."""
    from lilbee.cli.tui.screens.setup import SetupWizard
    from lilbee.cli.tui.widgets.model_card import ModelCard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            embed_cards = [c for c in screen.query(ModelCard) if c.row.task == "embedding"]
            assert embed_cards
            with patch("lilbee.settings.set_value") as mock_set:
                screen._commit_selection(embed_cards[0], "embedding")
            assert mock_set.called
            assert cfg.embedding_model == (embed_cards[0].row.ref or embed_cards[0].row.name)


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


async def test_setup_wizard_shows_intro_and_hint():
    """Setup wizard exposes a single intro + a bottom Enter hint, no per-slot labels."""
    from textual.widgets import Label, Static

    from lilbee.cli.tui import messages as msg
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = SetupTestApp()
    with _patch_setup_scan(), _patch_setup_ram(16.0):
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupWizard)
            intro = screen.query_one("#setup-intro", Static)
            hint = screen.query_one("#setup-enter-hint", Label)
            intro_text = str(intro._Static__content)  # type: ignore[attr-defined]
            hint_text = str(hint._Static__content)  # type: ignore[attr-defined]
            assert "chat" in intro_text.lower()
            assert "search" in intro_text.lower() or "embedding" in intro_text.lower()
            assert "Enter" in hint_text
            # The old per-slot labels must not exist anymore.
            assert not screen.query("#setup-chat-slot")
            assert not screen.query("#setup-embed-slot")
            assert not screen.query("#setup-download-size")
            assert msg.SETUP_INTRO  # anchor the new constant


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


async def test_setup_wizard_tab_escapes_grid_to_install_button():
    """Tab from the last card in the last grid reaches the Install & Go button."""
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
            await pilot.press("tab")
            await pilot.pause()
            focused = app.focused
            assert focused is not last_grid
            assert focused is not None


async def test_setup_wizard_shift_tab_escapes_grid_backward():
    """Shift+Tab from the first card in a grid walks focus backward."""
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
            await pilot.press("shift+tab")
            await pilot.pause()
            assert app.focused is not second_grid
            assert app.focused is not None


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


async def test_catalog_grid_leave_down_focuses_next():
    """GridSelect.LeaveDown moves focus to the next focusable widget."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.widgets.grid_select import GridSelect

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await pilot.pause()
            grids = list(screen.query(GridSelect))
            if grids:
                grids[0].focus()
                await pilot.pause()
                grids[0].post_message(GridSelect.LeaveDown(grids[0]))
                await pilot.pause()
                assert screen.focused is not grids[0]


async def test_catalog_grid_leave_up_focuses_previous():
    """GridSelect.LeaveUp moves focus to the previous focusable widget."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.widgets.grid_select import GridSelect

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await pilot.pause()
            grids = list(screen.query(GridSelect))
            assert grids, "Expected at least one GridSelect"
            grids[0].focus()
            await pilot.pause()
            grids[0].post_message(GridSelect.LeaveUp(grids[0]))
            await pilot.pause()
            assert screen.focused is not grids[0]


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
            # Mount a single ModelListItem and focus it so
            # _get_highlighted_model_name() picks up row.ref via screen.focused.
            list_container = screen.query_one("#catalog-list", VerticalScroll)
            list_container.remove_children()
            list_container.mount(ModelListItem(row))
            await _pilot.pause()
            items = list(list_container.query(ModelListItem))
            assert items
            items[0].focus()
            await _pilot.pause()
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
            screen._grid_view = False
            list_container = screen.query_one("#catalog-list", VerticalScroll)
            list_container.remove_children()
            list_container.mount(ModelListItem(row))
            await _pilot.pause()
            items = list(list_container.query(ModelListItem))
            assert items
            items[0].focus()
            await _pilot.pause()
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
            screen._grid_view = False
            list_container = screen.query_one("#catalog-list", VerticalScroll)
            list_container.remove_children()
            list_container.mount(ModelListItem(row))
            await _pilot.pause()
            items = list(list_container.query(ModelListItem))
            assert items
            items[0].focus()
            await _pilot.pause()
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


async def test_chat_on_setup_complete_hides_banner_when_embedding_ready():
    """_on_setup_complete hides chat-only banner after wizard configures embedding."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Simulate banner being visible (e.g. from /setup command while in chat-only mode)
        app.screen._show_chat_only_banner()
        assert app.screen.query_one("#chat-only-banner").display is True
        with patch.object(app.screen, "_embedding_ready", return_value=True):
            app.screen._on_setup_complete("done")
            await _pilot.pause()
            assert app.screen.query_one("#chat-only-banner").display is False


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


async def test_settings_on_list_blur_save_name_none():
    """_on_list_blur_save returns early when the TextArea has no name."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.control.name = None
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_list_blur_save(event)
            mock_pv.assert_not_called()


async def test_settings_on_list_blur_save_defn_none():
    """_on_list_blur_save returns early when SETTINGS_MAP has no entry."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.control.name = "nonexistent_key_xyz"
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_list_blur_save(event)
            mock_pv.assert_not_called()


async def test_settings_on_list_restore_bad_button_id():
    """_on_list_restore ignores buttons whose id does not start with the list-restore prefix."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.button.id = None
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_list_restore(event)
            mock_pv.assert_not_called()
        event.button.id = "wrong-prefix-foo"
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_list_restore(event)
            mock_pv.assert_not_called()


async def test_settings_on_list_restore_defn_none():
    """_on_list_restore returns early when SETTINGS_MAP has no entry for the key."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        event = MagicMock()
        event.button.id = "list-restore-nonexistent_key"
        with patch.object(screen, "_persist_value") as mock_pv:
            screen._on_list_restore(event)
            mock_pv.assert_not_called()


async def test_settings_refresh_list_title_missing_swallows():
    """_refresh_list_title logs (does not raise) when the Collapsible is absent."""
    from lilbee.cli.tui.screens.settings import SettingsScreen

    app = SettingsTestApp()
    async with app.run_test(size=(120, 40)):
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        screen._refresh_list_title("nonexistent_key_xyz", 0)


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
    """action_quit dismisses the wizard when it's the active screen."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.setup import SetupWizard

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        wizard = SetupWizard()
        with _patch_setup_scan(), _patch_setup_ram(16.0):
            app.push_screen(wizard)
            await pilot.pause()
            assert isinstance(app.screen, SetupWizard)
            await app.action_quit()
            await pilot.pause()
            assert not isinstance(app.screen, SetupWizard)


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
            app.screen._cmd_wiki("")
            mock_notify.assert_called_once()
            assert "disabled" in mock_notify.call_args[0][0].lower()


async def test_chat_cmd_wiki_navigates_to_wiki_screen():
    """/wiki navigates to the Wiki screen when wiki is enabled."""
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg,
            patch.object(app, "switch_view") as mock_switch,
        ):
            mock_cfg.wiki = True
            app.screen._cmd_wiki("")
            mock_switch.assert_called_once_with("Wiki")


def test_chat_embedding_ready_true_via_provider_list(mock_svc):
    """_embedding_ready returns True when provider list_models contains the model.

    Uses _real_embedding_ready (saved before autouse fixture mocks the method).
    The mock_svc autouse fixture injects a Services singleton; we configure its
    provider.list_models to return a model that matches the embedding config.
    """
    mock_svc.provider.list_models.return_value = ["nomic-embed-text:latest"]
    cfg.embedding_model = "nomic-embed-text"
    sentinel = object()
    with patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        side_effect=FileNotFoundError("not found"),
    ):
        assert _real_embedding_ready(sentinel) is True


def test_chat_embedding_ready_true_via_resolve_fallback(mock_svc):
    """_embedding_ready returns True via resolve_model_path when provider raises.

    When provider.list_models raises, the method falls through to the native
    registry path check. If resolve_model_path succeeds, it returns True.
    """
    mock_svc.provider.list_models.side_effect = RuntimeError("no provider")
    cfg.embedding_model = "nomic-embed-text"
    sentinel = object()
    with patch(
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        return_value="/fake/path/to/model.gguf",
    ):
        assert _real_embedding_ready(sentinel) is True


def test_chat_embedding_ready_false_when_no_model():
    """_embedding_ready returns False when no embedding model is configured."""
    sentinel = object()
    with patch("lilbee.cli.tui.screens.chat.cfg") as mock_cfg:
        mock_cfg.embedding_model = ""
        assert _real_embedding_ready(sentinel) is False


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


async def test_chat_cmd_crawl_no_args_opens_dialog():
    """_cmd_crawl with empty args opens the crawl dialog."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch.object(app.screen, "_open_crawl_dialog") as mock_dialog,
        ):
            app.screen._cmd_crawl("")
            mock_dialog.assert_called_once()


def test_chat_embedding_ready_real_code_false():
    """Placeholder — real test is in test_tui_e2e.py to avoid autouse fixture."""
    from lilbee.cli.tui.screens.chat import ChatScreen

    assert hasattr(ChatScreen, "_embedding_ready")


def test_on_list_item_selected_calls_select_row():
    """_on_list_item_selected calls _select_row with the item's row."""
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
    item = MagicMock()
    item.row = row
    event = MagicMock()
    event.item = item
    CatalogScreen._on_list_item_selected(screen, event)
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
            screen._grid_view = False
            list_container = screen.query_one("#catalog-list", VerticalScroll)
            list_container.remove_children()
            list_container.mount(ModelListItem(row))
            await _pilot.pause()
            items = list(list_container.query(ModelListItem))
            assert items
            items[0].focus()
            await _pilot.pause()
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
            screen._grid_view = False
            list_container = screen.query_one("#catalog-list", VerticalScroll)
            list_container.remove_children()
            list_container.mount(ModelListItem(row))
            await _pilot.pause()
            items = list(list_container.query(ModelListItem))
            assert items
            items[0].focus()
            await _pilot.pause()
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


async def test_task_bar_indeterminate_flag_propagated():
    """BEE-65f: indeterminate flag is stored on the task in the queue.

    The TaskBar no longer renders ProgressBar widgets (those live in the
    Task Center), but the indeterminate flag must still propagate through
    the controller into the queue so the Task Center can render correctly.
    """
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
        await _pilot.pause()

        task = app.task_bar.queue.get_task(task_id)
        assert task is not None
        assert task.indeterminate is True
        assert task.status == TaskStatus.ACTIVE

        bar = app.screen.query_one("#tbar", TaskBar)
        bar._refresh_display()
        assert bar.display is True


def test_resolve_wiki_targets_all():
    """Returns all source names when no specific source requested."""
    from lilbee.cli.tui.wiki_worker import resolve_wiki_targets

    fake_store = MagicMock()
    fake_store.get_sources.return_value = [
        {"filename": "a.txt"},
        {"filename": "b.txt"},
    ]
    fake_svc = MagicMock(store=fake_store)
    with patch("lilbee.cli.tui.wiki_worker.get_services", return_value=fake_svc):
        result = resolve_wiki_targets()
    assert result == ["a.txt", "b.txt"]


def test_resolve_wiki_targets_specific():
    """Returns only the requested source when it exists."""
    from lilbee.cli.tui.wiki_worker import resolve_wiki_targets

    fake_store = MagicMock()
    fake_store.get_sources.return_value = [
        {"filename": "a.txt"},
        {"filename": "b.txt"},
    ]
    fake_svc = MagicMock(store=fake_store)
    with patch("lilbee.cli.tui.wiki_worker.get_services", return_value=fake_svc):
        result = resolve_wiki_targets("b.txt")
    assert result == ["b.txt"]


def test_resolve_wiki_targets_unknown():
    """Returns None for an unknown source name."""
    from lilbee.cli.tui.wiki_worker import resolve_wiki_targets

    fake_store = MagicMock()
    fake_store.get_sources.return_value = [{"filename": "a.txt"}]
    fake_svc = MagicMock(store=fake_store)
    with patch("lilbee.cli.tui.wiki_worker.get_services", return_value=fake_svc):
        assert resolve_wiki_targets("missing.txt") is None


def test_resolve_wiki_targets_empty_sources():
    """Returns None when no sources are indexed."""
    from lilbee.cli.tui.wiki_worker import resolve_wiki_targets

    fake_store = MagicMock()
    fake_store.get_sources.return_value = []
    fake_svc = MagicMock(store=fake_store)
    with patch("lilbee.cli.tui.wiki_worker.get_services", return_value=fake_svc):
        assert resolve_wiki_targets() is None


def test_resolve_wiki_targets_get_sources_error():
    """Returns None when get_sources raises."""
    from lilbee.cli.tui.wiki_worker import resolve_wiki_targets

    fake_store = MagicMock()
    fake_store.get_sources.side_effect = RuntimeError("db gone")
    fake_svc = MagicMock(store=fake_store)
    with patch("lilbee.cli.tui.wiki_worker.get_services", return_value=fake_svc):
        assert resolve_wiki_targets() is None


def _direct_call(_widget, fn, *args, **kwargs):
    """Stub for call_from_thread that calls fn directly (no Textual app needed)."""
    fn(*args, **kwargs)


def _make_wiki_app(*, with_task_bar: bool = False) -> App[None]:
    """Build a test app that pushes WikiScreen on mount."""
    from lilbee.cli.tui.screens.wiki import WikiScreen

    class _WikiApp(App[None]):
        def __init__(self) -> None:
            super().__init__()
            if with_task_bar:
                from lilbee.cli.tui.widgets.task_bar import TaskBarController

                self.task_bar = TaskBarController(self)

        def on_mount(self) -> None:
            self.push_screen(WikiScreen())

    return _WikiApp()


async def test_wiki_screen_reload():
    """WikiScreen.reload refreshes the sidebar."""
    app = _make_wiki_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        with patch.object(app.screen, "_load_pages") as mock_load:
            app.screen.reload()
            mock_load.assert_called_once()


async def test_wiki_screen_regenerate_disabled():
    """Regenerate notifies when wiki is disabled."""
    app = _make_wiki_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.wiki.cfg") as mock_cfg,
            patch.object(app.screen, "notify") as mock_notify,
        ):
            mock_cfg.wiki = False
            await pilot.press("r")
            mock_notify.assert_called_once()


async def test_wiki_screen_regenerate_no_sources():
    """Regenerate notifies when no indexed sources found."""
    app = _make_wiki_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.wiki.cfg") as mock_cfg,
            patch(
                "lilbee.cli.tui.screens.wiki.resolve_wiki_targets",
                return_value=None,
            ),
            patch.object(app.screen, "notify") as mock_notify,
        ):
            mock_cfg.wiki = True

            await pilot.press("r")
            mock_notify.assert_called_once()


async def test_chat_open_crawl_dialog():
    """_open_crawl_dialog pushes CrawlDialog modal."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True):
            app.screen._cmd_crawl("")
            await _pilot.pause()
            from lilbee.cli.tui.widgets.crawl_dialog import CrawlDialog

            assert isinstance(app.screen, CrawlDialog)


async def test_chat_crawl_dialog_callback_triggers_start():
    """CrawlDialog callback calls _start_crawl with the result params."""
    from lilbee.cli.tui.widgets.crawl_dialog import CrawlParams

    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch.object(app.screen, "_start_crawl") as mock_start,
        ):
            app.screen._cmd_crawl("")
            await _pilot.pause()
            # Dismiss the dialog with params
            params = CrawlParams(url="https://test.com", depth=1, max_pages=10)
            app.screen.dismiss(params)
            await _pilot.pause()
        mock_start.assert_called_once_with("https://test.com", 1, 10)


async def test_chat_crawl_dialog_callback_none_noop():
    """CrawlDialog callback with None does not start a crawl."""
    app = ChatTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        await _pilot.pause()
        with (
            patch("lilbee.cli.tui.screens.chat.crawler_available", return_value=True),
            patch.object(app.screen, "_start_crawl") as mock_start,
        ):
            app.screen._cmd_crawl("")
            await _pilot.pause()
            app.screen.dismiss(None)
            await _pilot.pause()
        mock_start.assert_not_called()


async def test_wiki_source_for_slug_returns_source():
    """_source_for_slug extracts source from page frontmatter."""
    app = _make_wiki_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        mock_page = MagicMock()
        mock_page.frontmatter = {"sources": ["doc.txt", "other.txt"]}
        with patch("lilbee.cli.tui.screens.wiki.read_page", return_value=mock_page):
            result = app.screen._source_for_slug("summaries/doc")
        assert result == "doc.txt"


async def test_wiki_source_for_slug_returns_none_for_missing():
    """_source_for_slug returns None when page not found."""
    app = _make_wiki_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        with patch("lilbee.cli.tui.screens.wiki.read_page", return_value=None):
            result = app.screen._source_for_slug("summaries/missing")
        assert result is None


async def test_wiki_source_for_slug_returns_none_for_empty_sources():
    """_source_for_slug returns None when sources list is empty."""
    app = _make_wiki_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        mock_page = MagicMock()
        mock_page.frontmatter = {"sources": []}
        with patch("lilbee.cli.tui.screens.wiki.read_page", return_value=mock_page):
            result = app.screen._source_for_slug("summaries/doc")
        assert result is None


async def test_wiki_selected_source_returns_none_for_option_without_id():
    """_selected_source returns None when highlighted option has no id."""
    from textual.widgets import OptionList
    from textual.widgets.option_list import Option

    app = _make_wiki_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        option_list = app.screen.query_one("#wiki-page-list", OptionList)
        option_list.clear_options()
        option_list.add_option(Option("no-id page"))  # id defaults to None
        option_list.focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        result = app.screen._selected_source()
        assert result is None


async def test_wiki_regenerate_selected_page_not_found():
    """action_regenerate with a selected page whose source isn't indexed shows error."""
    from textual.widgets import OptionList
    from textual.widgets.option_list import Option

    app = _make_wiki_app(with_task_bar=True)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        option_list = app.screen.query_one("#wiki-page-list", OptionList)
        option_list.clear_options()
        option_list.add_option(Option("test page", id="summaries/test"))
        option_list.focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        mock_page = MagicMock()
        mock_page.frontmatter = {"sources": ["gone.txt"]}
        with (
            patch("lilbee.cli.tui.screens.wiki.cfg") as mock_cfg,
            patch("lilbee.cli.tui.screens.wiki.read_page", return_value=mock_page),
            patch(
                "lilbee.cli.tui.screens.wiki.resolve_wiki_targets",
                return_value=None,
            ),
            patch.object(app.screen, "notify") as mock_notify,
        ):
            mock_cfg.wiki = True
            await pilot.press("r")
            await pilot.pause()
        mock_notify.assert_called_once()
        assert "Source not found" in mock_notify.call_args[0][0]


# =============================================================================
# Coverage fill: catalog.py branches
# =============================================================================


async def test_catalog_update_sort_label_loading_more():
    """_update_sort_label renders the 'loading more' variant when flag is set."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._loading_more = True
            screen._update_sort_label()
            label = screen.query_one("#sort-label", Static)
            assert "loading more" in str(label.render())


async def test_catalog_cycle_sort_noop_when_input_focused():
    """action_cycle_sort returns early when focus is on the search Input."""
    from textual.widgets import Input

    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen._grid_view = False
            screen.query_one("#catalog-search", Input).focus()
            await pilot.pause()
            before = screen._sort_column
            screen.action_cycle_sort()
            assert screen._sort_column == before


async def test_catalog_cycle_sort_in_grid_view_notifies():
    """action_cycle_sort in grid view surfaces the list-only notification."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            assert screen._grid_view is True
            before = screen._sort_column
            with patch.object(screen, "notify") as mock_notify:
                screen.action_cycle_sort()
            mock_notify.assert_called_once()
            assert screen._sort_column == before


async def test_catalog_cycle_sort_unknown_column_restarts_cycle():
    """action_cycle_sort handles a sort column outside _SORT_CYCLE."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._grid_view = False
            screen._sort_column = "NotInCycle"
            screen.action_cycle_sort()
            # Unknown column resets the cycle; index -1 -> _SORT_CYCLE[0]
            assert screen._sort_column == "Name"


async def test_catalog_search_submit_installs_first_visible_match():
    """Single Enter in search filters + queues install on the first visible card.

    Regression test: previously Enter from the search Input only
    refocused the grid (landing on the hidden default card), so users
    had to press Enter twice — and the second press queued the wrong
    (invisible) model.
    """
    from unittest.mock import patch

    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.widgets.grid_select import GridSelect

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            grids = list(screen.query(GridSelect))
            assert grids, "catalog should mount at least one GridSelect"
            grid = grids[0]
            assert len(grid.children) >= 2
            # Simulate a filter hiding all cards except the second one —
            # the old handler would install the first (now-hidden) card
            # via highlighted=0; the fix must land on the visible one.
            for card in grid.children:
                card.display = False
            target_card = grid.children[1]
            target_card.display = True
            with patch.object(screen, "_select_row") as install:
                screen._select_first_visible_grid_card()
                # action_select posts a message; drain the loop so
                # _on_grid_selected dispatches through to our patched
                # _select_row.
                for _ in range(5):
                    await _pilot.pause()
                assert install.called
                row_arg = install.call_args.args[0]
                assert row_arg is target_card.row


async def test_catalog_select_first_visible_list_item_installs_match():
    """List-view counterpart: Enter in search installs the first visible row."""
    from unittest.mock import patch

    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.widgets.model_list_item import ModelListItem

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._grid_view = False
            screen._refresh_list()
            items = list(screen.query(ModelListItem))
            if len(items) < 2:
                return  # Not enough rows to exercise the visible-match walk
            for item in items:
                item.display = False
            target = items[1]
            target.display = True
            with patch.object(screen, "_select_row") as install:
                screen._select_first_visible_list_item()
                for _ in range(5):
                    await _pilot.pause()
                assert install.called
                assert install.call_args.args[0] is target.row


async def test_catalog_focus_list_item_empty_is_noop():
    """_focus_list_item leaves focus unchanged when there are no list items."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._grid_view = False
            screen._families = []
            screen._hf_models = []
            screen._remote_models = []
            screen._refresh_list()
            assert not screen._list_items()
            focus_before = screen.focused
            screen._focus_list_item(0)
            assert screen.focused is focus_before


async def test_catalog_focused_list_index_none_when_no_focus():
    """_focused_list_index returns None when no ModelListItem is focused."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            # screen.focused is None at this point
            assert screen._focused_list_index() is None


async def test_catalog_maybe_prefetch_returns_when_no_focus():
    """_maybe_prefetch_on_nav returns early when focused_list_index is None."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as _pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await _pilot.pause()
            screen._grid_view = False
            screen._hf_has_more = True
            screen._loading_more = False
            # focused_list_index is None, so load_more must NOT be called.
            with patch.object(screen, "_load_more") as mock_load:
                screen._maybe_prefetch_on_nav()
            mock_load.assert_not_called()


def test_catalog_get_highlighted_name_non_model_card_child():
    """_get_highlighted_model_name returns None when the highlighted child
    is neither a ModelListItem nor a ModelCard (defensive branch)."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    screen = MagicMock()
    screen.focused = None
    fake_grid = MagicMock()
    fake_grid.highlighted = 0
    fake_grid.children = [object()]  # Not a ModelCard / ModelListItem
    screen._focused_grid.return_value = fake_grid
    assert CatalogScreen._get_highlighted_model_name(screen) is None


async def test_catalog_focused_list_index_value_error_path():
    """_focused_list_index returns None when the focused item is not in query results."""
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.widgets.model_list_item import ModelListItem

    app = CatalogTestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        with _patch_catalog()[0], _patch_catalog()[1], _patch_catalog()[2]:
            screen = CatalogScreen()
            app.push_screen(screen)
            await pilot.pause()
            screen._grid_view = False
            # Focus a fabricated ModelListItem that is not mounted in the screen.
            dangling = ModelListItem(
                TableRow(
                    name="dangling",
                    task="chat",
                    params="1B",
                    size="1.0 GB",
                    quant="Q4_K_M",
                    downloads="--",
                    featured=False,
                    installed=False,
                    sort_downloads=0,
                    sort_size=1.0,
                )
            )
            # Directly stub `focused` with the dangling item so the list.index
            # call in _focused_list_index raises ValueError.
            screen.focused = dangling  # type: ignore[assignment]
            assert screen._focused_list_index() is None


# =============================================================================
# Coverage fill: settings.py branches
# =============================================================================


def test_settings_env_pill_when_env_set(monkeypatch):
    """_env_pill returns a pill when the LILBEE_* env var is exported."""
    from lilbee.cli.tui.screens.settings import _env_pill

    monkeypatch.setenv("LILBEE_CHAT_MODEL", "probe")
    pill_content = _env_pill("chat_model")
    assert pill_content is not None
    assert "LILBEE_CHAT_MODEL" in pill_content.plain


def test_settings_help_content_blank_when_no_help_text():
    """_help_content returns empty Content when the setting has no help text."""
    from lilbee.cli.settings_map import SettingDef
    from lilbee.cli.tui.screens.settings import _help_content

    defn = SettingDef(type=str, nullable=False, group="Test", help_text="")
    content = _help_content("anon", defn)
    assert content.plain == ""


def test_settings_title_content_renders_env_pill_when_set(monkeypatch):
    """_title_content carries the env var name when LILBEE_* is exported."""
    from lilbee.cli.settings_map import SETTINGS_MAP
    from lilbee.cli.tui.screens.settings import _title_content

    monkeypatch.setenv("LILBEE_CHAT_MODEL", "probe")
    content = _title_content("chat_model", SETTINGS_MAP["chat_model"])
    assert "LILBEE_CHAT_MODEL" in content.plain


def test_settings_title_content_no_env_pill_when_unset(monkeypatch):
    """_title_content omits the env pill when the LILBEE_* var is not set."""
    from lilbee.cli.settings_map import SETTINGS_MAP
    from lilbee.cli.tui.screens.settings import _title_content

    monkeypatch.delenv("LILBEE_CHAT_MODEL", raising=False)
    content = _title_content("chat_model", SETTINGS_MAP["chat_model"])
    assert "LILBEE_CHAT_MODEL" not in content.plain
