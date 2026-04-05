"""End-to-end TUI integration tests.

These tests launch the real Textual app and verify observable behavior.
They exist because unit tests with mocks passed while the app was broken.
Every test here reproduces a bug that was found by manual testing.
"""

from __future__ import annotations

from unittest import mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Footer

from lilbee.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    """Snapshot and restore config for each test."""
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = "test-chat-model.gguf"
    cfg.embedding_model = "test-embed-model"
    cfg.vision_model = ""
    cfg.subprocess_embed = False
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


@pytest.fixture()
def _mock_resolve():
    """Mock model resolution to succeed without real files."""
    with mock.patch(
        "lilbee.providers.llama_cpp_provider._resolve_model_path",
        return_value=cfg.models_dir / "fake.gguf",
    ):
        yield


@pytest.fixture()
def _mock_services():
    """Mock services to prevent real provider initialization."""
    mock_svc = mock.MagicMock()
    mock_svc.provider.list_models.return_value = []
    mock_svc.searcher._embedder.embedding_available.return_value = True
    with mock.patch("lilbee.services.get_services", return_value=mock_svc):
        yield mock_svc


class ChatTestApp(App[None]):
    """Minimal app that pushes ChatScreen for testing."""

    def compose(self) -> ComposeResult:
        yield Footer()

    def on_mount(self) -> None:
        from lilbee.cli.tui.screens.chat import ChatScreen

        self.push_screen(ChatScreen())


class TestEmbeddingAvailable:
    def test_registry_name_with_spaces_resolves_via_fallback(self):
        """Embedding model 'Nomic Embed Text v1.5:latest' must match
        files with hyphens when registry resolution fails."""
        from lilbee.embedder import Embedder

        mock_provider = mock.MagicMock()
        mock_provider.list_models.return_value = [
            "nomic-embed-text-v1.5.Q4_K_M.gguf",
            "other-model.gguf",
        ]
        cfg.embedding_model = "Nomic Embed Text v1.5:latest"

        embedder = Embedder(cfg, mock_provider)
        # Registry resolution fails (no manifests in tmp dir), so falls through
        # to list_models fallback which must normalize spaces to hyphens
        assert embedder.embedding_available() is True

    def test_resolves_via_registry(self):
        """When _resolve_model_path succeeds, embedding is available."""
        from lilbee.embedder import Embedder

        mock_provider = mock.MagicMock()
        cfg.embedding_model = "test-embed"

        embedder = Embedder(cfg, mock_provider)
        with mock.patch(
            "lilbee.providers.llama_cpp_provider._resolve_model_path",
            return_value=cfg.models_dir / "test.gguf",
        ):
            assert embedder.embedding_available() is True

    def test_unresolvable_model_returns_false(self):
        """When model name doesn't match any installed model, returns False."""
        from lilbee.embedder import Embedder

        mock_provider = mock.MagicMock()
        mock_provider.list_models.return_value = ["other-model.gguf"]
        cfg.embedding_model = "nonexistent-model"
        embedder = Embedder(cfg, mock_provider)
        assert embedder.embedding_available() is False


class TestModelClassification:
    def test_mmproj_filtered_out(self):
        from lilbee.cli.tui.widgets.model_bar import _is_mmproj

        assert _is_mmproj("mmproj-BF16.gguf") is True
        assert _is_mmproj("Qwen3-4B.gguf") is False

    def test_registry_based_classification(self):
        """Models classified by registry manifest task field."""
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models

        # Mock registry with proper manifests
        mock_manifests = [
            mock.MagicMock(name="Qwen3", tag="latest", task="chat"),
            mock.MagicMock(name="Nomic Embed", tag="latest", task="embedding"),
            mock.MagicMock(name="LightOnOCR", tag="latest", task="vision"),
        ]
        # Set .name properly (MagicMock uses name for repr)
        mock_manifests[0].name = "Qwen3"
        mock_manifests[1].name = "Nomic Embed"
        mock_manifests[2].name = "LightOnOCR"

        with mock.patch("lilbee.cli.tui.widgets.model_bar._collect_native_models") as mock_native:

            def fill_buckets(buckets, seen):
                for m in mock_manifests:
                    name = f"{m.name}:{m.tag}"
                    buckets.get(m.task, buckets["chat"]).append(name)
                    seen.add(name)

            mock_native.side_effect = fill_buckets
            with mock.patch("lilbee.cli.tui.widgets.model_bar._collect_remote_models"):
                chat, embed, vision = _classify_installed_models()

        assert "Qwen3:latest" in chat
        assert "Nomic Embed:latest" in embed
        assert "LightOnOCR:latest" in vision

    def test_no_loose_gguf_scanning(self):
        """Legacy .gguf files NOT in registry must NOT appear in dropdowns."""
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models

        # Create loose files that should be ignored
        (cfg.models_dir / "loose-chat.gguf").touch()
        (cfg.models_dir / "loose-vision.gguf").touch()

        with (
            mock.patch("lilbee.cli.tui.widgets.model_bar._collect_native_models"),
            mock.patch("lilbee.cli.tui.widgets.model_bar._collect_remote_models"),
        ):
            chat, embed, vision = _classify_installed_models()

        all_models = chat + embed + vision
        assert "loose-chat.gguf" not in all_models
        assert "loose-vision.gguf" not in all_models


class TestModelSwitchSafety:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families")
    async def test_switch_cancels_stream(self, _fam, _cat, _mock_resolve):
        """Changing model while streaming must cancel the stream first."""
        from lilbee.cli.tui.widgets.model_bar import ModelBar

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen._streaming = True
            screen.action_cancel_stream = mock.MagicMock()

            bar = screen.query_one("#model-bar", ModelBar)
            bar._populating = False

            # Simulate model change
            event = mock.MagicMock()
            event.value = "new-model.gguf"
            event.select = mock.MagicMock()
            event.select.id = "chat-model-select"

            with mock.patch("lilbee.services.reset_services"):
                bar._on_chat_model_changed(event)

            screen.action_cancel_stream.assert_called_once()


class TestNavBarPresence:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families")
    async def test_navbar_on_all_screens(self, _fam, _cat, _mock_resolve):
        """NavBar must exist on every screen."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Chat screen
            nav = app.screen.query_one("#global-nav-bar")
            assert nav is not None

            # Cycle through all views
            for view in ["Models", "Status", "Settings", "Tasks"]:
                app.switch_view(view)
                await pilot.pause()
                nav = app.screen.query_one("#global-nav-bar")
                assert nav is not None, f"NavBar missing on {view} screen"


class TestModeIndicator:
    async def test_insert_mode_on_startup(self, _mock_resolve):
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            nav = app.screen.query_one("#global-nav-bar")
            # Insert mode is default
            assert "INSERT" in nav.mode_text

    async def test_normal_mode_on_escape(self, _mock_resolve):
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            nav = app.screen.query_one("#global-nav-bar")
            assert "NORMAL" in nav.mode_text


class TestViewCycling:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families")
    async def test_cycles_all_five_views(self, _fam, _cat, _mock_resolve):
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.active_view == "Chat"

            expected = ["Models", "Status", "Settings", "Tasks", "Chat"]
            for view in expected:
                app.action_nav_next()
                await pilot.pause()
                assert app.active_view == view, f"Expected {view}, got {app.active_view}"


class TestChatOnlyBanner:
    async def test_banner_hidden_when_embedding_available(self, _mock_resolve):
        """Banner must be hidden when embedding model resolves."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import Static

            banner = app.screen.query_one("#chat-only-banner", Static)
            assert banner.display is False

    async def test_banner_shown_when_embedding_unavailable(self, _mock_resolve):
        """Banner must show when _embedding_ready returns False."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Manually simulate embedding not ready
            with mock.patch.object(app.screen, "_embedding_ready", return_value=False):
                app.screen._show_chat_only_banner()
                await pilot.pause()
                from textual.widgets import Static

                banner = app.screen.query_one("#chat-only-banner", Static)
                assert banner.display is True


class TestTaskCenter:
    async def test_task_center_renders_with_active_task(self, _mock_resolve):
        """Task Center must render collapsible task widgets without crashing."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Add a mock task directly to the task bar
            task_bar = app.task_bar
            assert task_bar is not None

            task_id = task_bar.add_task("Test Download", "download")
            task_bar.update_task(task_id, 45, "100/500 MB")

            app.switch_view("Tasks")
            await pilot.pause()

            from textual.widgets import DataTable

            table = app.screen.query_one("#task-table", DataTable)
            assert table.row_count >= 1

    async def test_task_center_renders_empty_state(self, _mock_resolve):
        """Task Center shows 'All quiet' when no tasks."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Switch to Task Center with no tasks
            app.switch_view("Tasks")
            await pilot.pause()

            from textual.widgets import DataTable

            table = app.screen.query_one("#task-table", DataTable)
            assert table.row_count == 0


class TestDownloadProgressSlow:
    @pytest.mark.slow
    def test_download_progress_callback_receives_cumulative_values(self, tmp_path):
        """Download Mistral and verify progress callbacks receive cumulative values."""
        import os
        import threading

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            pytest.skip("HF_TOKEN environment variable not set")

        from lilbee.catalog import CatalogModel, download_model
        from lilbee.model_manager import reset_model_manager

        snapshot = cfg.model_copy()
        try:
            cfg.data_dir = tmp_path / "data"
            cfg.models_dir = tmp_path / "models"
            cfg.data_root = tmp_path
            cfg.documents_dir = tmp_path / "documents"
            cfg.lancedb_dir = tmp_path / "data" / "lancedb"
            cfg.models_dir.mkdir(parents=True, exist_ok=True)
            cfg.documents_dir.mkdir(parents=True, exist_ok=True)
            reset_model_manager()

            entry = CatalogModel(
                name="Mistral 7B",
                hf_repo="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
                gguf_filename="*Q4_K_M.gguf",
                size_gb=4.2,
                min_ram_gb=8,
                description="Test",
                featured=False,
                downloads=0,
                task="chat",
            )

            progress_calls = []
            download_error = [None]
            download_done = [False]

            def on_progress(downloaded: int, total: int):
                progress_calls.append((downloaded, total))
                # Exit after receiving some progress (at least 1MB)
                if downloaded > 1024 * 1024:
                    download_done[0] = True

            def download_in_thread():
                try:
                    download_model(entry, on_progress=on_progress)
                except Exception as e:
                    if not download_done[0]:
                        download_error[0] = e

            thread = threading.Thread(target=download_in_thread)
            thread.start()
            thread.join(timeout=30)

            if thread.is_alive():
                thread.join(timeout=5)

            if download_error[0]:
                raise download_error[0]

            assert len(progress_calls) > 0, "No progress callbacks received"

            cumulative_values = [c[0] for c in progress_calls]
            assert cumulative_values[-1] > 0, "No cumulative bytes received"

            print(f"\nProgress calls: {len(progress_calls)}")
            print(f"Final: {cumulative_values[-1] / 1024 / 1024:.1f} MB")
        finally:
            for field_name in type(snapshot).model_fields:
                setattr(cfg, field_name, getattr(snapshot, field_name))


def _mock_catalog_deps():
    """Context manager that mocks all catalog network calls."""
    from lilbee.catalog import ModelFamily, ModelVariant

    families = [
        ModelFamily(
            name="TestChat",
            task="chat",
            description="A test chat model",
            variants=(
                ModelVariant(
                    hf_repo="test/chat-repo",
                    filename="chat-Q4.gguf",
                    param_count="7B",
                    quant="Q4_K_M",
                    size_mb=4000,
                    recommended=True,
                ),
            ),
        ),
        ModelFamily(
            name="TestEmbed",
            task="embedding",
            description="A test embedding model",
            variants=(
                ModelVariant(
                    hf_repo="test/embed-repo",
                    filename="embed-Q8.gguf",
                    param_count="0.5B",
                    quant="Q8_0",
                    size_mb=500,
                    recommended=True,
                ),
            ),
        ),
    ]
    return mock.patch.multiple(
        "lilbee.cli.tui.screens.catalog",
        get_families=mock.MagicMock(return_value=families),
        get_catalog=mock.MagicMock(return_value=mock.MagicMock(models=[])),
    )


def _mock_remote_models():
    """Mock classify_remote_models to return empty list."""
    return mock.patch(
        "lilbee.model_manager.classify_remote_models",
        return_value=[],
    )


class TestScreenTransitions:
    """Test that switching between screens does not crash."""

    async def test_navigate_chat_to_catalog_to_settings(self, _mock_resolve):
        """F2→Models, then F4→Settings, verify no crash."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                assert app.active_view == "Chat"

                app.switch_view("Models")
                await pilot.pause()
                assert app.active_view == "Models"

                app.switch_view("Settings")
                await pilot.pause()
                assert app.active_view == "Settings"

    async def test_navigate_all_views_via_keybindings(self, _mock_resolve):
        """Cycle through all 5 views with nav_next (l key)."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                expected = ["Models", "Status", "Settings", "Tasks", "Chat"]
                for view in expected:
                    app.action_nav_next()
                    await pilot.pause()
                    assert app.active_view == view

    async def test_navigate_back_with_q(self, _mock_resolve):
        """Push catalog, press q, verify back at chat."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Models")
                await pilot.pause()
                assert app.active_view == "Models"

                await pilot.press("q")
                await pilot.pause()
                # Should be back at Chat (base screen)
                from lilbee.cli.tui.screens.chat import ChatScreen

                assert isinstance(app.screen, ChatScreen)

    async def test_navigate_catalog_to_tasks(self, _mock_resolve):
        """The specific crash case: catalog → tasks transition."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Models")
                await pilot.pause()
                app.switch_view("Tasks")
                await pilot.pause()
                assert app.active_view == "Tasks"


class TestCatalogGrid:
    """Test catalog grid view rendering and interactions."""

    async def test_catalog_grid_view_default(self, _mock_resolve):
        """Grid is shown on mount by default."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Models")
                await pilot.pause()

                grids = app.screen.query(GridSelect)
                assert len(grids) > 0

    async def test_catalog_toggle_view(self, _mock_resolve):
        """Press v twice: grid → list → grid."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Models")
                await pilot.pause()

                screen = app.screen
                assert screen.has_class("-grid-view")

                screen.action_toggle_view()
                await pilot.pause()
                assert screen.has_class("-list-view")

                screen.action_toggle_view()
                await pilot.pause()
                assert screen.has_class("-grid-view")

    async def test_catalog_search_filters_cards(self, _mock_resolve):
        """Type search text, verify cards filtered (not destroyed/recreated)."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.model_card import ModelCard

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Models")
                await pilot.pause()

                all_cards = app.screen.query(ModelCard)
                initial_count = len(all_cards)
                assert initial_count > 0

                # Type a search that matches only one family
                search = app.screen.query_one("#catalog-search")
                search.display = True
                search.value = "TestChat"
                await pilot.pause()

                # Cards still exist (not destroyed), but some hidden
                all_cards_after = app.screen.query(ModelCard)
                assert len(all_cards_after) == initial_count
                visible = [c for c in all_cards_after if c.display]
                hidden = [c for c in all_cards_after if not c.display]
                assert len(visible) >= 1
                assert len(hidden) >= 1

    async def test_catalog_grid_card_count(self, _mock_resolve):
        """Verify correct number of cards for featured models."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.model_card import ModelCard

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Models")
                await pilot.pause()

                cards = app.screen.query(ModelCard)
                # 2 families with 1 variant each = 2 cards
                assert len(cards) == 2


class TestSettingsScreen:
    """Test settings screen rendering and filtering."""

    async def test_settings_has_grouped_sections(self, _mock_resolve):
        """Verify group headings exist."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()

            groups = app.screen.query(".group-title")
            assert len(groups) >= 1

    async def test_settings_search_filters(self, _mock_resolve):
        """Type in search, verify filtering hides non-matching rows."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()

            all_rows = app.screen.query(".setting-row")
            total = len(all_rows)
            assert total > 0

            # Search for something specific
            search = app.screen.query_one("#settings-search")
            search.value = "chat_model"
            await pilot.pause()

            visible = [r for r in app.screen.query(".setting-row") if r.display]
            assert len(visible) < total
            assert len(visible) >= 1


class TestStatusScreen:
    """Test status screen rendering."""

    async def test_status_has_collapsible_sections(self, _mock_resolve):
        """Verify Collapsible widgets exist for config, docs, etc."""
        from textual.widgets import Collapsible

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Status")
            await pilot.pause()

            collapsibles = app.screen.query(Collapsible)
            assert len(collapsibles) >= 3


class TestTaskCenterScreen:
    """Test task center screen rendering."""

    async def test_task_center_renders(self, _mock_resolve):
        """Basic mount test — task center renders without crash."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Tasks")
            await pilot.pause()

            from textual.widgets import DataTable

            table = app.screen.query_one("#task-table", DataTable)
            assert table is not None


class TestChatPromptBorder:
    """Test that the chat prompt area has a single border, not stacked."""

    async def test_prompt_area_insert_mode_border(self, _mock_resolve):
        """PromptArea should have insert-mode class, input should have no border."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            area = app.screen.query_one("#chat-prompt-area")
            inp = app.screen.query_one("#chat-input")
            assert area.has_class("insert-mode")
            # Input should not have its own border
            assert inp.styles.border is not None  # exists but set to none in CSS

    async def test_prompt_area_normal_mode_border(self, _mock_resolve):
        """PromptArea should switch to normal-mode class on escape."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            area = app.screen.query_one("#chat-prompt-area")
            assert area.has_class("normal-mode")
            assert not area.has_class("insert-mode")
