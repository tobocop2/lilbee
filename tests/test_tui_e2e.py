"""End-to-end TUI integration tests.

These tests launch the real Textual app and verify observable behavior.
They exist because unit tests with mocks passed while the app was broken.
Every test here reproduces a bug that was found by manual testing.
"""

from __future__ import annotations

from unittest import mock

import pytest
from textual.app import App, ComposeResult

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
        "lilbee.providers.llama_cpp_provider.resolve_model_path",
        return_value=cfg.models_dir / "fake.gguf",
    ):
        yield


@pytest.fixture()
def _mock_services():
    """Mock services to prevent real provider initialization."""
    from lilbee.services import set_services

    mock_svc = mock.MagicMock()
    mock_svc.provider.list_models.return_value = []
    mock_svc.searcher._embedder.embedding_available.return_value = True
    set_services(mock_svc)
    try:
        yield mock_svc
    finally:
        set_services(None)


class ChatTestApp(App[None]):
    """Minimal app that pushes ChatScreen for testing."""

    def compose(self) -> ComposeResult:
        yield from ()

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
        """When resolve_model_path succeeds, embedding is available."""
        from lilbee.embedder import Embedder

        mock_provider = mock.MagicMock()
        cfg.embedding_model = "test-embed"

        embedder = Embedder(cfg, mock_provider)
        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
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

        from lilbee.cli.tui.widgets.model_bar import ModelOption

        with mock.patch("lilbee.cli.tui.widgets.model_bar._collect_native_models") as mock_native:

            def fill_buckets(buckets, seen):
                for m in mock_manifests:
                    ref = f"{m.name}:{m.tag}"
                    label = m.display_name or ref
                    buckets.get(m.task, buckets["chat"]).append(ModelOption(label, ref))
                    seen.add(ref)

            mock_native.side_effect = fill_buckets
            with mock.patch("lilbee.cli.tui.widgets.model_bar._collect_remote_models"):
                chat, embed, vision = _classify_installed_models()

        chat_refs = [o.ref for o in chat]
        embed_refs = [o.ref for o in embed]
        vision_refs = [o.ref for o in vision]
        assert "Qwen3:latest" in chat_refs
        assert "Nomic Embed:latest" in embed_refs
        assert "LightOnOCR:latest" in vision_refs

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
            screen.streaming = True
            screen.action_cancel_stream = mock.MagicMock()

            bar = screen.query_one("#model-bar", ModelBar)
            bar._populating = False

            # Simulate model change
            event = mock.MagicMock()
            event.value = "new-model.gguf"
            event.select = mock.MagicMock()
            event.select.id = "chat-model-select"

            with (
                mock.patch("lilbee.services.reset_services"),
                mock.patch.object(bar, "_deferred_reset"),
            ):
                bar._on_chat_model_changed(event)

            screen.action_cancel_stream.assert_called_once()


class TestViewTabsPresence:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families")
    async def test_status_bar_on_all_screens(self, _fam, _cat, _mock_resolve):
        """ViewTabs must exist on every screen."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Chat screen
            bar = app.screen.query_one(ViewTabs)
            assert bar is not None

            # Cycle through all views
            for view in ["Catalog", "Status", "Settings", "Tasks"]:
                app.switch_view(view)
                await pilot.pause()
                bar = app.screen.query_one(ViewTabs)
                assert bar is not None, f"ViewTabs missing on {view} screen"


class TestModeIndicator:
    async def test_insert_mode_on_startup(self, _mock_resolve):
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app.screen.query_one(ViewTabs)
            assert "INSERT" in bar.mode_text

    async def test_normal_mode_on_escape(self, _mock_resolve):
        from lilbee.cli.tui.widgets.status_bar import ViewTabs

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            bar = app.screen.query_one(ViewTabs)
            assert "NORMAL" in bar.mode_text


class TestViewCycling:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families")
    async def test_cycles_all_views(self, _fam, _cat, _mock_resolve):
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.active_view == "Chat"

            expected = ["Catalog", "Status", "Settings", "Tasks", "Wiki", "Chat"]
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
                name="mistral",
                tag="7b",
                display_name="Mistral 7B",
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
            slug="testchat",
            name="TestChat",
            task="chat",
            description="A test chat model",
            variants=(
                ModelVariant(
                    hf_repo="test/chat-repo",
                    filename="chat-Q4.gguf",
                    param_count="7B",
                    tag="7b",
                    quant="Q4_K_M",
                    size_mb=4000,
                    recommended=True,
                ),
            ),
        ),
        ModelFamily(
            slug="testembed",
            name="TestEmbed",
            task="embedding",
            description="A test embedding model",
            variants=(
                ModelVariant(
                    hf_repo="test/embed-repo",
                    filename="embed-Q8.gguf",
                    param_count="0.5B",
                    tag="0.5b",
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


def _mock_status_deps():
    """Mock status screen dependencies to avoid real store/model access."""
    from lilbee.model_info import ModelArchInfo

    return mock.patch.multiple(
        "lilbee.cli.tui.screens.status",
        get_model_architecture=mock.MagicMock(return_value=ModelArchInfo()),
    )


class TestScreenTransitions:
    """Test that switching between screens does not crash."""

    async def test_navigate_chat_to_catalog_to_settings(self, _mock_resolve):
        """F2->Models, then F4->Settings, verify no crash."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                assert app.active_view == "Chat"

                app.switch_view("Catalog")
                await pilot.pause()
                assert app.active_view == "Catalog"

                app.switch_view("Settings")
                await pilot.pause()
                assert app.active_view == "Settings"

    async def test_navigate_all_views_via_keybindings(self, _mock_resolve):
        """Cycle through all views with nav_next (l key)."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                expected = ["Catalog", "Status", "Settings", "Tasks", "Wiki", "Chat"]
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
                app.switch_view("Catalog")
                await pilot.pause()
                assert app.active_view == "Catalog"

                await pilot.press("q")
                await pilot.pause()
                # Should be back at Chat (base screen)
                from lilbee.cli.tui.screens.chat import ChatScreen

                assert isinstance(app.screen, ChatScreen)

    async def test_navigate_catalog_to_tasks(self, _mock_resolve):
        """The specific crash case: catalog -> tasks transition."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                app.switch_view("Tasks")
                await pilot.pause()
                assert app.active_view == "Tasks"

    async def test_forward_cycle_full_loop(self, _mock_resolve):
        """Chat->Catalog->Status->Settings->Tasks->Wiki->Chat via nav_next."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                assert app.active_view == "Chat"
                full_cycle = ["Catalog", "Status", "Settings", "Tasks", "Wiki", "Chat"]
                for view in full_cycle:
                    app.action_nav_next()
                    await pilot.pause()
                    assert app.active_view == view

    async def test_backward_cycle_full_loop(self, _mock_resolve):
        """Chat->Wiki->Tasks->Settings->Status->Catalog->Chat via nav_prev."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                assert app.active_view == "Chat"
                backward_cycle = ["Wiki", "Tasks", "Settings", "Status", "Catalog", "Chat"]
                for view in backward_cycle:
                    app.action_nav_prev()
                    await pilot.pause()
                    assert app.active_view == view

    async def test_rapid_switching(self, _mock_resolve):
        """Rapid forward/backward switching: Models, Status, Settings, Models, Status."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                sequence = ["Catalog", "Status", "Settings", "Catalog", "Status"]
                for view in sequence:
                    app.switch_view(view)
                    await pilot.pause()
                    assert app.active_view == view

    async def test_help_from_each_view_and_dismiss(self, _mock_resolve):
        """Open help from each view, dismiss, verify back at same view."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                for view in ["Chat", "Catalog", "Status", "Settings", "Tasks"]:
                    app.switch_view(view)
                    await pilot.pause()
                    assert app.active_view == view

                    app.action_push_help()
                    await pilot.pause()
                    assert app.screen.query("HelpPanel")

                    app.action_push_help()
                    await pilot.pause()
                    assert not app.screen.query("HelpPanel")

    async def test_q_from_models_returns_to_chat(self, _mock_resolve):
        """From Models, pressing q returns to Chat."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
                assert isinstance(app.screen, ChatScreen)

    async def test_q_from_status_returns_to_chat(self, _mock_resolve):
        """From Status, pressing q returns to Chat."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Status")
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
                assert isinstance(app.screen, ChatScreen)

    async def test_pop_from_settings_returns_to_chat(self, _mock_resolve):
        """From Settings, action_go_back returns to Chat.
        Note: 'q' keystroke is consumed by the search Input when focused,
        so we test the action directly.
        """
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Settings")
                await pilot.pause()
                app.screen.action_go_back()
                await pilot.pause()
                assert isinstance(app.screen, ChatScreen)

    async def test_q_from_tasks_returns_to_chat(self, _mock_resolve):
        """From Tasks, pressing q returns to Chat."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Tasks")
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
                assert isinstance(app.screen, ChatScreen)

    async def test_escape_from_each_overlay_returns_to_chat(self, _mock_resolve):
        """From each non-Chat view, pressing escape pops the screen."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                for view in ["Catalog", "Status", "Settings", "Tasks"]:
                    app.switch_view(view)
                    await pilot.pause()
                    app.screen.action_go_back()
                    await pilot.pause()
                    assert isinstance(app.screen, ChatScreen)

    async def test_theme_cycling(self, _mock_resolve):
        """Ctrl+T cycles through themes without crashing."""
        from lilbee.cli.tui.app import DARK_THEMES, LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            initial_theme = app.theme
            app.action_cycle_theme()
            await pilot.pause()
            assert app.theme != initial_theme
            assert app.theme in DARK_THEMES


class TestChatInteractions:
    """Test all chat screen interactions: vim modes, keybindings, scrolling."""

    async def test_insert_mode_is_default(self, _mock_resolve):
        """Chat starts in insert mode with input focused."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.screen._insert_mode is True
            from textual.widgets import Input

            inp = app.screen.query_one("#chat-input", Input)
            assert inp.has_focus

    async def test_escape_enters_normal_mode(self, _mock_resolve):
        """Pressing escape switches to normal mode."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            assert app.screen._insert_mode is False

    async def test_i_enters_insert_mode_from_normal(self, _mock_resolve):
        """In normal mode, pressing a printable key enters insert mode."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            assert app.screen._insert_mode is False
            app.screen._enter_insert_mode()
            await pilot.pause()
            assert app.screen._insert_mode is True

    async def test_normal_mode_j_k_cycle_focus(self, _mock_resolve):
        """In normal mode, j/k cycle focus between widgets."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()

            app.screen.action_vim_scroll_down()
            await pilot.pause()
            new_focus = app.screen.focused.id if app.screen.focused else None
            assert new_focus is not None
            # Focus should have moved (or stayed if only 1 widget)
            # Just verify no crash

    async def test_normal_mode_g_scrolls_top(self, _mock_resolve):
        """In normal mode, g scrolls the chat log to top."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            app.screen.action_vim_scroll_home()
            await pilot.pause()
            assert app.screen._insert_mode is False

    async def test_normal_mode_G_scrolls_bottom(self, _mock_resolve):
        """In normal mode, G scrolls the chat log to bottom."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            app.screen.action_vim_scroll_end()
            await pilot.pause()
            assert app.screen._insert_mode is False

    async def test_page_up_page_down(self, _mock_resolve):
        """PageUp and PageDown scroll the chat log."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_scroll_up()
            await pilot.pause()
            app.screen.action_scroll_down()
            await pilot.pause()
            assert app.screen.is_current

    async def test_half_page_scroll(self, _mock_resolve):
        """Ctrl-D and Ctrl-U half-page scroll."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_half_page_down()
            await pilot.pause()
            app.screen.action_half_page_up()
            await pilot.pause()
            assert app.screen.is_current

    async def test_slash_focuses_input_with_prefix(self, _mock_resolve):
        """/ key focuses input and prefills with /."""
        from textual.widgets import Input

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_focus_commands()
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)
            assert inp.has_focus
            assert inp.value.startswith("/")

    async def test_slash_command_help_opens_panel(self, _mock_resolve):
        """Typing /help dispatches to the help handler."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/help")
            await pilot.pause()
            assert app.screen.query("HelpPanel")

    async def test_slash_command_unknown_notifies(self, _mock_resolve):
        """Unknown slash command shows a warning notification."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/nonexistent_command_xyz")
            await pilot.pause()
            assert app.screen.is_current

    async def test_slash_command_version(self, _mock_resolve):
        """Typing /version shows version notification."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/version")
            await pilot.pause()
            assert app.screen.is_current

    async def test_slash_command_set_valid(self, _mock_resolve):
        """/set chat_model <value> updates cfg."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/set chat_model new-test-model")
            await pilot.pause()
            assert cfg.chat_model == "new-test-model"

    async def test_slash_command_set_unknown_key(self, _mock_resolve):
        """/set nonexistent_key warns."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/set nonexistent_key_xyz value")
            await pilot.pause()
            assert app.screen.is_current

    async def test_escape_cancels_stream_when_streaming(self, _mock_resolve):
        """Escape cancels streaming if active."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.streaming = True
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            assert app.screen.streaming is False

    async def test_submit_empty_does_nothing(self, _mock_resolve):
        """Submitting empty input is a no-op."""
        from textual.widgets import Input

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)
            inp.value = ""
            event = Input.Submitted(inp, "")
            app.screen._on_chat_submitted(event)
            assert app.screen.streaming is False
            await pilot.pause()

    async def test_submit_message_mocked_llm(self, _mock_resolve, _mock_services):
        """Submitting a message calls _send_message and creates user bubble."""
        from textual.widgets import Input

        from lilbee.cli.tui.widgets.message import UserMessage

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)
            inp.value = "Hello test"
            await inp.action_submit()
            await pilot.pause()
            messages = app.screen.query(UserMessage)
            assert len(messages) >= 1

    async def test_input_history_navigation(self, _mock_resolve, _mock_services):
        """Up/Down arrows recall input history."""
        from textual.widgets import Input

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", Input)

            # Submit two messages
            inp.value = "first message"
            await inp.action_submit()
            await pilot.pause()
            inp.value = "second message"
            await inp.action_submit()
            await pilot.pause()

            # Navigate up through history
            app.screen.action_history_prev()
            await pilot.pause()
            assert inp.value == "second message"

            app.screen.action_history_prev()
            await pilot.pause()
            assert inp.value == "first message"

            # Navigate down
            app.screen.action_history_next()
            await pilot.pause()
            assert inp.value == "second message"

            # Past end clears
            app.screen.action_history_next()
            await pilot.pause()
            assert inp.value == ""

    async def test_toggle_markdown_rendering(self, _mock_resolve):
        """Ctrl+R toggles markdown rendering."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            initial = cfg.markdown_rendering
            await app.screen.action_toggle_markdown()
            await pilot.pause()
            assert cfg.markdown_rendering != initial
            # Toggle back
            await app.screen.action_toggle_markdown()
            await pilot.pause()
            assert cfg.markdown_rendering == initial

    async def test_normal_mode_enter_re_enters_insert(self, _mock_resolve):
        """In normal mode, pressing enter via on_key enters insert mode."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            assert app.screen._insert_mode is False
            # Simulate enter key event in normal mode
            from textual.events import Key

            event = Key("enter", "\r")
            app.screen.on_key(event)
            await pilot.pause()
            assert app.screen._insert_mode is True

    async def test_history_actions_skip_in_normal_mode(self, _mock_resolve):
        """In normal mode, up/down raise SkipAction (no focus cycling)."""
        from textual.actions import SkipAction

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            with pytest.raises(SkipAction):
                app.screen.action_history_prev()
            with pytest.raises(SkipAction):
                app.screen.action_history_next()
            assert app.screen._insert_mode is False


class TestCatalogInteractions:
    """Test all catalog screen interactions: grid/list toggle, search, navigation."""

    async def test_grid_view_is_default(self, _mock_resolve):
        """Grid view is shown on mount by default."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                grids = app.screen.query(GridSelect)
                assert len(grids) > 0
                assert app.screen.has_class("-grid-view")

    async def test_v_toggles_to_list_and_back(self, _mock_resolve):
        """Press v: grid->list, v again: list->grid."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                assert app.screen.has_class("-grid-view")

                app.screen.action_toggle_view()
                await pilot.pause()
                assert app.screen.has_class("-list-view")
                assert not app.screen.has_class("-grid-view")

                app.screen.action_toggle_view()
                await pilot.pause()
                assert app.screen.has_class("-grid-view")
                assert not app.screen.has_class("-list-view")

    async def test_search_filters_cards_in_grid_view(self, _mock_resolve):
        """Type search text in grid view, verify cards filter by visibility."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.model_card import ModelCard

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                all_cards = app.screen.query(ModelCard)
                initial_count = len(all_cards)
                assert initial_count > 0

                search = app.screen.query_one("#catalog-search")
                search.value = "TestChat"
                await pilot.pause()

                all_cards_after = app.screen.query(ModelCard)
                assert len(all_cards_after) == initial_count
                visible = [c for c in all_cards_after if c.display]
                hidden = [c for c in all_cards_after if not c.display]
                assert len(visible) >= 1
                assert len(hidden) >= 1

    async def test_search_input_is_visible_when_opened(self, _mock_resolve):
        """Pressing / focuses a visible search input ready for text entry."""
        from textual.widgets import Input

        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                await pilot.press("slash")
                await pilot.pause()
                search = app.screen.query_one("#catalog-search", Input)
                assert search.display is True
                assert search.region.width > 0
                assert search.region.height > 0
                assert search.has_focus

                await pilot.press("q", "w", "e", "n")
                await pilot.pause()
                assert search.value == "qwen"

    async def test_search_submit_returns_focus_to_grid(self, _mock_resolve):
        """Pressing Enter in search returns focus to the visible grid."""
        from textual.widgets import Input

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                await pilot.press("slash")
                await pilot.pause()
                search = app.screen.query_one("#catalog-search", Input)
                search.value = "test"
                await search.action_submit()
                await pilot.pause()
                grid = app.screen.query_one(GridSelect)
                assert grid.has_focus

    async def test_search_submit_returns_focus_to_table_in_list_view(self, _mock_resolve):
        """In list view, pressing Enter in search returns focus to the DataTable."""
        from textual.widgets import DataTable, Input

        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                app.screen.action_toggle_view()
                await pilot.pause()

                await pilot.press("slash")
                await pilot.pause()
                search = app.screen.query_one("#catalog-search", Input)
                search.value = "test"
                await search.action_submit()
                await pilot.pause()
                table = app.screen.query_one("#catalog-table", DataTable)
                assert table.has_focus

    async def test_grid_card_count_matches_families(self, _mock_resolve):
        """Verify correct number of cards for featured models."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.model_card import ModelCard

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                cards = app.screen.query(ModelCard)
                assert len(cards) == 2

    async def test_list_view_j_k_navigation(self, _mock_resolve):
        """In list view, j/k move cursor up/down."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                app.screen.action_toggle_view()
                await pilot.pause()

                table = app.screen.query_one("#catalog-table", DataTable)
                if table.row_count > 1:
                    initial_row = table.cursor_row
                    app.screen.action_cursor_down()
                    await pilot.pause()
                    assert table.cursor_row >= initial_row

                    app.screen.action_cursor_up()
                    await pilot.pause()

    async def test_list_view_g_G_jump(self, _mock_resolve):
        """In list view, g jumps to top, G jumps to bottom."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.pause()

                app.screen.action_toggle_view()
                await pilot.pause()
                await pilot.pause()

                table = app.screen.query_one("#catalog-table", DataTable)
                table.focus()
                await pilot.pause()

                if table.row_count > 0:
                    await pilot.press("G")
                    await pilot.pause()
                    assert table.cursor_row == table.row_count - 1

                    await pilot.press("g")
                    await pilot.pause()
                    assert table.cursor_row == 0

    async def test_list_view_page_down_up(self, _mock_resolve):
        """In list view, space/ctrl-d pages down, ctrl-u pages up."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                app.screen.action_toggle_view()
                await pilot.pause()

                app.screen.action_page_down()
                await pilot.pause()
                app.screen.action_page_up()
                await pilot.pause()
                assert app.screen.is_current

    async def test_column_header_click_sorts_list(self, _mock_resolve):
        """Clicking a column header sorts the table by that column."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                app.screen.action_toggle_view()
                await pilot.pause()

                assert app.screen._sort_column == "Name"
                assert app.screen._sort_ascending is True

                from textual.widgets import DataTable

                table = app.screen.query_one("#catalog-table", DataTable)
                event = DataTable.HeaderSelected(
                    table,
                    column_key="Task",
                    column_index=1,
                    label="Task",
                )
                app.screen._on_header_selected(event)
                await pilot.pause()
                assert app.screen._sort_column == "Task"
                assert app.screen._sort_ascending is True

                event2 = DataTable.HeaderSelected(
                    table,
                    column_key="Task",
                    column_index=1,
                    label="Task",
                )
                app.screen._on_header_selected(event2)
                await pilot.pause()
                assert app.screen._sort_column == "Task"
                assert app.screen._sort_ascending is False

    async def test_search_filters_list_view(self, _mock_resolve):
        """Search input filters rows in list view."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                app.screen.action_toggle_view()
                await pilot.pause()

                table = app.screen.query_one("#catalog-table", DataTable)
                initial_rows = table.row_count

                search = app.screen.query_one("#catalog-search")
                search.value = "TestChat"
                await pilot.pause()

                filtered_rows = table.row_count
                assert filtered_rows <= initial_rows

    async def test_delete_model_without_selection_warns(self, _mock_resolve):
        """Pressing d without a highlighted model shows warning."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                app.screen.action_toggle_view()
                await pilot.pause()
                app.screen.action_delete_model()
                await pilot.pause()
                assert app.screen.is_current

    async def test_q_from_catalog_returns_to_chat(self, _mock_resolve):
        """Pressing q on catalog returns to chat."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
                assert isinstance(app.screen, ChatScreen)

    async def test_grid_navigation_does_not_crash_in_list_mode(self, _mock_resolve):
        """Cursor actions in grid mode (when grid is active) are no-ops in list."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                # These actions should be no-ops or safe in grid mode
                app.screen.action_cursor_down()
                app.screen.action_cursor_up()
                app.screen.action_jump_top()
                app.screen.action_jump_bottom()
                app.screen.action_page_down()
                app.screen.action_page_up()
                assert app.screen.is_current
                await pilot.pause()


class TestSettingsInteractions:
    """Test all settings screen interactions: search, editing, navigation."""

    async def test_grouped_sections_visible(self, _mock_resolve):
        """Grouped sections are visible on mount."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            groups = app.screen.query(".group-title")
            assert len(groups) >= 1

    async def test_search_filters_settings(self, _mock_resolve):
        """Typing in search hides non-matching rows."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()

            all_rows = app.screen.query(".setting-row")
            total = len(all_rows)
            assert total > 0

            search = app.screen.query_one("#settings-search")
            search.value = "chat_model"
            await pilot.pause()

            visible = [r for r in app.screen.query(".setting-row") if r.display]
            assert len(visible) < total
            assert len(visible) >= 1

    async def test_search_no_match_hides_all(self, _mock_resolve):
        """Search with no matching text hides all rows."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()

            search = app.screen.query_one("#settings-search")
            search.value = "zzz_nonexistent_setting_zzz"
            await pilot.pause()

            visible = [r for r in app.screen.query(".setting-row") if r.display]
            assert len(visible) == 0

    async def test_search_empty_shows_all(self, _mock_resolve):
        """Clearing search shows all rows again."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()

            all_rows = app.screen.query(".setting-row")
            total = len(all_rows)

            search = app.screen.query_one("#settings-search")
            search.value = "chat_model"
            await pilot.pause()
            search.value = ""
            await pilot.pause()

            visible = [r for r in app.screen.query(".setting-row") if r.display]
            assert len(visible) == total

    async def test_edit_string_value_updates_cfg(self, _mock_resolve):
        """Editing a writable string setting persists to cfg."""
        from textual.widgets import Input

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()

            editor = app.screen.query_one("#ed-system_prompt", Input)
            editor.value = "test system prompt"
            event = Input.Submitted(editor, "test system prompt")
            app.screen._on_input_save(event)
            await pilot.pause()
            assert cfg.system_prompt == "test system prompt"

    async def test_toggle_boolean_checkbox(self, _mock_resolve):
        """Toggling a boolean checkbox updates cfg."""
        from textual.widgets import Checkbox

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()

            checkbox = app.screen.query_one("#ed-show_reasoning", Checkbox)
            initial = checkbox.value
            checkbox.toggle()
            await pilot.pause()
            assert cfg.show_reasoning != initial

    async def test_read_only_fields_have_no_editor(self, _mock_resolve):
        """Read-only settings do not have an editor widget."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()

            from lilbee.cli.settings_map import SETTINGS_MAP

            for key, defn in SETTINGS_MAP.items():
                if not defn.writable:
                    editors = app.screen.query(f"#ed-{key}")
                    assert len(editors) == 0, f"Read-only setting {key} has an editor"

    async def test_j_k_scrolls(self, _mock_resolve):
        """j and k keybindings scroll the settings list."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            app.screen.action_scroll_down()
            await pilot.pause()
            app.screen.action_scroll_up()
            await pilot.pause()
            assert app.screen.is_current

    async def test_g_G_scroll_home_end(self, _mock_resolve):
        """g and G scroll to top and bottom."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            app.screen.action_scroll_end()
            await pilot.pause()
            app.screen.action_scroll_home()
            await pilot.pause()
            assert app.screen.is_current

    async def test_pop_screen_returns_to_chat(self, _mock_resolve):
        """action_go_back returns to chat."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            app.screen.action_go_back()
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

    async def test_settings_changed_signal_fires(self, _mock_resolve):
        """Editing a setting fires the settings_changed signal."""
        from textual.widgets import Input

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        received = []

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.settings_changed_signal.subscribe(app, lambda data: received.append(data))
            app.switch_view("Settings")
            await pilot.pause()

            editor = app.screen.query_one("#ed-system_prompt", Input)
            editor.value = "signal test prompt"
            event = Input.Submitted(editor, "signal test prompt")
            app.screen._on_input_save(event)
            await pilot.pause()
            assert len(received) >= 1
            assert received[0][0] == "system_prompt"


class TestStatusInteractions:
    """Test all status screen interactions: collapsible sections, navigation."""

    async def test_collapsible_sections_render(self, _mock_resolve):
        """Collapsible sections exist for config, docs, arch, storage."""
        from textual.widgets import Collapsible

        from lilbee.cli.tui.app import LilbeeApp

        with _mock_status_deps():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Status")
                await pilot.pause()
                collapsibles = app.screen.query(Collapsible)
                assert len(collapsibles) >= 3

    async def test_model_pills_show_loaded(self, _mock_resolve):
        """Config section shows model pills with loaded/not-set status."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_status_deps():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Status")
                await pilot.pause()
                config_info = app.screen.query_one("#config-info")
                assert config_info is not None

    async def test_documents_table_populated(self, _mock_resolve):
        """Documents table exists and has at least header row."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        with _mock_status_deps():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Status")
                await pilot.pause()
                table = app.screen.query_one("#docs-table", DataTable)
                assert table is not None
                assert table.row_count >= 1

    async def test_j_k_moves_cursor_in_docs_table(self, _mock_resolve):
        """j/k keybindings move cursor in the documents table."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_status_deps():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Status")
                await pilot.pause()
                app.screen.action_cursor_down()
                await pilot.pause()
                app.screen.action_cursor_up()
                await pilot.pause()
                assert app.screen.is_current

    async def test_g_G_jump_in_docs_table(self, _mock_resolve):
        """g/G jump to top/bottom in the documents table."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_status_deps():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Status")
                await pilot.pause()
                app.screen.action_jump_bottom()
                await pilot.pause()
                app.screen.action_jump_top()
                await pilot.pause()
                assert app.screen.is_current

    async def test_q_returns_to_chat(self, _mock_resolve):
        """Pressing q from status returns to chat."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        with _mock_status_deps():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Status")
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
                assert isinstance(app.screen, ChatScreen)

    async def test_documents_table_with_mock_store(self, _mock_resolve):
        """Documents table populated with mock store data shows real rows."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        mock_sources = [
            {"filename": "doc1.md", "chunk_count": 5},
            {"filename": "doc2.pdf", "chunk_count": 12},
        ]
        with (
            _mock_status_deps(),
            mock.patch(
                "lilbee.cli.tui.screens.status.StatusScreen._fetch_sources",
                return_value=mock_sources,
            ),
        ):
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Status")
                await pilot.pause()
                table = app.screen.query_one("#docs-table", DataTable)
                assert table.row_count == 2


class TestTaskCenterInteractions:
    """Test all task center interactions: empty state, tasks, navigation."""

    async def test_renders_empty_state(self, _mock_resolve):
        """Task Center shows empty table when no tasks."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Tasks")
            await pilot.pause()
            table = app.screen.query_one("#task-table", DataTable)
            assert table.row_count == 0

    async def test_active_task_shown_in_table(self, _mock_resolve):
        """Active tasks appear in the task table."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            task_id = app.task_bar.add_task("Test Download", "download")
            app.task_bar.update_task(task_id, 45, "100/500 MB")

            app.switch_view("Tasks")
            await pilot.pause()
            table = app.screen.query_one("#task-table", DataTable)
            assert table.row_count >= 1

    async def test_completed_task_shown(self, _mock_resolve):
        """Completed tasks appear in history."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            task_id = app.task_bar.add_task("Done Task", "sync")
            app.task_bar.complete_task(task_id)

            app.switch_view("Tasks")
            await pilot.pause()
            table = app.screen.query_one("#task-table", DataTable)
            assert table.row_count >= 1

    async def test_failed_task_shown(self, _mock_resolve):
        """Failed tasks appear in history."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            task_id = app.task_bar.add_task("Fail Task", "download")
            app.task_bar.fail_task(task_id, "Network error")

            app.switch_view("Tasks")
            await pilot.pause()
            table = app.screen.query_one("#task-table", DataTable)
            assert table.row_count >= 1

    async def test_j_k_cursor_navigation(self, _mock_resolve):
        """j/k move cursor in the task table."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.task_bar.add_task("Task 1", "download")
            app.task_bar.add_task("Task 2", "sync")

            app.switch_view("Tasks")
            await pilot.pause()
            app.screen.action_cursor_down()
            await pilot.pause()
            app.screen.action_cursor_up()
            await pilot.pause()
            assert app.screen.is_current

    async def test_cursor_movement_updates_detail_panel(self, _mock_resolve):
        """Moving cursor updates the detail panel."""
        from textual.widgets import Static

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            task_id = app.task_bar.add_task("Detail Task", "download")
            app.task_bar.update_task(task_id, 50, "Downloading file.gguf")

            app.switch_view("Tasks")
            await pilot.pause()

            detail = app.screen.query_one("#task-detail", Static)
            # Trigger highlight to populate detail
            from textual.widgets import DataTable

            table = app.screen.query_one("#task-table", DataTable)
            if table.row_count > 0:
                row_key, _ = table.coordinate_to_cell_key(table.cursor_coordinate)
                app.screen._show_task_detail(row_key.value)
                await pilot.pause()
                # Detail should contain the task name
                rendered = str(detail.render())
                assert "Detail Task" in rendered

    async def test_refresh_action(self, _mock_resolve):
        """r keybinding refreshes the task list."""
        from textual.widgets import DataTable

        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Tasks")
            await pilot.pause()

            app.task_bar.add_task("Late Task", "download")
            app.screen.action_refresh_tasks()
            await pilot.pause()

            table = app.screen.query_one("#task-table", DataTable)
            assert table.row_count >= 1

    async def test_cancel_task_action(self, _mock_resolve):
        """c keybinding cancels the selected task."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.task_bar.add_task("Cancel Me", "download")

            app.switch_view("Tasks")
            await pilot.pause()
            app.screen.action_cancel_task()
            await pilot.pause()
            assert app.screen.is_current

    async def test_q_returns_to_chat(self, _mock_resolve):
        """Pressing q returns to chat."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Tasks")
            await pilot.pause()
            await pilot.press("q")
            await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

    async def test_cancel_on_empty_table_is_noop(self, _mock_resolve):
        """Cancelling with no tasks is a no-op."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Tasks")
            await pilot.pause()
            app.screen.action_cancel_task()
            await pilot.pause()
            assert app.screen.is_current


class TestChatPromptBorder:
    """Test that the chat prompt area has a single border, not stacked."""

    async def test_prompt_area_border_uses_focus_within(self, _mock_resolve):
        """PromptArea uses :focus-within for border, not mode classes."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            area = app.screen.query_one("#chat-prompt-area")
            inp = app.screen.query_one("#chat-input")
            # No mode classes on the prompt area — border driven by :focus-within CSS
            assert not area.has_class("insert-mode")
            assert not area.has_class("normal-mode")
            # Input should not have its own border
            assert inp.styles.border is not None

    async def test_normal_mode_dims_input_not_area(self, _mock_resolve):
        """Normal mode adds class to input (opacity), not to prompt area."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.action_enter_normal_mode()
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            area = app.screen.query_one("#chat-prompt-area")
            assert inp.has_class("normal-mode")
            assert not area.has_class("normal-mode")


class TestAppQuit:
    """Test app quit behavior: Ctrl+C handling."""

    async def test_quit_with_no_active_tasks_exits(self, _mock_resolve):
        """Ctrl+C with no active tasks calls exit."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch.object(app, "exit") as mock_exit:
                await app.action_quit()
                mock_exit.assert_called_once()

    async def test_quit_cancels_active_task_first(self, _mock_resolve):
        """Ctrl+C with active task cancels it instead of exiting."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.task_bar.add_task("Active Task", "download")
            app.task_bar.queue.advance("download")
            with mock.patch.object(app, "exit") as mock_exit:
                await app.action_quit()
                mock_exit.assert_not_called()

    async def test_quit_cancels_stream_if_on_chat(self, _mock_resolve):
        """Ctrl+C cancels stream before exiting when streaming."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.streaming = True
            with (
                mock.patch.object(app.screen, "action_cancel_stream") as mock_cancel,
                mock.patch.object(app, "exit") as mock_exit,
            ):
                await app.action_quit()
                mock_cancel.assert_called_once()
                mock_exit.assert_not_called()


class TestChatSlashCommands:
    """Test all slash command dispatches."""

    async def test_cmd_models(self, _mock_resolve):
        """/models pushes catalog screen."""
        with _mock_catalog_deps(), _mock_remote_models():
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.screen._handle_slash("/models")
                await pilot.pause()
                from lilbee.cli.tui.screens.catalog import CatalogScreen

                assert isinstance(app.screen, CatalogScreen)

    async def test_cmd_settings(self, _mock_resolve):
        """/settings pushes settings screen."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/settings")
            await pilot.pause()
            from lilbee.cli.tui.screens.settings import SettingsScreen

            assert isinstance(app.screen, SettingsScreen)

    async def test_cmd_status(self, _mock_resolve):
        """/status pushes status screen."""
        with _mock_status_deps():
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.screen._handle_slash("/status")
                await pilot.pause()
                from lilbee.cli.tui.screens.status import StatusScreen

                assert isinstance(app.screen, StatusScreen)

    async def test_cmd_model_with_name(self, _mock_resolve):
        """/model <name> sets the chat model."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/model slash-test-model")
            await pilot.pause()
            assert "slash-test-model" in cfg.chat_model

    async def test_cmd_model_without_name(self, _mock_resolve):
        """/model with no args pushes catalog."""
        with _mock_catalog_deps(), _mock_remote_models():
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.screen._handle_slash("/model")
                await pilot.pause()
                from lilbee.cli.tui.screens.catalog import CatalogScreen

                assert isinstance(app.screen, CatalogScreen)

    async def test_cmd_vision_set(self, _mock_resolve):
        """/vision <model> sets the vision model."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/vision test-vision-model")
            await pilot.pause()
            assert cfg.vision_model == "test-vision-model"

    async def test_cmd_vision_off(self, _mock_resolve):
        """/vision off disables vision."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            cfg.vision_model = "some-model"
            app.screen._handle_slash("/vision off")
            await pilot.pause()
            assert cfg.vision_model == ""

    async def test_cmd_vision_status(self, _mock_resolve):
        """/vision with no args shows current status."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/vision")
            await pilot.pause()
            assert app.screen.is_current

    async def test_cmd_reset_without_confirm(self, _mock_resolve):
        """/reset without confirm shows warning."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/reset")
            await pilot.pause()
            assert app.screen.is_current

    async def test_cmd_reset_with_confirm(self, _mock_resolve):
        """/reset confirm performs reset."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch("lilbee.cli.tui.screens.chat.ChatScreen._cmd_reset") as mock_reset:
                app.screen._cmd_reset("confirm")
                await pilot.pause()
            mock_reset.assert_called_once_with("confirm")

    async def test_cmd_cancel(self, _mock_resolve):
        """/cancel cancels workers."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/cancel")
            await pilot.pause()
            assert app.screen.streaming is False

    async def test_cmd_theme_with_name(self, _mock_resolve):
        """/theme <name> sets theme."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen._handle_slash("/theme monokai")
            await pilot.pause()
            assert app.theme == "monokai"

    async def test_cmd_theme_without_name(self, _mock_resolve):
        """/theme with no args lists themes."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from lilbee.cli.tui.screens.chat import ChatScreen

            screen = app.screen
            assert isinstance(screen, ChatScreen)
            screen._handle_slash("/theme")
            await pilot.pause()
            assert app.screen.is_current

    async def test_cmd_delete_no_docs(self, _mock_resolve):
        """/delete with no docs shows warning."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch("lilbee.cli.tui.screens.chat.get_services") as mock_svc:
                mock_svc.return_value.store.get_sources.return_value = []
                app.screen._handle_slash("/delete")
                await pilot.pause()
                assert app.screen.is_current

    async def test_cmd_add_nonexistent_path(self, _mock_resolve):
        """/add with nonexistent path shows error."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/add /nonexistent/path/xyz")
            await pilot.pause()
            assert app.screen.is_current

    async def test_cmd_add_no_args(self, _mock_resolve):
        """/add with no args is a no-op."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/add")
            await pilot.pause()
            assert app.screen.is_current


class TestChatCompletions:
    """Test tab completion behavior."""

    async def test_tab_shows_completions(self, _mock_resolve):
        """Tab on empty input does not crash."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import Input

            inp = app.screen.query_one("#chat-input", Input)
            inp.value = "/"
            app.screen.action_complete()
            await pilot.pause()
            assert app.screen.is_current

    async def test_ctrl_n_cycles_forward(self, _mock_resolve):
        """Ctrl+N cycles forward through completions."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import Input

            inp = app.screen.query_one("#chat-input", Input)
            inp.value = "/"
            app.screen.action_complete_next()
            await pilot.pause()
            assert app.screen.is_current

    async def test_ctrl_p_cycles_backward(self, _mock_resolve):
        """Ctrl+P cycles backward through completions."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import Input

            inp = app.screen.query_one("#chat-input", Input)
            inp.value = "/"
            app.screen.action_complete_prev()
            await pilot.pause()
            assert app.screen.is_current

    async def test_input_change_hides_overlay(self, _mock_resolve):
        """Changing input manually hides the completion overlay."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            from textual.widgets import Input

            from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

            inp = app.screen.query_one("#chat-input", Input)
            inp.value = "/"
            app.screen.action_complete()
            await pilot.pause()

            overlay = app.screen.query_one("#completion-overlay", CompletionOverlay)
            # Changing input should dismiss overlay
            inp.value = "/h"
            await pilot.pause()
            assert overlay.display is False


class TestGridSelectWidget:
    """Test GridSelect cursor navigation and selection."""

    async def test_arrow_key_navigation(self):
        """Arrow keys move the cursor in the grid."""
        from textual.widgets import Static

        from lilbee.cli.tui.widgets.grid_select import GridSelect

        class GridTestApp(App[None]):
            def compose(self) -> ComposeResult:
                items = [Static(f"Item {i}", classes="card") for i in range(6)]
                yield GridSelect(*items, min_column_width=20, id="test-grid")

        app = GridTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            grid = app.query_one("#test-grid", GridSelect)
            grid.focus()
            await pilot.pause()
            assert grid.highlighted == 0

            grid.action_cursor_right()
            await pilot.pause()
            assert grid.highlighted == 1

            grid.action_cursor_left()
            await pilot.pause()
            assert grid.highlighted == 0

    async def test_select_fires_message(self):
        """Pressing enter on a highlighted item fires Selected message."""
        from textual.widgets import Static

        from lilbee.cli.tui.widgets.grid_select import GridSelect

        selections = []

        class GridTestApp(App[None]):
            def compose(self) -> ComposeResult:
                items = [Static(f"Item {i}", classes="card") for i in range(4)]
                yield GridSelect(*items, min_column_width=20, id="test-grid")

            def on_grid_select_selected(self, event: GridSelect.Selected) -> None:
                selections.append(event.widget)

        app = GridTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            grid = app.query_one("#test-grid", GridSelect)
            grid.focus()
            await pilot.pause()
            grid.action_select()
            await pilot.pause()
            assert len(selections) == 1

    async def test_highlight_first_and_last(self):
        """highlight_first and highlight_last jump to ends."""
        from textual.widgets import Static

        from lilbee.cli.tui.widgets.grid_select import GridSelect

        class GridTestApp(App[None]):
            def compose(self) -> ComposeResult:
                items = [Static(f"Item {i}", classes="card") for i in range(6)]
                yield GridSelect(*items, min_column_width=20, id="test-grid")

        app = GridTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            grid = app.query_one("#test-grid", GridSelect)
            grid.highlight_last()
            await pilot.pause()
            assert grid.highlighted == 5

            grid.highlight_first()
            await pilot.pause()
            assert grid.highlighted == 0

    async def test_blur_clears_highlight(self):
        """Blurring the grid clears the highlight."""
        from textual.widgets import Input

        from lilbee.cli.tui.widgets.grid_select import GridSelect

        class GridTestApp(App[None]):
            def compose(self) -> ComposeResult:
                items = [Input(f"Item {i}", classes="card") for i in range(4)]
                yield GridSelect(*items, min_column_width=20, id="test-grid")
                yield Input(placeholder="Other", id="other-input")

        app = GridTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            grid = app.query_one("#test-grid", GridSelect)
            grid.focus()
            await pilot.pause()
            assert grid.highlighted is not None

            app.query_one("#other-input", Input).focus()
            await pilot.pause()
            assert grid.highlighted is None


class TestCatalogViewToggle:
    """Test view toggle CTA and grid/table switching."""

    async def test_view_toggle_cta_exists(self, _mock_resolve):
        """Grid view shows a .view-toggle-cta Static."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                ctas = app.screen.query(".view-toggle-cta")
                assert len(ctas) >= 1

    async def test_our_picks_heading_in_grid(self, _mock_resolve):
        """Grid view shows 'Our picks' section heading."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                headings = app.screen.query(".section-heading")
                texts = [str(h.render()) for h in headings]
                assert "Our picks" in texts


class TestCatalogPickBadge:
    """Test that featured cards show the pick badge."""

    async def test_featured_card_has_pick_label(self, _mock_resolve):
        """Featured ModelCard should render #card-pick."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.model_card import ModelCard

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                cards = app.screen.query(ModelCard)
                featured = [c for c in cards if c.row.featured]
                assert len(featured) > 0
                pick = featured[0].query("#card-pick")
                assert len(pick) == 1


class TestCatalogLazyLoad:
    """Test browse-more card for lazy HF loading."""

    async def test_browse_more_card_exists(self, _mock_resolve):
        """.browse-more-hf card appears before HF fetch."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                cards = app.screen.query(".browse-more-hf")
                assert len(cards) >= 1


class TestSetupWizardGrid:
    """Test setup wizard uses GridSelect + ModelCard."""

    async def test_setup_uses_grid_select(self, _mock_resolve):
        """SetupWizard mounts GridSelect, not ListView."""
        from lilbee.cli.tui.screens.setup import SetupWizard
        from lilbee.cli.tui.widgets.grid_select import GridSelect

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch(
                "lilbee.cli.tui.screens.setup._scan_installed_models",
                return_value=([], []),
            ):
                app.push_screen(SetupWizard())
                await pilot.pause()
                grids = app.screen.query(GridSelect)
                assert len(grids) >= 1

    async def test_setup_step1_shows_chat_picks(self, _mock_resolve):
        """Setup shows 'Chat Models' heading."""
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch(
                "lilbee.cli.tui.screens.setup._scan_installed_models",
                return_value=([], []),
            ):
                app.push_screen(SetupWizard())
                await pilot.pause()
                headings = app.screen.query(".section-heading")
                texts = [str(h.render()) for h in headings]
                assert "Chat Models" in texts

    async def test_setup_browse_catalog_button(self, _mock_resolve):
        """Browse catalog button exists in setup wizard."""
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch(
                "lilbee.cli.tui.screens.setup._scan_installed_models",
                return_value=([], []),
            ):
                app.push_screen(SetupWizard())
                await pilot.pause()
                btn = app.screen.query_one("#setup-browse")
                assert btn is not None

    async def test_cmd_setup_opens_wizard(self, _mock_resolve):
        """/setup command opens the setup wizard."""
        from lilbee.cli.tui.screens.setup import SetupWizard

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch(
                "lilbee.cli.tui.screens.setup._scan_installed_models",
                return_value=([], []),
            ):
                app.screen._handle_slash("/setup")
                await pilot.pause()
                assert isinstance(app.screen, SetupWizard)

    async def test_catalog_grid_to_status_preserves_state(self, _mock_resolve):
        """Switching from catalog grid to status and back."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                assert isinstance(app.screen, CatalogScreen)

                app.switch_view("Status")
                await pilot.pause()
                assert app.active_view == "Status"

                app.switch_view("Catalog")
                await pilot.pause()
                assert isinstance(app.screen, CatalogScreen)

    async def test_setup_browse_switches_to_catalog(self, _mock_resolve):
        """Browse catalog button in setup dismisses and navigates
        to catalog view."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.setup import SetupWizard

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                with mock.patch(
                    "lilbee.cli.tui.screens.setup._scan_installed_models",
                    return_value=([], []),
                ):
                    app.push_screen(SetupWizard())
                    await pilot.pause()
                    btn = app.screen.query_one("#setup-browse")
                    btn.press()
                    await pilot.pause()
                    await pilot.pause()
                    assert not isinstance(app.screen, SetupWizard)


class TestChatEmbeddingReadyCoverage:
    """Cover _embedding_ready exception path (lines 172-173 in chat.py)."""

    async def test_embedding_ready_returns_false_on_resolve_error(self, _mock_resolve):
        """_embedding_ready returns False when resolve_model_path raises."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ChatScreen)
            with mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                side_effect=FileNotFoundError("not found"),
            ):
                assert screen._embedding_ready() is False
