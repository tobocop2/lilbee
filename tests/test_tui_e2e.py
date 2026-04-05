"""End-to-end TUI integration tests.

These tests launch the real Textual app and verify observable behavior.
They exist because unit tests with mocks passed while the app was broken.
Every test here reproduces a bug that was found by manual testing.
"""

from __future__ import annotations

from unittest import mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Collapsible, Footer, Static

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


# -- Bug: embedding model set but search shows "chat only" --


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


# -- Bug: chat dropdown showed vision models --


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


# -- Bug: model switch during stream caused segfault --


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
            from textual.widgets import Select

            event = mock.MagicMock()
            event.value = "new-model.gguf"
            event.select = mock.MagicMock()
            event.select.id = "chat-model-select"

            with mock.patch("lilbee.services.reset_services"):
                bar.on_select_changed(event)

            screen.action_cancel_stream.assert_called_once()


# -- Bug: NavBar missing from some screens --


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
                app._switch_view(view)
                await pilot.pause()
                nav = app.screen.query_one("#global-nav-bar")
                assert nav is not None, f"NavBar missing on {view} screen"


# -- Bug: mode indicator not showing --


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


# -- Bug: view cycling broken --


class TestViewCycling:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families")
    async def test_cycles_all_five_views(self, _fam, _cat, _mock_resolve):
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app._active_view == "Chat"

            expected = ["Models", "Status", "Settings", "Tasks", "Chat"]
            for view in expected:
                app.action_nav_next()
                await pilot.pause()
                assert app._active_view == view, f"Expected {view}, got {app._active_view}"


# -- Bug: chat-only banner shown despite embedding model set --


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


# -- Bug: download progress stuck at 0% --


class TestDownloadProgress:
    def test_progress_poller_reports_file_size(self, tmp_path):
        """Progress poller must report bytes from .incomplete file."""
        import threading
        import time

        from lilbee.catalog import _start_progress_poller

        # Set up fake HF cache structure
        repo_dir = tmp_path / "hub" / "models--test--repo" / "blobs"
        repo_dir.mkdir(parents=True)
        incomplete = repo_dir / "abc123.incomplete"
        incomplete.write_bytes(b"\0" * 500)

        results = []

        def callback(downloaded, total):
            results.append((downloaded, total))

            stop = _start_progress_poller(
                "test/repo", callback, 1000, cache_dir=str(tmp_path / "hub")
            )
            time.sleep(1.5)  # 3 poll cycles at 0.5s
            stop()

        assert len(results) > 0
        assert results[0] == (500, 1000)

    def test_progress_poller_tracks_growth(self, tmp_path):
        """Poller must report increasing sizes as file grows."""
        import time

        from lilbee.catalog import _start_progress_poller

        repo_dir = tmp_path / "hub" / "models--test--repo" / "blobs"
        repo_dir.mkdir(parents=True)
        incomplete = repo_dir / "abc123.incomplete"
        incomplete.write_bytes(b"\0" * 100)

        results = []

        def callback(downloaded, total):
            results.append(downloaded)

            stop = _start_progress_poller(
                "test/repo", callback, 1000, cache_dir=str(tmp_path / "hub")
            )
            time.sleep(0.7)
            # Grow the file
            with open(incomplete, "ab") as f:
                f.write(b"\0" * 400)
            time.sleep(0.7)
            stop()

        assert len(results) >= 2
        assert results[-1] > results[0]

    def test_progress_poller_no_crash_on_missing_dir(self, tmp_path):
        """Poller must not crash if cache dir doesn't exist."""
        import time

        from lilbee.catalog import _start_progress_poller

        with mock.patch("lilbee.catalog.HF_HUB_CACHE", str(tmp_path / "nonexistent")):
            stop = _start_progress_poller("test/repo", lambda d, t: None, 1000)
            time.sleep(0.7)
            stop()
        # No exception = pass

    def test_download_model_calls_progress(self):
        """download_model must call on_progress during download."""
        from lilbee.catalog import CatalogModel, download_model

        entry = CatalogModel(
            name="test",
            hf_repo="test/repo",
            gguf_filename="test.gguf",
            size_gb=0.001,
            min_ram_gb=1,
            description="test",
            featured=False,
            downloads=0,
            task="chat",
        )
        results = []

        # Mock hf_hub_download to simulate a download
        def fake_download(**kwargs):
            progress_updater = kwargs.get("progress_updater")
            if progress_updater:
                progress_updater(500, 1000)
                progress_updater(1000, 1000)
            return str(cfg.models_dir / "test.gguf")

        # Create the dest file so _register_model works
        (cfg.models_dir / "test.gguf").write_bytes(b"\0" * 100)

        with (
            mock.patch("lilbee.catalog.resolve_filename", return_value="test.gguf"),
            mock.patch("lilbee.catalog.hf_hub_download", side_effect=fake_download),
            mock.patch("lilbee.catalog._register_model"),
        ):
            # Delete dest so it triggers the download path
            dest = cfg.models_dir / "test.gguf"
            dest.unlink()
            download_model(entry, on_progress=lambda d, t: results.append((d, t)))

        assert len(results) >= 2
        assert results[-1][0] > 0


class TestTaskCenter:
    async def test_task_center_renders_with_active_task(self, _mock_resolve):
        """Task Center must render collapsible task widgets without crashing."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Add a mock task directly to the task bar
            task_bar = getattr(app, "_task_bar", None)
            assert task_bar is not None

            task_id = task_bar.add_task("Test Download", "download")

            # Update task to have progress
            task_bar.update_task(task_id, 45, "100/500 MB")

            # Switch to Task Center - this triggers on_mount which calls _refresh_tasks
            app._switch_view("Tasks")
            await pilot.pause()

            # Verify the task list has the task
            task_list = app.screen.query_one("#task-list")
            assert task_list is not None

            # Should have at least one collapsible
            collapsibles = app.screen.query(Collapsible)
            assert len(collapsibles) >= 1, (
                f"Expected at least 1 Collapsible, got {len(collapsibles)}"
            )

    async def test_task_center_renders_empty_state(self, _mock_resolve):
        """Task Center shows 'All quiet' when no tasks."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # Switch to Task Center with no tasks
            app._switch_view("Tasks")
            await pilot.pause()

            # The task-list VerticalScroll should have children
            task_list = app.screen.query_one("#task-list")
            # Should have at least one child (the empty state Static)
            assert len(task_list.children) >= 1, (
                f"Expected at least 1 child, got {len(task_list.children)}"
            )


class TestDownloadProgress:
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
