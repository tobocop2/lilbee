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

        with mock.patch(
            "lilbee.cli.tui.widgets.model_bar._collect_native_models"
        ) as mock_native:
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

        with mock.patch(
            "lilbee.cli.tui.widgets.model_bar._collect_native_models"
        ), mock.patch("lilbee.cli.tui.widgets.model_bar._collect_remote_models"):
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
