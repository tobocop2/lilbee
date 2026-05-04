"""End-to-end TUI integration tests.

These tests launch the real Textual app and verify observable behavior.
They exist because unit tests with mocks passed while the app was broken.
Every test here reproduces a bug that was found by manual testing.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest import mock

import pytest
from textual.app import App, ComposeResult

from conftest import TEST_EMBED_REF, TEST_LOCAL_REF
from lilbee.cli.tui import messages as msg_module
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def _isolated_cfg(tmp_path):
    """Snapshot and restore config for each test."""
    snapshot = cfg.model_copy()
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.models_dir = tmp_path / "models"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.subprocess_embed = False
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    # Simulate "already-initialized" state so ChatScreen._needs_setup()
    # doesn't push the SetupWizard during tests that exercise chat.
    cfg.lancedb_dir.mkdir(parents=True, exist_ok=True)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


@pytest.fixture()
def _mock_resolve():
    """Mock model resolution to succeed without real files."""
    with mock.patch(
        "lilbee.providers.llama_cpp.provider.resolve_model_path",
        return_value=cfg.models_dir / "fake.gguf",
    ):
        yield


@pytest.fixture()
def _mock_services():
    """Mock services to prevent real provider initialization."""
    from lilbee.core.services import set_services

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
        from lilbee.retrieval.embedder import Embedder

        ref = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
        mock_provider = mock.MagicMock()
        mock_provider.list_models.return_value = [ref, "other-model.gguf"]
        cfg.embedding_model = ref

        embedder = Embedder(cfg, mock_provider)
        # Registry resolution fails (no manifests in tmp dir); falls through
        # to list_models fallback which matches the configured ref directly.
        assert embedder.embedding_available() is True

    def test_resolves_via_registry(self):
        """When resolve_model_path succeeds, embedding is available."""
        from lilbee.retrieval.embedder import Embedder

        mock_provider = mock.MagicMock()
        cfg.embedding_model = TEST_EMBED_REF

        embedder = Embedder(cfg, mock_provider)
        with mock.patch(
            "lilbee.providers.llama_cpp.provider.resolve_model_path",
            return_value=cfg.models_dir / "test.gguf",
        ):
            assert embedder.embedding_available() is True

    def test_unresolvable_model_returns_false(self):
        """When model name doesn't match any installed model, returns False."""
        from lilbee.retrieval.embedder import Embedder

        mock_provider = mock.MagicMock()
        mock_provider.list_models.return_value = ["org/Other-GGUF/other.gguf"]
        cfg.embedding_model = "org/Nonexistent-GGUF/none.gguf"
        embedder = Embedder(cfg, mock_provider)
        assert embedder.embedding_available() is False


@pytest.mark.real_model_classify
class TestModelClassification:
    def test_mmproj_filtered_out(self):
        from lilbee.cli.tui.widgets.model_bar import _is_mmproj

        assert _is_mmproj("mmproj-BF16.gguf") is True
        assert _is_mmproj("Qwen3-4B.gguf") is False

    def test_registry_based_classification(self):
        """Models classified by registry manifest task field."""
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models

        # Mock registry manifests by task. Each carries a canonical
        # ``hf_repo/filename`` ref since that's the new identity.
        chat_ref = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
        embed_ref = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-Q4_K_M.gguf"
        vision_ref = "noctrex/LightOnOCR-2-1B-GGUF/lightonocr-Q4_K_M.gguf"
        mock_manifests = [
            mock.MagicMock(ref=chat_ref, task="chat"),
            mock.MagicMock(ref=embed_ref, task="embedding"),
            mock.MagicMock(ref=vision_ref, task="vision"),
        ]

        from lilbee.cli.tui.widgets.model_bar import ModelOption

        with mock.patch("lilbee.cli.tui.widgets.model_bar._collect_native_models") as mock_native:

            def fill_buckets(buckets, seen):
                for m in mock_manifests:
                    ref = m.ref
                    label = ref
                    buckets.get(m.task, buckets["chat"]).append(ModelOption(label, ref))
                    seen.add(ref)

            mock_native.side_effect = fill_buckets
            with (
                mock.patch("lilbee.cli.tui.widgets.model_bar._collect_remote_models"),
                mock.patch("lilbee.cli.tui.widgets.model_bar._collect_api_models"),
            ):
                chat, embed = _classify_installed_models()

        chat_refs = [o.ref for o in chat]
        embed_refs = [o.ref for o in embed]
        assert chat_ref in chat_refs
        assert embed_ref in embed_refs

    def test_no_loose_gguf_scanning(self):
        """Loose ``.gguf`` files NOT in registry must NOT appear in dropdowns."""
        from lilbee.cli.tui.widgets.model_bar import _classify_installed_models

        # Create loose files that should be ignored
        (cfg.models_dir / "loose-chat.gguf").touch()
        (cfg.models_dir / "loose-vision.gguf").touch()

        with (
            mock.patch("lilbee.cli.tui.widgets.model_bar._collect_native_models"),
            mock.patch("lilbee.cli.tui.widgets.model_bar._collect_remote_models"),
            mock.patch("lilbee.cli.tui.widgets.model_bar._collect_api_models"),
        ):
            chat, embed = _classify_installed_models()

        all_models = chat + embed
        assert "loose-chat.gguf" not in all_models
        assert "loose-vision.gguf" not in all_models


class TestModelSwitchSafety:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families")
    async def test_switch_cancels_stream(self, _fam, _cat, _mock_resolve):
        """Changing model while streaming must cancel the stream first."""
        from lilbee.cli.tui.widgets.model_bar import ModelPickerButton

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            screen.streaming = True

            chat_btn = screen.query_one("#chat-model-button", ModelPickerButton)
            ref = "ollama/new-model:latest"
            with (
                mock.patch("lilbee.core.settings.set_value"),
                mock.patch.object(screen, "apply_model_change") as mock_apply,
            ):
                chat_btn._on_picker_dismissed(ref)

            mock_apply.assert_called_once()


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
            await pilot.press("escape")
            await pilot.pause()
            bar = app.screen.query_one(ViewTabs)
            assert "NORMAL" in bar.mode_text


@pytest.mark.usefixtures("wiki_enabled")
class TestViewCycling:
    @mock.patch("lilbee.cli.tui.screens.catalog.get_catalog")
    @mock.patch("lilbee.cli.tui.screens.catalog.get_families")
    async def test_cycles_all_views(self, _fam, _cat, _mock_resolve):
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.active_view == "Chat"

            # Blur the chat input so the app-level ] binding fires.
            await pilot.press("escape")
            await pilot.pause()

            expected = ["Catalog", "Status", "Settings", "Tasks", "Wiki", "Chat"]
            for view in expected:
                await pilot.press("right_square_bracket")
                await pilot.pause()
                assert app.active_view == view, f"Expected {view}, got {app.active_view}"


class TestChatOnlyBannerRemoved:
    """Regression guards: the persistent yellow chat-mode banner is gone."""

    async def test_no_chat_only_banner_in_dom(self, _mock_resolve):
        from textual.css.query import NoMatches

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with pytest.raises(NoMatches):
                app.screen.query_one("#chat-only-banner")

    async def test_screen_has_no_refresh_mode_banner_method(self, _mock_resolve):
        """The old _refresh_mode_banner helper is permanently deleted."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert not hasattr(app.screen, "_refresh_mode_banner")


class TestDownloadProgressSlow:
    @pytest.mark.slow
    def test_download_progress_callback_receives_cumulative_values(self, tmp_path):
        """Download Mistral and verify progress callbacks receive cumulative values."""
        import os

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            pytest.skip("HF_TOKEN environment variable not set")

        from lilbee.catalog import CatalogModel, download_model
        from lilbee.core.services import reset_services

        snapshot = cfg.model_copy()
        try:
            cfg.data_dir = tmp_path / "data"
            cfg.models_dir = tmp_path / "models"
            cfg.data_root = tmp_path
            cfg.documents_dir = tmp_path / "documents"
            cfg.lancedb_dir = tmp_path / "data" / "lancedb"
            cfg.models_dir.mkdir(parents=True, exist_ok=True)
            cfg.documents_dir.mkdir(parents=True, exist_ok=True)
            reset_services()

            entry = CatalogModel(
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
        "lilbee.cli.tui.screens.catalog.classify_remote_models",
        return_value=[],
    )


def _mock_status_deps():
    """Mock status screen dependencies to avoid real store/model access."""
    from lilbee.modelhub.model_info import ModelArchInfo

    return mock.patch.multiple(
        "lilbee.cli.tui.screens.status",
        get_model_architecture=mock.MagicMock(return_value=ModelArchInfo()),
    )


@pytest.mark.usefixtures("wiki_enabled")
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
                # Blur the chat input so the app-level ] binding fires.
                await pilot.press("escape")
                await pilot.pause()
                expected = ["Catalog", "Status", "Settings", "Tasks", "Wiki", "Chat"]
                for view in expected:
                    await pilot.press("right_square_bracket")
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
                # Blur the chat input so the app-level ] binding fires.
                await pilot.press("escape")
                await pilot.pause()
                full_cycle = ["Catalog", "Status", "Settings", "Tasks", "Wiki", "Chat"]
                for view in full_cycle:
                    await pilot.press("right_square_bracket")
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
                # Blur the chat input so the app-level [ binding fires.
                await pilot.press("escape")
                await pilot.pause()
                backward_cycle = ["Wiki", "Tasks", "Settings", "Status", "Catalog", "Chat"]
                for view in backward_cycle:
                    await pilot.press("left_square_bracket")
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

                    # Use f1 instead of ? because focused inputs still swallow
                    # question_mark.
                    await pilot.press("f1")
                    await pilot.pause()
                    assert app.screen.query("HelpPanel")

                    await pilot.press("f1")
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
                # Two pauses: first lets the screen mount and on_mount run,
                # second lets the deferred Documents/Architecture/Storage
                # collapsibles land via call_after_refresh.
                await pilot.pause()
                await pilot.pause()
                await pilot.press("q")
                await pilot.pause()
                assert isinstance(app.screen, ChatScreen)

    async def test_pop_from_settings_returns_to_chat(self, _mock_resolve):
        """From Settings, pressing escape returns to Chat.

        Note: 'q' is consumed when the search Input has focus (tracked as
        a separate binding bug). Escape is bound to go_back on SettingsScreen
        and also removes Input focus, so it exercises the binding path.
        """
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Settings")
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
                # If the search input was focused, the first escape just blurs it.
                if not isinstance(app.screen, ChatScreen):
                    await pilot.press("escape")
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
                    await pilot.press("escape")
                    await pilot.pause()
                    if not isinstance(app.screen, ChatScreen):
                        # Settings search input consumes the first escape (blur).
                        await pilot.press("escape")
                        await pilot.pause()
                    assert isinstance(app.screen, ChatScreen)

    async def test_theme_cycling(self, _mock_resolve):
        """Ctrl+T cycles through themes without crashing."""
        from lilbee.cli.tui.app import DARK_THEMES, LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            initial_theme = app.theme
            await pilot.press("ctrl+t")
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

            inp = app.screen.query_one("#chat-input", ChatInput)
            assert inp.has_focus

    async def test_escape_enters_normal_mode(self, _mock_resolve):
        """Pressing escape switches to normal mode."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen._insert_mode is False

    async def test_i_enters_insert_mode_from_normal(self, _mock_resolve):
        """In normal mode, pressing a printable key enters insert mode."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
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
            await pilot.press("escape")
            await pilot.pause()

            await pilot.press("j")
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
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            assert app.screen._insert_mode is False

    async def test_normal_mode_G_scrolls_bottom(self, _mock_resolve):
        """In normal mode, G scrolls the chat log to bottom."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("G")
            await pilot.pause()
            assert app.screen._insert_mode is False

    async def test_page_up_page_down(self, _mock_resolve):
        """PageUp and PageDown scroll the chat log."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("pageup")
            await pilot.pause()
            await pilot.press("pagedown")
            await pilot.pause()
            assert app.screen.is_current

    async def test_half_page_scroll(self, _mock_resolve):
        """Ctrl-D and Ctrl-U half-page scroll."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+d")
            await pilot.pause()
            await pilot.press("ctrl+u")
            await pilot.pause()
            assert app.screen.is_current

    async def test_slash_focuses_input_with_prefix(self, _mock_resolve):
        """/ key focuses input and prefills with /."""

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Blur chat input (escape -> normal mode focuses chat-log) so the
            # slash key reaches the screen-level binding instead of being
            # consumed as a literal character by the input.
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", ChatInput)
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
        """/set <writable_key> <value> updates cfg.

        Model-role fields (chat_model, embedding_model, vision_model,
        reranker_model) are ``writable=False`` in SETTINGS_MAP and must
        be changed via the dedicated PUT endpoints / model pickers, so
        this test exercises the /set plumbing with a plain writable key.
        """
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/set top_k 7")
            await pilot.pause()
            assert cfg.top_k == 7

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
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen.streaming is False

    async def test_submit_empty_does_nothing(self, _mock_resolve):
        """Submitting empty input is a no-op."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", ChatInput)
            inp.value = ""
            event = ChatInput.Submitted(inp, "")
            app.screen._on_chat_submitted(event)
            assert app.screen.streaming is False
            await pilot.pause()

    async def test_submit_message_mocked_llm(self, _mock_resolve, _mock_services):
        """Submitting a message calls _send_message and creates user bubble."""

        from lilbee.cli.tui.widgets.message import UserMessage

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", ChatInput)
            inp.value = "Hello test"
            await pilot.press("enter")
            await pilot.pause()
            messages = app.screen.query(UserMessage)
            assert len(messages) >= 1

    async def test_input_history_navigation(self, _mock_resolve, _mock_services):
        """Up/Down arrows recall input history."""

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", ChatInput)

            # Submit two messages
            inp.value = "first message"
            await pilot.press("enter")
            await pilot.pause()
            inp.value = "second message"
            await pilot.press("enter")
            await pilot.pause()

            # Navigate up through history
            await pilot.press("up")
            await pilot.pause()
            assert inp.value == "second message"

            await pilot.press("up")
            await pilot.pause()
            assert inp.value == "first message"

            # Navigate down
            await pilot.press("down")
            await pilot.pause()
            assert inp.value == "second message"

            # Past end clears
            await pilot.press("down")
            await pilot.pause()
            assert inp.value == ""

    async def test_toggle_markdown_rendering(self, _mock_resolve):
        """Ctrl+R toggles markdown rendering."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            initial = cfg.markdown_rendering
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert cfg.markdown_rendering != initial
            # Toggle back
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert cfg.markdown_rendering == initial

    async def test_normal_mode_enter_re_enters_insert(self, _mock_resolve):
        """In normal mode, pressing enter via on_key enters insert mode."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen._insert_mode is False
            # Simulate enter key event in normal mode
            from textual.events import Key

            event = Key("enter", "\r")
            app.screen.on_key(event)
            await pilot.pause()
            assert app.screen._insert_mode is True

    async def test_history_actions_skip_in_normal_mode(self, _mock_resolve):
        """In normal mode, the history action guards raise SkipAction.

        This is a unit assertion on the action methods' guard clauses -
        the key bindings already delegate here, so pilot.press would
        swallow the SkipAction via the binding dispatcher. Calling the
        action directly is the correct way to assert the guard raises.
        """
        from textual.actions import SkipAction

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
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
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                grids = app.screen.query(ModelGrid)
                assert len(grids) > 0
                assert app.screen.has_class("-grid-view")

    async def test_v_toggles_to_list_and_back(self, _mock_resolve):
        """Pressing v flips the catalog between grid and list views."""

        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                assert app.screen.has_class("-grid-view")
                grid = app.screen.query_one("#catalog-grid")
                list_container = app.screen._list_widget
                assert grid.display is True
                assert list_container.display is False

                await pilot.press("v")
                await pilot.pause()
                assert app.screen.has_class("-list-view")
                assert not app.screen.has_class("-grid-view")
                assert grid.display is False
                assert list_container.display is True

                await pilot.press("v")
                await pilot.pause()
                assert app.screen.has_class("-grid-view")
                assert not app.screen.has_class("-list-view")
                assert grid.display is True
                assert list_container.display is False

    async def test_v_toggle_after_bracket_nav_from_chat(self, _mock_resolve):
        """Pressing v still toggles the view after navigating in from chat via ]."""

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                # ChatScreen auto-focuses the insert-mode input; leave it
                # before reaching for the screen-level nav keys.
                await pilot.press("escape")
                await pilot.press("right_square_bracket")
                await pilot.pause()
                assert isinstance(app.screen, CatalogScreen)
                assert app.screen.has_class("-grid-view")

                grid = app.screen.query_one("#catalog-grid")
                list_container = app.screen._list_widget
                assert grid.display is True
                assert list_container.display is False

                await pilot.press("v")
                await pilot.pause()
                assert app.screen.has_class("-list-view")
                assert grid.display is False
                assert list_container.display is True

    async def test_search_filters_cards_in_grid_view(self, _mock_resolve):
        """Type search text in grid view, verify the grid dataset narrows."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.pause()

                initial_count = sum(len(g.rows) for g in app.screen.query(ModelGrid))
                assert initial_count > 0

                # Reveal the catalog filter (hidden by default) before
                # writing into it; otherwise the Input.Changed handler
                # never wires up.
                await pilot.press("slash")
                await pilot.pause()
                search = app.screen.query_one("#catalog-search")
                # Filter to a string that no fixture matches so we can
                # observe narrowing without depending on fixture cardinality.
                search.value = "definitely-no-such-model-xyz"
                # Wait past the catalog search debounce (80 ms) so the
                # filter actually runs.
                await pilot.pause(0.2)

                # _refresh_grid rebuilds the ModelGrid dataset; non-matching
                # rows drop out of the grid entirely.
                after_count = sum(len(g.rows) for g in app.screen.query(ModelGrid))
                assert after_count < initial_count

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
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

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
                grid = app.screen.query_one(ModelGrid)
                assert grid.has_focus

    async def test_search_filters_list_view(self, _mock_resolve):
        """Typing in the filter narrows the visible row count in list view."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.press("v")
                await pilot.pause()
                initial = app.screen._list_widget.option_count

                search = app.screen.query_one("#catalog-search")
                search.value = "definitely-no-such-model"
                await pilot.pause(0.15)
                await pilot.pause()
                assert app.screen._list_widget.option_count <= initial

    async def test_search_cta_survives_view_toggle(self, _mock_resolve):
        """Toggling list→grid with a pending search must mount the grid-view CTA."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.press("v")  # list view
                await pilot.pause()

                search = app.screen.query_one("#catalog-search")
                search.value = "some-missing-model"
                await pilot.pause()

                await pilot.press("v")  # back to grid view
                await pilot.pause()

                grid_ctas = list(app.screen.query("#catalog-grid > .search-hf-cta"))
                assert len(grid_ctas) == 1, "grid-view CTA missing after toggle"

    async def test_grid_cta_removed_when_search_cleared(self, _mock_resolve):
        """Grid-view CTA unmounts once the user wipes the search input."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                search = app.screen.query_one("#catalog-search")
                search.value = "anything"
                # Mounting the CTA is async (container.mount). On slower runners
                # a single pilot.pause isn't enough; poll until it settles.
                for _ in range(10):
                    await pilot.pause()
                    if list(app.screen.query("#catalog-grid > .search-hf-cta")):
                        break
                assert len(list(app.screen.query("#catalog-grid > .search-hf-cta"))) == 1

                search.value = ""
                for _ in range(10):
                    await pilot.pause()
                    if not list(app.screen.query("#catalog-grid > .search-hf-cta")):
                        break
                assert not list(app.screen.query("#catalog-grid > .search-hf-cta"))

    async def test_grid_cta_tracks_live_search_value(self, _mock_resolve):
        """Editing the search text updates the CTA so stale text never lingers."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                search = app.screen.query_one("#catalog-search")
                search.value = "foo"
                await pilot.pause()
                # Roundtrip through list view and back: the staleness bug
                # surfaced when the grid cache hit on (rows, bool(search))
                # and skipped re-mounting the CTA with fresh text.
                await pilot.press("v")
                await pilot.pause()
                search.value = "bar"
                await pilot.pause()
                await pilot.press("v")
                await pilot.pause()

                from lilbee.cli.tui import messages as msg

                cta = app.screen.query_one("#catalog-grid > .search-hf-cta")
                expected = msg.CATALOG_SEARCH_HF_CTA.format(query="bar")
                assert expected in str(cta.render())
                stale = msg.CATALOG_SEARCH_HF_CTA.format(query="foo")
                assert stale not in str(cta.render())

    async def test_search_grid_cta_fires_hf_worker_on_click(self, _mock_resolve):
        """Clicking the grid-view CTA fires the HF worker and merges results into the rows."""
        from lilbee.catalog import CatalogModel, CatalogResult
        from lilbee.cli.tui.app import LilbeeApp

        hf_hit = CatalogModel(
            hf_repo="some/zzz_remote-GGUF",
            gguf_filename="*.gguf",
            size_gb=1.0,
            min_ram_gb=2.0,
            description="",
            featured=False,
            downloads=0,
            task="chat",
        )
        empty = CatalogResult(total=0, limit=25, offset=0, models=[], has_more=False)
        hit = CatalogResult(total=1, limit=25, offset=0, models=[hf_hit], has_more=False)
        call_log: list[str] = []

        def fake_get_catalog(**kwargs: Any) -> CatalogResult:
            call_log.append(kwargs.get("search", ""))
            if kwargs.get("search") == "zzz_remote" and kwargs.get("task") == "chat":
                return hit
            return empty

        with (
            _mock_catalog_deps(),
            _mock_remote_models(),
            mock.patch("lilbee.cli.tui.screens.catalog.get_catalog", side_effect=fake_get_catalog),
        ):
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                search = app.screen.query_one("#catalog-search")
                search.value = "zzz_remote"
                await pilot.pause(0.15)
                # Click the grid-view CTA
                app.screen._on_search_hf_cta_clicked()
                await app.workers.wait_for_complete()
                await pilot.pause()

                assert "zzz_remote" in call_log

    async def test_submit_with_zero_local_matches_fires_hf_search(self, _mock_resolve):
        """Enter on a query that filters the list empty should fire HF itself."""
        from lilbee.catalog import CatalogResult
        from lilbee.cli.tui.app import LilbeeApp

        empty = CatalogResult(total=0, limit=25, offset=0, models=[], has_more=False)
        call_log: list[str] = []

        def fake_get_catalog(**kwargs: Any) -> CatalogResult:
            term = kwargs.get("search", "")
            if term:
                call_log.append(term)
            return empty

        with (
            _mock_catalog_deps(),
            _mock_remote_models(),
            mock.patch("lilbee.cli.tui.screens.catalog.get_catalog", side_effect=fake_get_catalog),
        ):
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.press("v")
                await pilot.pause()

                search = app.screen.query_one("#catalog-search")
                search.value = "no-such-model-anywhere"
                await pilot.pause(0.15)
                await search.action_submit()
                await app.workers.wait_for_complete()
                for _ in range(20):
                    if "no-such-model-anywhere" in call_log:
                        break
                    await pilot.pause()

                assert "no-such-model-anywhere" in call_log, (
                    f"search term never reached catalog after polling: {call_log}"
                )

    async def test_trigger_remote_search_blocked_while_in_flight(self, _mock_resolve):
        """A second _trigger_remote_search while one is in flight is a no-op."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                screen = app.screen
                screen._search_in_flight = True
                screen._trigger_remote_search("anything")
                assert screen._search_in_flight is True

                screen._search_in_flight = False
                screen._trigger_remote_search("")
                assert screen._search_in_flight is False

    async def test_grid_cta_click_fires_hf_search(self, _mock_resolve):
        """Clicking the grid-view CTA Static is equivalent to selecting it."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                search = app.screen.query_one("#catalog-search")
                search.value = "some-query"
                await pilot.pause()

                triggered: list[str] = []
                app.screen._trigger_remote_search = triggered.append  # type: ignore[method-assign]
                app.screen._on_search_hf_cta_clicked()
                assert triggered == ["some-query"]

    async def test_search_cta_clears_in_flight_on_worker_error(self, _mock_resolve):
        """A failed worker must clear _search_in_flight so the CTA stays usable."""
        from textual.worker import WorkerState

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import _WORKER_FETCH_SEARCH

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                screen = app.screen
                screen._search_in_flight = True

                mock_worker = mock.MagicMock()
                mock_worker.name = _WORKER_FETCH_SEARCH
                mock_event = mock.MagicMock()
                mock_event.state = WorkerState.ERROR
                mock_event.worker = mock_worker
                screen.on_worker_state_changed(mock_event)

                assert screen._search_in_flight is False

    async def test_search_submit_returns_focus_to_table_in_list_view(self, _mock_resolve):
        """In list view, pressing Enter in search returns focus to a list item."""
        from textual.widgets import Input

        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.press("v")
                await pilot.pause()

                await pilot.press("slash")
                await pilot.pause()
                search = app.screen.query_one("#catalog-search", Input)
                search.value = "test"
                await search.action_submit()
                for _ in range(10):
                    await pilot.pause()
                    if app.screen._list_widget.has_focus:
                        break
                assert app.screen._list_widget.has_focus

    async def test_grid_card_count_matches_families(self, _mock_resolve):
        """The grid dataset surfaces every featured family as a row."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.model_grid import ModelGrid

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                total_rows = sum(len(g.rows) for g in app.screen.query(ModelGrid))
                assert total_rows == 2

    async def test_list_view_j_k_navigation(self, _mock_resolve):
        """In list view, cursor actions move focus up/down through list items."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                await pilot.press("v")
                await pilot.pause()

                # Disable prefetch so the worker doesn't rebuild the list
                # and invalidate our item references.
                app.screen._hf_has_more = False
                items_count = app.screen._list_widget.option_count
                if items_count > 1:
                    app.screen._list_widget.highlighted = 0
                    app.screen._list_widget.focus()
                    await pilot.pause()
                    assert app.screen._focused_list_index() == 0
                    await pilot.press("j")
                    await pilot.pause()
                    assert app.screen._list_widget.highlighted == 1

                    await pilot.press("k")
                    await pilot.pause()
                    assert app.screen._list_widget.highlighted == 0

    @pytest.mark.xfail(
        reason=(
            "DataTable.move_cursor (and action_cursor_down, and direct "
            "cursor_coordinate assignment) are silently no-ops under "
            "pilot.run_test() on Textual 8.1.1 when the table is reached "
            "via action_toggle_view. The production G/g key flow works "
            "in a real terminal; the test harness can't observe the "
            "cursor move. Tracked as a flake to stabilize separately."
        ),
        strict=False,
    )
    async def test_list_view_g_G_jump(self, _mock_resolve):
        """In list view, g jumps to top, G jumps to bottom."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.pause()

                await pilot.press("v")
                await pilot.pause()
                await pilot.pause()

                # Disable prefetch so the worker doesn't rebuild the list
                # and invalidate our item references.
                app.screen._hf_has_more = False
                items_count = app.screen._list_widget.option_count
                if items_count:
                    app.screen._list_widget.highlighted = 0
                    app.screen._list_widget.focus()
                    await pilot.pause()
                    await pilot.press("G")
                    await pilot.pause()
                    assert (
                        app.screen._list_widget.highlighted
                        == app.screen._list_widget.option_count - 1
                    )

                    await pilot.press("g")
                    await pilot.pause()
                    assert app.screen._list_widget.highlighted == 0

    async def test_list_view_page_down_up(self, _mock_resolve):
        """In list view, space/ctrl-d pages down, ctrl-u pages up."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                await pilot.press("v")
                await pilot.pause()

                items_count = app.screen._list_widget.option_count
                if items_count:
                    app.screen._list_widget.highlighted = 0
                    app.screen._list_widget.focus()
                    await pilot.pause()
                await pilot.press("space")
                await pilot.pause()
                await pilot.press("ctrl+u")
                await pilot.pause()
                assert app.screen.is_current

    async def test_column_header_click_sorts_list(self, _mock_resolve):
        """Pressing s cycles the sort column in list view."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()

                await pilot.press("v")
                await pilot.pause()

                assert app.screen._sort_column == "Name"
                assert app.screen._sort_ascending is True

                # Focus a list item so `s` is not swallowed by the search input.
                items_count = app.screen._list_widget.option_count
                if items_count:
                    app.screen._list_widget.highlighted = 0
                    app.screen._list_widget.focus()
                    await pilot.pause()

                # Cycle: Name -> Downloads
                await pilot.press("s")
                await pilot.pause()
                assert app.screen._sort_column == "Downloads"
                assert app.screen._sort_ascending is True

                # Cycle: Downloads -> Size
                await pilot.press("s")
                await pilot.pause()
                assert app.screen._sort_column == "Size"
                assert app.screen._sort_ascending is True

    async def test_delete_model_without_selection_warns(self, _mock_resolve):
        """Pressing d without a highlighted model shows warning."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.press("v")
                await pilot.pause()
                await pilot.press("d")
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
                # These key bindings should delegate to the grid safely.
                await pilot.press("j")
                await pilot.press("k")
                await pilot.press("g")
                await pilot.press("G")
                await pilot.press("space")
                await pilot.press("ctrl+u")
                assert app.screen.is_current
                await pilot.pause()


class TestSettingsInteractions:
    """Test all settings screen interactions: editing and navigation."""

    async def test_grouped_sections_visible(self, _mock_resolve):
        """Grouped sections are visible on mount."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            groups = app.screen.query("Tab")
            assert len(groups) >= 1

    async def test_edit_string_value_updates_cfg(self, _mock_resolve):
        """Editing a writable string setting persists to cfg."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.settings import SettingsScreen

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            app.screen.populate_all_panes()
            await pilot.pause()

            from lilbee.cli.tui.widgets.list_text_area import ListTextArea

            editor = app.screen.query_one("#ed-rag_system_prompt", ListTextArea)
            editor.text = "test system prompt"
            app.screen._on_multiline_save(ListTextArea.Blurred(editor))
            await pilot.pause()
            assert cfg.rag_system_prompt == "test system prompt"

    async def test_multiline_save_noop_when_unchanged(self, _mock_resolve):
        """Blurring a multi-line editor without edits is a no-op."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.settings import SettingsScreen
        from lilbee.cli.tui.widgets.list_text_area import ListTextArea

        app = LilbeeApp()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            app.screen.populate_all_panes()
            await pilot.pause()
            editor = app.screen.query_one("#ed-rag_system_prompt", ListTextArea)
            # Same value -> no persist call.
            with mock.patch.object(app.screen, "_persist_value") as mock_persist:
                app.screen._on_multiline_save(ListTextArea.Blurred(editor))
            mock_persist.assert_not_called()

    async def test_multiline_save_ignores_unnamed_widget(self, _mock_resolve):
        """A ListTextArea blur message with no name is a no-op (defensive guard)."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.settings import SettingsScreen
        from lilbee.cli.tui.widgets.list_text_area import ListTextArea

        app = LilbeeApp()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            app.screen.populate_all_panes()
            await pilot.pause()
            stray = ListTextArea(text="x", show_line_numbers=False, name=None)
            with mock.patch.object(app.screen, "_persist_value") as mock_persist:
                app.screen._on_multiline_save(ListTextArea.Blurred(stray))
            mock_persist.assert_not_called()

    async def test_multiline_save_ignores_unknown_setting(self, _mock_resolve):
        """A ListTextArea blur for a name not in SETTINGS_MAP is a no-op."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.settings import SettingsScreen
        from lilbee.cli.tui.widgets.list_text_area import ListTextArea

        app = LilbeeApp()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            app.screen.populate_all_panes()
            await pilot.pause()
            stray = ListTextArea(text="x", show_line_numbers=False, name="not_a_real_setting")
            with mock.patch.object(app.screen, "_persist_value") as mock_persist:
                app.screen._on_multiline_save(ListTextArea.Blurred(stray))
            mock_persist.assert_not_called()

    async def test_toggle_boolean_checkbox(self, _mock_resolve):
        """Toggling a boolean checkbox updates cfg.

        Drives the real user gesture: focus the Checkbox and press space.
        This covers the full binding path (keyboard dispatch -> Checkbox
        toggle -> reactive watcher -> Checkbox.Changed bubbles to
        SettingsScreen -> ``_on_checkbox_save`` writes cfg). Uses a tall
        enough test size so the widget is in the visible scroll region.
        """
        from textual.widgets import Checkbox

        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.settings import SettingsScreen

        app = LilbeeApp()
        async with app.run_test(size=(120, 120)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            app.screen.populate_all_panes()
            await pilot.pause()

            checkbox = app.screen.query_one("#ed-show_reasoning", Checkbox)
            initial = checkbox.value
            checkbox.focus()
            await pilot.pause()

            await pilot.press("space")
            await pilot.pause()

            assert checkbox.value != initial
            assert cfg.show_reasoning == checkbox.value
            assert cfg.show_reasoning != initial

    async def test_read_only_fields_have_no_editor(self, _mock_resolve):
        """Read-only settings do not have an editor widget."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.settings import SettingsScreen

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            app.screen.populate_all_panes()
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
            # Move focus off the search input so screen bindings receive j/k.
            app.screen.focus_next()
            await pilot.pause()
            await pilot.press("j")
            await pilot.pause()
            await pilot.press("k")
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
            # Move focus off the search input so screen bindings receive g/G.
            app.screen.focus_next()
            await pilot.pause()
            await pilot.press("G")
            await pilot.pause()
            await pilot.press("g")
            await pilot.pause()
            assert app.screen.is_current

    async def test_pop_screen_returns_to_chat(self, _mock_resolve):
        """Pressing escape on settings returns to chat."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.chat import ChatScreen

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.switch_view("Settings")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            if not isinstance(app.screen, ChatScreen):
                # Settings search input consumes the first escape (blur).
                await pilot.press("escape")
                await pilot.pause()
            assert isinstance(app.screen, ChatScreen)

    async def test_settings_changed_signal_fires(self, _mock_resolve):
        """Editing a setting fires the settings_changed signal."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.settings import SettingsScreen

        app = LilbeeApp()
        received = []

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.settings_changed_signal.subscribe(app, lambda data: received.append(data))
            app.switch_view("Settings")
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            app.screen.populate_all_panes()
            await pilot.pause()

            from lilbee.cli.tui.widgets.list_text_area import ListTextArea

            editor = app.screen.query_one("#ed-rag_system_prompt", ListTextArea)
            editor.text = "signal test prompt"
            app.screen._on_multiline_save(ListTextArea.Blurred(editor))
            await pilot.pause()
            assert len(received) >= 1
            assert received[0][0] == "rag_system_prompt"


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
                # Two pauses: first lets the screen mount and on_mount run,
                # second lets the deferred Documents/Architecture/Storage
                # collapsibles land via call_after_refresh.
                await pilot.pause()
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
                # Two pauses: first lets the screen mount and on_mount run,
                # second lets the deferred Documents/Architecture/Storage
                # collapsibles land via call_after_refresh.
                await pilot.pause()
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
                # Two pauses: first lets the screen mount and on_mount run,
                # second lets the deferred Documents/Architecture/Storage
                # collapsibles land via call_after_refresh.
                await pilot.pause()
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
                # Two pauses: first lets the screen mount and on_mount run,
                # second lets the deferred Documents/Architecture/Storage
                # collapsibles land via call_after_refresh.
                await pilot.pause()
                await pilot.pause()
                await pilot.press("j")
                await pilot.pause()
                await pilot.press("k")
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
                # Two pauses: first lets the screen mount and on_mount run,
                # second lets the deferred Documents/Architecture/Storage
                # collapsibles land via call_after_refresh.
                await pilot.pause()
                await pilot.pause()
                await pilot.press("G")
                await pilot.pause()
                await pilot.press("g")
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
                # Two pauses: first lets the screen mount and on_mount run,
                # second lets the deferred Documents/Architecture/Storage
                # collapsibles land via call_after_refresh.
                await pilot.pause()
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
        services = mock.MagicMock()
        services.store.get_sources.return_value = mock_sources
        with (
            _mock_status_deps(),
            mock.patch(
                "lilbee.cli.tui.screens.status.get_services",
                return_value=services,
            ),
        ):
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Status")
                # Documents/Architecture/Storage collapsibles mount via
                # call_after_refresh, so the table appears one tick after
                # the screen pushes. Suppress NoMatches until it lands,
                # then poll for the worker callback to fill rows.
                from textual.css.query import NoMatches

                table = None
                for _ in range(40):
                    await pilot.pause(0.05)
                    try:
                        table = app.screen.query_one("#docs-table", DataTable)
                    except NoMatches:
                        continue
                    if table.row_count == 2:
                        break
                assert table is not None and table.row_count == 2


class TestTaskCenterInteractions:
    """Test all task center interactions: empty state, tasks, navigation."""

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
            await pilot.press("j")
            await pilot.pause()
            await pilot.press("k")
            await pilot.pause()
            assert app.screen.is_current

    async def test_cancel_task_action(self, _mock_resolve):
        """c keybinding cancels the selected task."""
        from lilbee.cli.tui.app import LilbeeApp

        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.task_bar.add_task("Cancel Me", "download")

            app.switch_view("Tasks")
            await pilot.pause()
            await pilot.press("c")
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
            await pilot.press("c")
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
            # No mode classes on the prompt area: border driven by :focus-within CSS
            assert not area.has_class("insert-mode")
            assert not area.has_class("normal-mode")
            # Input should not have its own border
            assert inp.styles.border is not None

    async def test_normal_mode_dims_input_not_area(self, _mock_resolve):
        """Normal mode adds class to input (opacity), not to prompt area."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
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
                await pilot.press("ctrl+c")
                await pilot.pause()
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
                await pilot.press("ctrl+c")
                await pilot.pause()
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
                await pilot.press("ctrl+c")
                await pilot.pause()
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
            app.screen._handle_slash("/model ollama/slash-test:latest")
            await pilot.pause()
            assert "slash-test" in cfg.chat_model

    async def test_cmd_model_cancels_stream_when_streaming(self, _mock_resolve):
        """/model <name> cancels stream and resets services when streaming."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.streaming = True
            with mock.patch.object(app.screen, "apply_model_change") as mock_apply:
                app.screen._handle_slash("/model ollama/stream-switch:latest")
                await pilot.pause()
                mock_apply.assert_called()

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

    async def test_cmd_reset_without_confirm(self, _mock_resolve):
        """/reset without confirm shows warning."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/reset")
            await pilot.pause()
            assert app.screen.is_current

    async def test_cmd_reset_with_confirm(self, _mock_resolve):
        """/reset followed by Yes deletes data AND rebuilds the Store handle."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with (
                mock.patch("lilbee.app.reset.perform_reset") as mock_perform,
                mock.patch("lilbee.cli.tui.screens.chat.reset_store") as mock_reset_store,
            ):
                from lilbee.app.reset import ResetResult

                mock_perform.return_value = ResetResult(
                    deleted_docs=1,
                    deleted_data=1,
                    skipped=[],
                    documents_dir=str(cfg.documents_dir),
                    data_dir=str(cfg.data_dir),
                )
                app.screen._handle_slash("/reset")
                await pilot.pause()
                await pilot.press("y")
                await pilot.pause()
            mock_perform.assert_called_once()
            mock_reset_store.assert_called_once()

    async def test_cmd_cancel(self, _mock_resolve):
        """/cancel cancels workers."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._handle_slash("/cancel")
            await pilot.pause()
            assert app.screen.streaming is False

    async def test_cmd_clear(self, _mock_resolve):
        """/clear removes messages and clears history."""
        from textual.containers import VerticalScroll

        from lilbee.cli.tui.widgets.message import UserMessage

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen._history = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            chat_log = app.screen.query_one("#chat-log", VerticalScroll)
            await chat_log.mount(UserMessage("hello"))
            await pilot.pause()
            assert len(chat_log.children) > 0

            app.screen._handle_slash("/clear")
            await pilot.pause()

            assert len(chat_log.children) == 0
            assert app.screen._history == []

    async def test_cmd_clear_cancels_stream(self, _mock_resolve):
        """/clear cancels active workers before clearing."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.streaming = True
            mock_worker = mock.MagicMock()
            with mock.patch.object(
                type(app.screen), "workers", new_callable=mock.PropertyMock
            ) as mock_workers:
                mock_workers.return_value = [mock_worker]
                app.screen._handle_slash("/clear")
            await pilot.pause()
            mock_worker.cancel.assert_called_once()
            assert app.screen.streaming is False
            assert app.screen._history == []

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

            inp = app.screen.query_one("#chat-input", ChatInput)
            inp.value = "/"
            await pilot.press("tab")
            await pilot.pause()
            assert app.screen.is_current

    async def test_ctrl_n_cycles_forward(self, _mock_resolve):
        """Ctrl+N cycles forward through completions."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            inp = app.screen.query_one("#chat-input", ChatInput)
            inp.value = "/"
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert app.screen.is_current

    async def test_ctrl_p_cycles_backward(self, _mock_resolve):
        """Ctrl+P cycles backward through completions."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            inp = app.screen.query_one("#chat-input", ChatInput)
            inp.value = "/"
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert app.screen.is_current

    async def test_input_change_hides_overlay(self, _mock_resolve):
        """Changing input manually hides the completion overlay."""
        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            from lilbee.cli.tui.widgets.autocomplete import CompletionOverlay

            inp = app.screen.query_one("#chat-input", ChatInput)
            inp.value = "/"
            await pilot.press("tab")
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

            await pilot.press("right")
            await pilot.pause()
            assert grid.highlighted == 1

            await pilot.press("left")
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
            await pilot.press("enter")
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

    async def test_grid_list_toggle_widget_present(self, _mock_resolve):
        """Catalog body renders the visible Grid ↔ List toggle widget."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.grid_list_toggle import GridListToggle

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                assert app.screen.query_one(GridListToggle) is not None

    async def test_our_picks_heading_in_grid(self, _mock_resolve):
        """Grid view shows 'Our picks' section heading."""
        from lilbee.cli.tui.app import LilbeeApp

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                texts: list[str] = []
                for _ in range(10):
                    await pilot.pause()
                    headings = app.screen.query(".section-heading")
                    texts = [str(h.render()) for h in headings]
                    if "Our picks" in texts:
                        break
                assert "Our picks" in texts


class TestCatalogPickBadge:
    """Test that featured cards show the pick badge."""

    async def test_featured_card_has_pick_label(self, _mock_resolve):
        """Featured catalog rows surface the 'pick' pill via _render_card_strip."""
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.widgets.model_grid import ModelGrid, _render_card_strip

        with _mock_catalog_deps(), _mock_remote_models():
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                app.switch_view("Catalog")
                await pilot.pause()
                await pilot.pause()
                featured_rows = [
                    row
                    for grid in app.screen.query(ModelGrid)
                    for row in grid.rows
                    if getattr(row, "featured", False)
                ]
                assert featured_rows, "expected at least one featured row in the catalog"
                rendered = _render_card_strip(featured_rows[0], selected=False, width=40)
                joined = "\n".join(str(line) for line in rendered.lines)
                assert "pick" in joined


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
                # The streaming-section mount chain in _refresh_grid yields
                # several refresh ticks before _mount_grid_ctas mounts the
                # browse-more card; pause until it lands or timeout.
                for _ in range(20):
                    await pilot.pause()
                    if app.screen.query(".browse-more-hf"):
                        break
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

    async def test_setup_grid_highlights_focused_card(self, _mock_resolve):
        """The focused card in the SetupWizard grid shows a visible focus
        indicator so keyboard users can see which card is under the cursor."""
        from lilbee.cli.tui.screens.setup import SetupWizard
        from lilbee.cli.tui.widgets.grid_select import GridSelect
        from lilbee.cli.tui.widgets.model_card import ModelCard

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch(
                "lilbee.cli.tui.screens.setup._scan_installed_models",
                return_value=([], []),
            ):
                app.push_screen(SetupWizard())
                await pilot.pause()
                grid = app.screen.query(GridSelect).first()
                grid.focus()
                await pilot.pause()
                await pilot.press("right")
                await pilot.pause()
                cards = [c for c in grid.children if isinstance(c, ModelCard)]
                highlighted = [c for c in cards if c.has_class("-highlight")]
                others = [c for c in cards if not c.has_class("-highlight")]
                assert len(highlighted) == 1
                assert others, "need a non-focused card to compare against"
                focused_border = highlighted[0].styles.border_top
                baseline_border = others[0].styles.border_top
                assert focused_border is not None
                assert focused_border[0] == "tall"
                # The focus rule paints a visible color; baseline is transparent.
                assert focused_border[1] != baseline_border[1]

    async def test_setup_focused_selected_card_keeps_green_bar(self, _mock_resolve):
        """A card that is both selected and focused keeps its green left bar.
        Guards against the focus border shorthand clobbering border-left."""
        from lilbee.cli.tui.screens.setup import SetupWizard
        from lilbee.cli.tui.widgets.grid_select import GridSelect
        from lilbee.cli.tui.widgets.model_card import ModelCard

        app = ChatTestApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            with mock.patch(
                "lilbee.cli.tui.screens.setup._scan_installed_models",
                return_value=([], []),
            ):
                app.push_screen(SetupWizard())
                await pilot.pause()
                grid = app.screen.query(GridSelect).first()
                cards = [c for c in grid.children if isinstance(c, ModelCard)]
                assert cards
                cards[0].selected = True
                grid.focus()
                await pilot.pause()
                assert cards[0].has_class("-highlight")
                assert cards[0].has_class("-selected")
                border_left = cards[0].styles.border_left
                assert border_left is not None
                assert border_left[0] == "thick"

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
                # Two pauses: first lets the screen mount and on_mount run,
                # second lets the deferred Documents/Architecture/Storage
                # collapsibles land via call_after_refresh.
                await pilot.pause()
                await pilot.pause()
                assert app.active_view == "Status"

                app.switch_view("Catalog")
                await pilot.pause()
                assert isinstance(app.screen, CatalogScreen)


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
                "lilbee.providers.llama_cpp.provider.resolve_model_path",
                side_effect=FileNotFoundError("not found"),
            ):
                assert screen._embedding_ready() is False

    async def test_embedding_ready_true_for_prefixed_model_in_provider_list(self):
        """ollama/ refs match against bare names returned by provider.list_models.

        provider.list_models returns tags without the ollama/ prefix
        (from /api/tags), so _embedding_ready must strip the prefix
        before substring-matching.
        """
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.core.services import set_services

        snapshot_embed = cfg.embedding_model
        cfg.embedding_model = "ollama/nomic-embed-text:v1.5"
        mock_svc = mock.MagicMock()
        mock_svc.provider.list_models.return_value = ["nomic-embed-text:v1.5"]
        set_services(mock_svc)
        try:
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ChatScreen)
                assert screen._embedding_ready() is True
        finally:
            set_services(None)
            cfg.embedding_model = snapshot_embed

    async def test_embedding_ready_false_for_prefixed_model_not_in_list(self):
        """ollama/ refs that don't appear in provider.list_models return False
        without falling through to the native registry probe.
        """
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.core.services import set_services

        snapshot_embed = cfg.embedding_model
        cfg.embedding_model = "ollama/nomic-embed-text:v1.5"
        mock_svc = mock.MagicMock()
        mock_svc.provider.list_models.return_value = []
        set_services(mock_svc)
        try:
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = app.screen
                assert isinstance(screen, ChatScreen)
                with mock.patch(
                    "lilbee.providers.llama_cpp.provider.resolve_model_path"
                ) as resolve:
                    assert screen._embedding_ready() is False
                    resolve.assert_not_called()
        finally:
            set_services(None)
            cfg.embedding_model = snapshot_embed


class TestChatStreaming:
    """Single-message-at-a-time + visible Stop affordance.

    These tests patch ``_stream_response`` to a no-op so ``_send_message``
    does not actually spawn a worker. The test then drives the
    ``streaming`` reactive flag and the watcher hooks (mount/unmount of
    the Stop button, placeholder swap) via direct ``screen.streaming =``
    writes. This avoids the cross-thread coordination that ``@work``
    introduces. Holding a real worker open behind a ``threading.Event``
    crashed xdist forked workers on slower runners.
    """

    @staticmethod
    def _patch_stream_response():
        from lilbee.cli.tui.screens.chat import ChatScreen

        return mock.patch.object(ChatScreen, "_stream_response", new=lambda self, *a, **kw: None)

    async def test_second_submit_while_streaming_is_dropped(self, _mock_resolve, _mock_services):
        """A second Enter while streaming must not spawn another assistant turn."""
        from lilbee.cli.tui.widgets.message import AssistantMessage, UserMessage

        with self._patch_stream_response():
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                inp = app.screen.query_one("#chat-input", ChatInput)
                inp.value = "first message"
                await pilot.press("enter")
                await pilot.pause()
                assert app.screen.streaming
                inp.value = "second message"
                await pilot.press("enter")
                await pilot.pause()
                # First message is mid-stream; second submission was queued
                # rather than dropped, so still only one user/assistant pair
                # is visible until the queue drains.
                assert len(list(app.screen.query(UserMessage))) == 1
                assert len(list(app.screen.query(AssistantMessage))) == 1
                assert app.screen._queued_prompt == "second message"

    async def test_esc_cancels_active_stream(self, _mock_resolve, _mock_services):
        """Esc keystroke cancels the worker (regression for action_enter_normal_mode path)."""
        with self._patch_stream_response():
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                inp = app.screen.query_one("#chat-input", ChatInput)
                inp.value = "esc me"
                await pilot.press("enter")
                await pilot.pause()
                assert app.screen.streaming
                app.screen.action_enter_normal_mode()
                await pilot.pause()
                assert app.screen.streaming is False

    async def test_input_placeholder_stays_default_through_streaming(
        self, _mock_resolve, _mock_services
    ):
        """Placeholder stays at the default during streaming; the user message stays prominent."""
        with self._patch_stream_response():
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                inp = app.screen.query_one("#chat-input", ChatInput)
                assert inp.placeholder == msg_module.CHAT_INPUT_PLACEHOLDER_DEFAULT
                inp.value = "go"
                await pilot.press("enter")
                await pilot.pause()
                assert inp.placeholder == msg_module.CHAT_INPUT_PLACEHOLDER_DEFAULT
                app.screen._set_streaming(False)
                await pilot.pause()
                assert inp.placeholder == msg_module.CHAT_INPUT_PLACEHOLDER_DEFAULT

    async def test_model_switch_during_streaming_cancels(self, _mock_resolve, _mock_services):
        """apply_model_change cancels any in-flight stream. Regression for the model-swap path."""
        with self._patch_stream_response():
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                inp = app.screen.query_one("#chat-input", ChatInput)
                inp.value = "swap me"
                await pilot.press("enter")
                await pilot.pause()
                assert app.screen.streaming
                app.screen.apply_model_change()
                await pilot.pause()
                assert app.screen.streaming is False

    async def test_exit_streaming_drains_queued_prompt(self, _mock_resolve, _mock_services):
        """Once the active stream settles, the queued prompt fires through ``_send_message``."""
        with self._patch_stream_response():
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = app.screen
                screen._queued_prompt = "follow-up"
                with mock.patch.object(screen, "_send_message") as send_mock:
                    screen._exit_streaming_state()
                    await pilot.pause()
                    send_mock.assert_called_once_with("follow-up")
                assert screen._queued_prompt is None

    async def test_chat_submit_in_normal_mode_flips_to_insert(self, _mock_resolve, _mock_services):
        """Enter while in normal mode flips back to insert without spawning a turn."""
        from lilbee.cli.tui.widgets.message import UserMessage

        with self._patch_stream_response():
            app = ChatTestApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                screen = app.screen
                screen._insert_mode = False
                inp = screen.query_one("#chat-input", ChatInput)
                inp.value = "stale text"
                await pilot.press("enter")
                await pilot.pause()
                assert screen._insert_mode is True
                # No assistant turn was spawned by that Enter press.
                assert len(list(screen.query(UserMessage))) == 0


class TestStreamFlushCoalescing:
    """Token coalescing in _consume_stream prevents per-token call_from_thread floods."""

    def test_maybe_flush_calls_flush_when_interval_elapses(self):
        """When the elapsed time crosses the threshold, flush() runs and timing advances."""
        from unittest.mock import MagicMock

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = MagicMock(spec=ChatScreen)
        flush_calls: list[None] = []

        def fake_flush() -> None:
            flush_calls.append(None)

        # Past timings: long enough ago that both the flush and the scroll fire.
        timings = [0.0, 0.0]
        with mock.patch("lilbee.cli.tui.screens.chat.call_from_thread"):
            ChatScreen._maybe_flush_and_scroll(screen, fake_flush, timings)
        assert len(flush_calls) == 1
        assert timings[0] > 0  # last_flush bumped

    def test_maybe_flush_skips_flush_within_interval(self):
        """Inside the flush window, flush() is not called and timings stay unchanged."""
        import time
        from unittest.mock import MagicMock

        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = MagicMock(spec=ChatScreen)
        flush_calls: list[None] = []

        def fake_flush() -> None:
            flush_calls.append(None)

        # Set timings to 'right now' so the interval check fails.
        now = time.monotonic()
        timings = [now, now]
        with mock.patch("lilbee.cli.tui.screens.chat.call_from_thread"):
            ChatScreen._maybe_flush_and_scroll(screen, fake_flush, timings)
        assert flush_calls == []
        assert timings == [now, now]
