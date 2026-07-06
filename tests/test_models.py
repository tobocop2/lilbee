"""Tests for models.py: RAM detection, model selection, picker UI, auto-install."""

from unittest import mock

import pytest

from lilbee.modelhub import models
from lilbee.modelhub.models import MODEL_CATALOG, ModelInfo


class TestModelCatalog:
    def test_not_empty(self):
        assert len(MODEL_CATALOG) > 0

    def test_all_model_info(self):
        for m in MODEL_CATALOG:
            assert isinstance(m, ModelInfo)

    def test_derived_from_catalog(self):
        """MODEL_CATALOG entries match catalog.py's FEATURED_CHAT."""
        from lilbee.catalog import FEATURED_CHAT

        assert len(MODEL_CATALOG) == len(FEATURED_CHAT)
        for mc, fc in zip(MODEL_CATALOG, FEATURED_CHAT, strict=True):
            assert mc.ref == fc.ref

    def test_frozen(self):
        with pytest.raises(AttributeError):
            MODEL_CATALOG[0].ref = "nope"  # type: ignore[misc]


class TestGetSystemRamGb:
    def test_unix_ram_detection(self):
        mock_sysconf = mock.Mock(
            side_effect=lambda key: {
                "SC_PHYS_PAGES": 4194304,
                "SC_PAGE_SIZE": 4096,
            }[key]
        )
        with (
            mock.patch.object(models.sys, "platform", "linux"),
            mock.patch("os.sysconf", mock_sysconf, create=True),
        ):
            ram = models.get_system_ram_gb()
            assert abs(ram - 16.0) < 0.01

    def test_fallback_on_error(self):
        with (
            mock.patch.object(models.sys, "platform", "linux"),
            mock.patch("os.sysconf", side_effect=OSError("not supported"), create=True),
        ):
            assert models.get_system_ram_gb() == 8.0

    def test_windows_ram_detection(self):
        """Mock ctypes.windll to simulate Windows RAM detection."""
        mock_windll = mock.MagicMock()

        def fake_global_memory(byref_stat):
            stat = byref_stat._obj
            stat.ullTotalPhys = 16 * 1024**3

        mock_windll.kernel32.GlobalMemoryStatusEx.side_effect = fake_global_memory

        with (
            mock.patch.object(models.sys, "platform", "win32"),
            mock.patch("ctypes.windll", mock_windll, create=True),
        ):
            ram = models.get_system_ram_gb()
            assert abs(ram - 16.0) < 0.01

    def test_windows_fallback_on_error(self):
        mock_windll = mock.MagicMock()
        mock_windll.kernel32.GlobalMemoryStatusEx.side_effect = OSError("fail")
        with (
            mock.patch.object(models.sys, "platform", "win32"),
            mock.patch("ctypes.windll", mock_windll, create=True),
        ):
            assert models.get_system_ram_gb() == 8.0


class TestGetFreeDiskGb:
    def test_returns_free_space(self, tmp_path):
        usage = mock.Mock(free=50 * 1024**3)
        with mock.patch("shutil.disk_usage", return_value=usage):
            assert models.get_free_disk_gb(tmp_path) == 50.0

    def test_walks_up_to_existing_parent(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        usage = mock.Mock(free=10 * 1024**3)
        with mock.patch("shutil.disk_usage", return_value=usage) as mock_du:
            result = models.get_free_disk_gb(deep)
            assert result == 10.0
            mock_du.assert_called_once_with(tmp_path)


class TestPickDefaultModel:
    def test_returns_model_info(self):
        result = models.pick_default_model(8.0)
        assert isinstance(result, ModelInfo)

    def test_low_ram_picks_small(self):
        result = models.pick_default_model(4.0)
        assert result.min_ram_gb <= 4.0

    def test_8gb_ram(self):
        result = models.pick_default_model(8.0)
        assert result.min_ram_gb <= 8.0

    def test_16gb_ram(self):
        result = models.pick_default_model(16.0)
        assert result.min_ram_gb <= 16.0

    def test_32gb_ram(self):
        result = models.pick_default_model(32.0)
        assert result.min_ram_gb <= 32.0

    def test_tiny_ram_picks_smallest(self):
        result = models.pick_default_model(2.0)
        assert result.min_ram_gb <= 2.0
        assert "SmolLM2" in result.ref or "Qwen3-0.6B" in result.ref


class TestModelDownloadSizeGb:
    def test_known_models(self):
        first = MODEL_CATALOG[0]
        assert models._model_download_size_gb(first.ref) == first.size_gb

    def test_unknown_model_returns_fallback(self):
        result = models._model_download_size_gb("unknown:latest")
        assert isinstance(result, float)
        assert result > 0


class TestDisplayModelPicker:
    def test_renders_table(self, capsys):
        recommended = models.display_model_picker(16.0, 50.0)
        captured = capsys.readouterr()
        assert "Available Models" in captured.err
        assert MODEL_CATALOG[0].display_name in captured.err
        assert isinstance(recommended, ModelInfo)

    def test_recommended_highlighted(self, capsys):
        recommended = models.display_model_picker(32.0, 100.0)
        assert recommended.min_ram_gb <= 32.0
        captured = capsys.readouterr()
        # The star marker should be in the output
        assert "\u2605" in captured.err

    def test_disk_warning_with_low_space(self, capsys):
        models.display_model_picker(32.0, 3.0)
        captured = capsys.readouterr()
        # Table still renders with disk info showing low space
        assert "3.0 GB free disk" in captured.err
        assert "Available Models" in captured.err

    def test_shows_system_stats(self, capsys):
        models.display_model_picker(16.0, 42.5)
        captured = capsys.readouterr()
        assert "16 GB RAM" in captured.err
        assert "42.5 GB free disk" in captured.err

    def test_shows_browse_link(self, capsys):
        models.display_model_picker(8.0, 50.0)
        captured = capsys.readouterr()
        assert models.MODELS_BROWSE_URL in captured.err


class TestPromptModelChoice:
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_default_choice(self, mock_disk_estimate):
        with mock.patch("builtins.input", return_value=""):
            result = models.prompt_model_choice(8.0)
        assert isinstance(result, ModelInfo)
        # Default = recommended for 8 GB
        assert result == models.pick_default_model(8.0)

    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_numeric_choice(self, mock_disk_estimate):
        with mock.patch("builtins.input", return_value="1"):
            result = models.prompt_model_choice(8.0)
        assert result == MODEL_CATALOG[0]

    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_invalid_then_valid(self, mock_disk_estimate):
        with mock.patch("builtins.input", side_effect=["abc", "99", "2"]):
            result = models.prompt_model_choice(8.0)
        assert result == MODEL_CATALOG[1]

    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_eof_returns_recommended(self, mock_disk_estimate):
        with mock.patch("builtins.input", side_effect=EOFError):
            result = models.prompt_model_choice(8.0)
        assert result == models.pick_default_model(8.0)

    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_keyboard_interrupt_returns_recommended(self, mock_disk_estimate):
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            result = models.prompt_model_choice(8.0)
        assert result == models.pick_default_model(8.0)


class TestValidateDiskAndPull:
    @mock.patch.object(models, "pull_with_progress")
    def test_pulls_and_returns_ref(self, mock_pull):
        ref = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
        info = ModelInfo(
            ref=ref,
            display_name="Qwen3 0.6B",
            size_gb=1.0,
            min_ram_gb=4,
            description="test",
        )
        result = models.validate_disk_and_pull(info, 50.0)
        mock_pull.assert_called_once_with(ref, console=None)
        assert result == ref

    def test_insufficient_disk_raises(self):
        info = ModelInfo(
            ref="org/Big-GGUF/big-Q4_K_M.gguf",
            display_name="Big",
            size_gb=20.0,
            min_ram_gb=32,
            description="big",
        )
        with pytest.raises(RuntimeError, match="Not enough disk space"):
            models.validate_disk_and_pull(info, 5.0)


class TestPullWithProgress:
    @mock.patch("lilbee.app.services.get_services")
    def test_calls_manager_pull(self, mock_get_manager):
        mock_manager = mock.MagicMock()

        def fake_pull(model, source, *, on_bytes=None):
            if on_bytes:
                on_bytes(100, 100)
            return

        mock_manager.pull.side_effect = fake_pull
        mock_get_manager.return_value.model_manager = mock_manager
        models.pull_with_progress("test-model")
        mock_manager.pull.assert_called_once()

    @mock.patch("lilbee.app.services.get_services")
    def test_handles_zero_total(self, mock_get_manager):
        mock_manager = mock.MagicMock()

        def fake_pull(model, source, *, on_bytes=None):
            if on_bytes:
                on_bytes(0, 0)
            return

        mock_manager.pull.side_effect = fake_pull
        mock_get_manager.return_value.model_manager = mock_manager
        models.pull_with_progress("test-model")


class TestEnsureChatModel:
    """ensure_chat_model bootstraps only when no chat-task model is installed.

    It consults the task-aware ``list_installed_models`` (chat-task only), so a
    pulled vision/reranker model -- or a remote non-chat model -- no longer
    short-circuits the bootstrap.
    """

    def test_noop_when_chat_model_exists(self):
        with mock.patch.object(models, "list_installed_models", return_value=["llama3:latest"]):
            assert models.ensure_chat_model() is None  # no pull

    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch.object(models, "list_installed_models", return_value=[])
    def test_non_interactive_auto_picks(
        self, _mock_list, mock_vram_estimate, mock_disk_estimate, mock_pull
    ):
        with mock.patch.object(models.sys.stdin, "isatty", return_value=False):
            pulled = models.ensure_chat_model()
        expected = models.pick_default_model(32.0)
        mock_pull.assert_called_once_with(expected.ref, console=None)
        assert pulled == expected.ref

    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=8.0)
    @mock.patch.object(models, "list_installed_models", return_value=[])
    def test_non_interactive_low_ram(
        self, _mock_list, mock_vram_estimate, mock_disk_estimate, mock_pull
    ):
        with mock.patch.object(models.sys.stdin, "isatty", return_value=False):
            models.ensure_chat_model()
        expected = models.pick_default_model(8.0)
        mock_pull.assert_called_once_with(expected.ref, console=None)

    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=16.0)
    @mock.patch.object(models, "list_installed_models", return_value=[])
    def test_interactive_uses_picker(
        self, _mock_list, mock_vram_estimate, mock_disk_estimate, mock_pull
    ):
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value="1"),
        ):
            models.ensure_chat_model()
        mock_pull.assert_called_once_with(MODEL_CATALOG[0].ref, console=None)

    @mock.patch.object(models, "get_free_disk_gb", return_value=0.01)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch.object(models, "list_installed_models", return_value=[])
    def test_insufficient_disk_raises(self, _mock_list, mock_vram_estimate, mock_disk_estimate):
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=False),
            pytest.raises(RuntimeError, match="Not enough disk space"),
        ):
            models.ensure_chat_model()

    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=16.0)
    def test_non_chat_only_install_still_pulls(self, mock_ram, mock_disk, mock_pull):
        """Regression (bb-ziks.67): an installed reranker/vision model is not a chat
        model, so the real (task-filtered) list_installed_models excludes it and the
        bootstrap still pulls one rather than short-circuiting. Drives the real
        list_installed_models through reclassify_by_name (no mock of the classifier)."""
        from lilbee.catalog.types import ModelTask

        # A manifest that declares task="chat" but whose ref names a reranker; the
        # name-based reclassifier must demote it so it never counts as a chat model.
        reranker = mock.Mock(ref="bge-reranker-v2-m3", task=ModelTask.CHAT)
        with (
            mock.patch.object(models, "ModelRegistry") as mock_registry,
            mock.patch("lilbee.modelhub.model_manager.classify_all_remote_models", return_value=[]),
            mock.patch.object(models.sys.stdin, "isatty", return_value=False),
        ):
            mock_registry.return_value.list_installed.return_value = [reranker]
            # The reranker is excluded, so the chat-task list is empty...
            assert models.list_installed_models() == []
            # ...and the bootstrap therefore pulls a real chat model.
            pulled = models.ensure_chat_model()
        assert pulled == models.pick_default_model(16.0).ref
        mock_pull.assert_called_once()


# ensure_tag was removed alongside the alias system. Tag normalisation has
# no meaning under the new HF-keyed identity, so callers either pass a
# canonical ref through directly or fail loudly via parse_model_ref.
