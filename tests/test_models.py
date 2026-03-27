"""Tests for models.py — RAM detection, model selection, picker UI, auto-install."""

from unittest import mock

import pytest

from lilbee import models
from lilbee.config import cfg
from lilbee.models import MODEL_CATALOG, VISION_CATALOG, ModelInfo


class TestModelCatalog:
    def test_not_empty(self):
        assert len(MODEL_CATALOG) > 0

    def test_all_model_info(self):
        for m in MODEL_CATALOG:
            assert isinstance(m, ModelInfo)

    def test_sorted_by_size(self):
        sizes = [m.size_gb for m in MODEL_CATALOG]
        assert sizes == sorted(sizes)

    def test_frozen(self):
        with pytest.raises(AttributeError):
            MODEL_CATALOG[0].name = "nope"  # type: ignore[misc]


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
        assert result.name == "qwen3:1.7b"

    def test_8gb_ram(self):
        result = models.pick_default_model(8.0)
        assert result.name == "qwen3:8b"

    def test_16gb_ram(self):
        result = models.pick_default_model(16.0)
        assert result.name == "qwen3:8b"

    def test_32gb_ram(self):
        result = models.pick_default_model(32.0)
        assert result.name == "qwen3-coder:30b"

    def test_tiny_ram_picks_first(self):
        result = models.pick_default_model(2.0)
        assert result.name == "qwen3:1.7b"


class TestModelDownloadSizeGb:
    def test_known_models(self):
        assert models._model_download_size_gb("qwen3:8b") == 5.0
        assert models._model_download_size_gb("qwen3-coder:30b") == 18.0

    def test_unknown_model_returns_fallback(self):
        assert models._model_download_size_gb("unknown:latest") == 5.0


class TestDisplayModelPicker:
    def test_renders_table(self, capsys):
        recommended = models.display_model_picker(16.0, 50.0)
        captured = capsys.readouterr()
        assert "Available Models" in captured.err
        assert "qwen3:8b" in captured.err
        assert isinstance(recommended, ModelInfo)

    def test_recommended_highlighted(self, capsys):
        recommended = models.display_model_picker(32.0, 100.0)
        assert recommended.name == "qwen3-coder:30b"
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
        assert result.name == "qwen3:1.7b"

    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    def test_invalid_then_valid(self, mock_disk_estimate):
        with mock.patch("builtins.input", side_effect=["abc", "99", "2"]):
            result = models.prompt_model_choice(8.0)
        assert result.name == "qwen3:4b"

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
    @mock.patch("lilbee.settings.set_value")
    @mock.patch.object(models, "pull_with_progress")
    def test_pulls_and_persists(self, mock_pull, mock_save):
        info = ModelInfo("test:1b", 1.0, 4, "test")
        models.validate_disk_and_pull(info, 50.0)
        mock_pull.assert_called_once_with("test:1b", console=None)
        mock_save.assert_called_once_with(cfg.data_root, "chat_model", "test:1b")

    def test_insufficient_disk_raises(self):
        info = ModelInfo("test:big", 20.0, 32, "big")
        with pytest.raises(RuntimeError, match="Not enough disk space"):
            models.validate_disk_and_pull(info, 5.0)


class TestPullWithProgress:
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_calls_manager_pull(self, mock_get_manager):
        mock_manager = mock.MagicMock()

        def fake_pull(model, source, *, on_progress=None):
            if on_progress:
                on_progress({"total": 100, "completed": 100})
            return None

        mock_manager.pull.side_effect = fake_pull
        mock_get_manager.return_value = mock_manager
        models.pull_with_progress("test-model")
        mock_manager.pull.assert_called_once()

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_handles_zero_total(self, mock_get_manager):
        mock_manager = mock.MagicMock()

        def fake_pull(model, source, *, on_progress=None):
            if on_progress:
                on_progress({"total": 0, "completed": 0})
            return None

        mock_manager.pull.side_effect = fake_pull
        mock_get_manager.return_value = mock_manager
        models.pull_with_progress("test-model")


class TestEnsureChatModel:
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_noop_when_chat_models_exist(self, mock_get_manager):
        mock_manager = mock.MagicMock()
        mock_manager.list_installed.return_value = ["llama3:latest", "nomic-embed-text:latest"]
        mock_get_manager.return_value = mock_manager
        models.ensure_chat_model()  # should not raise or pull

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_connection_error_raises(self, mock_get_manager):
        mock_manager = mock.MagicMock()
        mock_manager.list_installed.side_effect = RuntimeError("refused")
        mock_get_manager.return_value = mock_manager
        with pytest.raises(RuntimeError, match="Cannot list models"):
            models.ensure_chat_model()

    @mock.patch("lilbee.settings.set_value")
    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_non_interactive_auto_picks(
        self, mock_get_manager, mock_vram_estimate, mock_disk_estimate, mock_pull, mock_save
    ):
        mock_manager = mock.MagicMock()
        mock_manager.list_installed.return_value = ["nomic-embed-text:latest"]
        mock_get_manager.return_value = mock_manager
        with mock.patch.object(models.sys.stdin, "isatty", return_value=False):
            models.ensure_chat_model()
        mock_pull.assert_called_once_with("qwen3-coder:30b", console=None)
        mock_save.assert_called_once_with(cfg.data_root, "chat_model", "qwen3-coder:30b")

    @mock.patch("lilbee.settings.set_value")
    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=8.0)
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_non_interactive_low_ram(
        self, mock_get_manager, mock_vram_estimate, mock_disk_estimate, mock_pull, mock_save_setting
    ):
        mock_manager = mock.MagicMock()
        mock_manager.list_installed.return_value = []
        mock_get_manager.return_value = mock_manager
        with mock.patch.object(models.sys.stdin, "isatty", return_value=False):
            models.ensure_chat_model()
        mock_pull.assert_called_once_with("qwen3:8b", console=None)

    @mock.patch("lilbee.settings.set_value")
    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=16.0)
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_interactive_uses_picker(
        self, mock_get_manager, mock_vram_estimate, mock_disk_estimate, mock_pull, mock_save_setting
    ):
        mock_manager = mock.MagicMock()
        mock_manager.list_installed.return_value = []
        mock_get_manager.return_value = mock_manager
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value="1"),
        ):
            models.ensure_chat_model()
        mock_pull.assert_called_once_with("qwen3:1.7b", console=None)

    @mock.patch.object(models, "get_free_disk_gb", return_value=3.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_insufficient_disk_raises(
        self, mock_get_manager, mock_vram_estimate, mock_disk_estimate
    ):
        mock_manager = mock.MagicMock()
        mock_manager.list_installed.return_value = []
        mock_get_manager.return_value = mock_manager
        with (
            mock.patch.object(models.sys.stdin, "isatty", return_value=False),
            pytest.raises(RuntimeError, match="Not enough disk space"),
        ):
            models.ensure_chat_model()

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_empty_model_list_triggers_pull(self, mock_get_manager):
        mock_manager = mock.MagicMock()
        mock_manager.list_installed.return_value = []
        mock_get_manager.return_value = mock_manager
        with (
            mock.patch.object(models, "get_system_ram_gb", return_value=16.0),
            mock.patch.object(models, "get_free_disk_gb", return_value=50.0),
            mock.patch.object(models, "pull_with_progress"),
            mock.patch.object(models.sys.stdin, "isatty", return_value=False),
        ):
            models.ensure_chat_model()

    @mock.patch("lilbee.model_manager.get_model_manager")
    def test_only_embedding_model_triggers_pull(self, mock_get_manager):
        mock_manager = mock.MagicMock()
        mock_manager.list_installed.return_value = ["nomic-embed-text:latest"]
        mock_get_manager.return_value = mock_manager
        with (
            mock.patch.object(models, "get_system_ram_gb", return_value=16.0),
            mock.patch.object(models, "get_free_disk_gb", return_value=50.0),
            mock.patch.object(models, "pull_with_progress"),
            mock.patch.object(models.sys.stdin, "isatty", return_value=False),
        ):
            models.ensure_chat_model()


class TestVisionCatalog:
    def test_catalog_not_empty(self) -> None:
        assert len(VISION_CATALOG) > 0

    def test_all_entries_are_model_info(self) -> None:
        for m in VISION_CATALOG:
            assert isinstance(m, ModelInfo)

    def test_catalog_is_ordered_by_quality(self) -> None:
        """First entry should be the best quality (LightOnOCR-2)."""
        assert "LightOnOCR" in VISION_CATALOG[0].name

    def test_frozen(self) -> None:
        with pytest.raises(AttributeError):
            VISION_CATALOG[0].name = "nope"  # type: ignore[misc]

    def test_all_names_have_explicit_tags(self) -> None:
        for m in VISION_CATALOG:
            assert ":" in m.name, f"Vision catalog entry '{m.name}' missing explicit tag"


class TestPickDefaultVisionModel:
    def test_returns_first_catalog_entry(self) -> None:
        """Always returns the best-quality model (first in catalog)."""
        assert models.pick_default_vision_model() == VISION_CATALOG[0]


class TestDisplayVisionPicker:
    def test_renders_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        m = models.display_vision_picker(32, 50.0)
        captured = capsys.readouterr()
        assert "Vision OCR Models" in captured.err
        assert isinstance(m, ModelInfo)

    def test_recommended_highlighted(self, capsys: pytest.CaptureFixture[str]) -> None:
        recommended = models.display_vision_picker(32.0, 100.0)
        assert isinstance(recommended, ModelInfo)
        captured = capsys.readouterr()
        assert "\u2605" in captured.err

    def test_disk_warning_with_low_space(self, capsys: pytest.CaptureFixture[str]) -> None:
        models.display_vision_picker(32.0, 3.0)
        captured = capsys.readouterr()
        assert "3.0 GB free disk" in captured.err
        assert "Vision OCR Models" in captured.err

    def test_shows_system_stats(self, capsys: pytest.CaptureFixture[str]) -> None:
        models.display_vision_picker(16.0, 42.5)
        captured = capsys.readouterr()
        assert "16 GB RAM" in captured.err
        assert "42.5 GB free disk" in captured.err

    def test_shows_browse_link(self, capsys: pytest.CaptureFixture[str]) -> None:
        models.display_vision_picker(8.0, 50.0)
        captured = capsys.readouterr()
        assert models.MODELS_BROWSE_URL in captured.err


class TestEnsureTag:
    def test_appends_latest_when_no_tag(self) -> None:
        assert models.ensure_tag("llama3") == "llama3:latest"

    def test_preserves_explicit_tag(self) -> None:
        assert models.ensure_tag("qwen3:8b") == "qwen3:8b"

    def test_preserves_latest_tag(self) -> None:
        assert models.ensure_tag("llama3:latest") == "llama3:latest"

    def test_empty_string_returns_empty(self) -> None:
        assert models.ensure_tag("") == ""

    def test_namespaced_model_without_tag(self) -> None:
        assert models.ensure_tag("maternion/LightOnOCR-2") == "maternion/LightOnOCR-2:latest"

    def test_namespaced_model_with_tag(self) -> None:
        assert models.ensure_tag("maternion/LightOnOCR-2:latest") == "maternion/LightOnOCR-2:latest"
