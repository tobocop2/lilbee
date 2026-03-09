"""Tests for models.py — RAM detection, model selection, auto-install."""

from types import SimpleNamespace
from unittest import mock

import pytest

from lilbee import models


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
            # Access the underlying struct from ctypes.byref wrapper
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
            # Should check tmp_path since a/b/c don't exist
            mock_du.assert_called_once_with(tmp_path)


class TestPickDefaultModel:
    def test_low_ram_picks_small(self):
        assert models.pick_default_model(8.0) == "qwen3:8b"

    def test_high_ram_picks_large(self):
        assert models.pick_default_model(32.0) == "qwen3-coder:30b"

    def test_threshold_picks_large(self):
        assert models.pick_default_model(16.0) == "qwen3-coder:30b"

    def test_below_threshold_picks_small(self):
        assert models.pick_default_model(15.9) == "qwen3:8b"


class TestModelDownloadSizeGb:
    def test_known_models(self):
        assert models._model_download_size_gb("qwen3:8b") == 5
        assert models._model_download_size_gb("qwen3-coder:30b") == 18

    def test_unknown_model_returns_small_default(self):
        assert models._model_download_size_gb("unknown:latest") == 5


class TestPullWithProgress:
    @mock.patch("ollama.pull")
    def test_calls_ollama_pull(self, mock_pull):
        event = SimpleNamespace(total=100, completed=100)
        mock_pull.return_value = iter([event])
        models.pull_with_progress("test-model")
        mock_pull.assert_called_once_with("test-model", stream=True)

    @mock.patch("ollama.pull")
    def test_handles_zero_total(self, mock_pull):
        event = SimpleNamespace(total=0, completed=0)
        mock_pull.return_value = iter([event])
        models.pull_with_progress("test-model")


class TestEnsureChatModel:
    def _make_model(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(model=name)

    @mock.patch("ollama.list")
    def test_noop_when_chat_models_exist(self, mock_list):
        mock_list.return_value = SimpleNamespace(
            models=[self._make_model("llama3:latest"), self._make_model("nomic-embed-text:latest")]
        )
        models.ensure_chat_model()  # should not raise or pull

    @mock.patch("lilbee.settings.set_value")
    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch("ollama.list")
    def test_auto_pulls_when_no_chat_models(self, mock_list, _ram, _disk, mock_pull, mock_save):
        # Only embedding model installed
        mock_list.return_value = SimpleNamespace(
            models=[self._make_model("nomic-embed-text:latest")]
        )
        models.ensure_chat_model()
        mock_pull.assert_called_once_with("qwen3-coder:30b")
        mock_save.assert_called_once_with("chat_model", "qwen3-coder:30b")

    @mock.patch.object(models, "pull_with_progress")
    @mock.patch.object(models, "get_free_disk_gb", return_value=50.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=8.0)
    @mock.patch("ollama.list")
    def test_low_ram_picks_small_model(self, mock_list, _ram, _disk, mock_pull):
        mock_list.return_value = SimpleNamespace(models=[])
        models.ensure_chat_model()
        mock_pull.assert_called_once_with("qwen3:8b")

    @mock.patch.object(models, "get_free_disk_gb", return_value=3.0)
    @mock.patch.object(models, "get_system_ram_gb", return_value=32.0)
    @mock.patch("ollama.list")
    def test_insufficient_disk_raises(self, mock_list, _ram, _disk):
        mock_list.return_value = SimpleNamespace(models=[])
        with pytest.raises(RuntimeError, match="Not enough disk space"):
            models.ensure_chat_model()

    @mock.patch("ollama.list", side_effect=ConnectionError("refused"))
    def test_connection_error_raises(self, _):
        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            models.ensure_chat_model()

    @mock.patch("ollama.list")
    def test_empty_model_list_triggers_pull(self, mock_list):
        mock_list.return_value = SimpleNamespace(models=[])
        with (
            mock.patch.object(models, "get_system_ram_gb", return_value=16.0),
            mock.patch.object(models, "get_free_disk_gb", return_value=50.0),
            mock.patch.object(models, "pull_with_progress"),
        ):
            models.ensure_chat_model()

    @mock.patch("ollama.list")
    def test_only_embedding_model_triggers_pull(self, mock_list):
        mock_list.return_value = SimpleNamespace(
            models=[self._make_model("nomic-embed-text:latest")]
        )
        with (
            mock.patch.object(models, "get_system_ram_gb", return_value=16.0),
            mock.patch.object(models, "get_free_disk_gb", return_value=50.0),
            mock.patch.object(models, "pull_with_progress"),
        ):
            models.ensure_chat_model()
