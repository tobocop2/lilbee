"""Tests for model_manager.py — model lifecycle management across sources."""

from pathlib import Path
from unittest import mock

import httpx
import pytest

from lilbee.model_manager import (
    ModelManager,
    ModelSource,
    get_model_manager,
    reset_model_manager,
)


class TestModelSource:
    def test_native_value(self) -> None:
        assert ModelSource.NATIVE.value == "native"

    def test_ollama_value(self) -> None:
        assert ModelSource.OLLAMA.value == "ollama"

    def test_members(self) -> None:
        assert set(ModelSource) == {ModelSource.NATIVE, ModelSource.OLLAMA}


class TestModelManagerListInstalled:
    def test_native_lists_gguf_files(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "llama3-8b.gguf").touch()
        (models_dir / "mistral-7b.gguf").touch()
        (models_dir / "notes.txt").write_text("not a model")

        mgr = ModelManager(models_dir, "http://localhost:11434")
        result = mgr.list_installed(ModelSource.NATIVE)

        assert set(result) == {"llama3-8b.gguf", "mistral-7b.gguf"}

    def test_native_empty_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.list_installed(ModelSource.NATIVE) == []

    def test_native_missing_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "nonexistent"

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.list_installed(ModelSource.NATIVE) == []

    def test_ollama_lists_models(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest", "size": 4661211808},
                {"name": "nomic-embed-text:latest", "size": 274302448},
            ]
        }
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response) as mock_get:
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.list_installed(ModelSource.OLLAMA)

        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=30.0)
        assert set(result) == {"llama3:latest", "nomic-embed-text:latest"}

    def test_ollama_connection_error(self) -> None:
        with mock.patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.list_installed(ModelSource.OLLAMA)

        assert result == []

    def test_ollama_empty_response(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.list_installed(ModelSource.OLLAMA)

        assert result == []

    def test_none_source_lists_both(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "native-model.gguf").touch()

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "ollama-model:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir, "http://localhost:11434")
            result = mgr.list_installed(None)

        assert set(result) == {"native-model.gguf", "ollama-model:latest"}

    def test_none_source_deduplicates(self, tmp_path: Path) -> None:
        """If same model appears in both sources, it should appear once."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "llama3:latest.gguf").touch()

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir, "http://localhost:11434")
            result = mgr.list_installed(None)

        assert result.count("llama3:latest") == 1


class TestModelManagerIsInstalled:
    def test_native_installed(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "llama3-8b.gguf").touch()

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.is_installed("llama3-8b.gguf", ModelSource.NATIVE) is True

    def test_native_not_installed(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.is_installed("missing.gguf", ModelSource.NATIVE) is False

    def test_ollama_installed(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.is_installed("llama3:latest", ModelSource.OLLAMA)

        assert result is True

    def test_ollama_not_installed(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.is_installed("missing:latest", ModelSource.OLLAMA)

        assert result is False

    def test_none_source_checks_both(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "native-model.gguf").touch()

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir, "http://localhost:11434")
            assert mgr.is_installed("native-model.gguf", None) is True
            assert mgr.is_installed("ollama-model:latest", None) is False


class TestModelManagerGetSource:
    def test_native_model(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "my-model.gguf").touch()

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.get_source("my-model.gguf") == ModelSource.NATIVE

    def test_ollama_model(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.get_source("llama3:latest")

        assert result == ModelSource.OLLAMA

    def test_native_takes_precedence(self, tmp_path: Path) -> None:
        """When model exists in both sources, NATIVE takes precedence."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "shared:latest.gguf").touch()

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "shared:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir, "http://localhost:11434")
            result = mgr.get_source("shared:latest.gguf")

        assert result == ModelSource.NATIVE

    def test_not_found_returns_none(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir, "http://localhost:11434")
            result = mgr.get_source("nonexistent.gguf")

        assert result is None


class TestModelManagerPull:
    def test_native_delegates_to_catalog(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        fake_entry = mock.Mock()
        fake_entry.name = "test-model"

        def fake_download(entry: object) -> Path:
            path = models_dir / f"{entry.name}.gguf"
            path.write_text("fake model")
            return path

        mgr = ModelManager(models_dir, "http://localhost:11434")
        with (
            mock.patch("lilbee.catalog.find_catalog_entry", return_value=fake_entry) as mock_find,
            mock.patch("lilbee.catalog.download_model", side_effect=fake_download) as mock_dl,
        ):
            result = mgr.pull("test-model", ModelSource.NATIVE)

        mock_find.assert_called_once_with("test-model")
        mock_dl.assert_called_once_with(fake_entry)
        assert result is not None
        assert result.name == "test-model.gguf"

    def test_native_pull_not_in_catalog(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir, "http://localhost:11434")
        with (
            mock.patch("lilbee.catalog.find_catalog_entry", return_value=None),
            pytest.raises(RuntimeError, match="not found in catalog"),
        ):
            mgr.pull("nonexistent-model", ModelSource.NATIVE)

    def test_ollama_pull_success(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        events = [
            {"status": "pulling manifest"},
            {"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 50},
            {"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 100},
            {"status": "success"},
        ]

        mock_response = mock.Mock()
        mock_response.iter_lines.return_value = iter([__import__("json").dumps(e) for e in events])
        mock_response.raise_for_status = mock.Mock()
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=False)

        mock_client = mock.Mock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)

        progress_calls: list[dict] = []

        def on_progress(data: dict) -> None:
            progress_calls.append(data)

        mgr = ModelManager(models_dir, "http://localhost:11434")
        with mock.patch("httpx.Client", return_value=mock_client):
            result = mgr.pull("llama3:latest", ModelSource.OLLAMA, on_progress=on_progress)

        mock_client.stream.assert_called_once()
        call_args = mock_client.stream.call_args
        assert call_args[0] == ("POST", "http://localhost:11434/api/pull")
        assert call_args[1]["json"] == {"name": "llama3:latest", "stream": True}

        assert result is None  # Ollama pull doesn't return a path
        assert len(progress_calls) > 0

    def test_ollama_pull_error(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mock_response = mock.Mock()
        mock_response.iter_lines.return_value = iter(['{"error": "model not found"}'])
        mock_response.raise_for_status = mock.Mock()
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=False)

        mock_client = mock.Mock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)

        mgr = ModelManager(models_dir, "http://localhost:11434")
        with (
            mock.patch("httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="model not found"),
        ):
            mgr.pull("nonexistent:model", ModelSource.OLLAMA)

    def test_ollama_connection_error_during_pull(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mock_client = mock.Mock()
        mock_client.stream.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)

        mgr = ModelManager(models_dir, "http://localhost:11434")
        with (
            mock.patch("httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="Cannot connect to Ollama"),
        ):
            mgr.pull("llama3:latest", ModelSource.OLLAMA)

    def test_ollama_pull_without_progress_callback(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        events = [
            {"status": "pulling manifest"},
            {"status": "success"},
        ]

        mock_response = mock.Mock()
        mock_response.iter_lines.return_value = iter([__import__("json").dumps(e) for e in events])
        mock_response.raise_for_status = mock.Mock()
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=False)

        mock_client = mock.Mock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)

        mgr = ModelManager(models_dir, "http://localhost:11434")
        with mock.patch("httpx.Client", return_value=mock_client):
            result = mgr.pull("llama3:latest", ModelSource.OLLAMA)

        assert result is None

    def test_ollama_pull_skips_empty_lines(self, tmp_path: Path) -> None:
        """Empty strings from iter_lines are skipped."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mock_response = mock.Mock()
        mock_response.iter_lines.return_value = iter(["", '{"status": "success"}'])
        mock_response.raise_for_status = mock.Mock()
        mock_response.__enter__ = mock.Mock(return_value=mock_response)
        mock_response.__exit__ = mock.Mock(return_value=False)

        mock_client = mock.Mock()
        mock_client.stream.return_value = mock_response
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)

        mgr = ModelManager(models_dir, "http://localhost:11434")
        with mock.patch("httpx.Client", return_value=mock_client):
            result = mgr.pull("llama3:latest", ModelSource.OLLAMA)

        assert result is None


class TestModelManagerRemove:
    def test_native_removes_file(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "llama3-8b.gguf"
        model_file.write_text("fake model data")

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.remove("llama3-8b.gguf", ModelSource.NATIVE) is True
        assert not model_file.exists()

    def test_native_remove_nonexistent(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.remove("missing.gguf", ModelSource.NATIVE) is False

    def test_ollama_remove_success(self) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 200

        with mock.patch("httpx.request", return_value=mock_response) as mock_req:
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.remove("llama3:latest", ModelSource.OLLAMA)

        mock_req.assert_called_once()
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs["content"] == b'{"model": "llama3:latest"}'
        assert call_kwargs["headers"]["Content-Type"] == "application/json"
        assert result is True

    def test_ollama_remove_not_found(self) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 404

        with mock.patch("httpx.request", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.remove("nonexistent:latest", ModelSource.OLLAMA)

        assert result is False

    def test_ollama_connection_error_during_remove(self) -> None:
        with mock.patch("httpx.request", side_effect=httpx.ConnectError("Connection refused")):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
                mgr.remove("llama3:latest", ModelSource.OLLAMA)

    def test_ollama_remove_unexpected_status(self) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 500

        with mock.patch("httpx.request", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.remove("llama3:latest", ModelSource.OLLAMA)

        assert result is False

    def test_none_source_removes_from_all(self, tmp_path: Path) -> None:
        """source=None tries native first, then ollama."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "my-model.gguf"
        model_file.write_text("fake")

        mock_response = mock.Mock()
        mock_response.status_code = 200

        with mock.patch("httpx.request", return_value=mock_response):
            mgr = ModelManager(models_dir, "http://localhost:11434")
            result = mgr.remove("my-model.gguf", None)

        assert result is True
        assert not model_file.exists()


class TestSingleton:
    def setup_method(self) -> None:
        reset_model_manager()

    def teardown_method(self) -> None:
        reset_model_manager()

    def test_creates_singleton(self) -> None:
        with mock.patch("lilbee.config.cfg") as mock_cfg:
            mock_cfg.models_dir = Path("/tmp/models")
            mock_cfg.llm_base_url = "http://localhost:11434"
            mgr = get_model_manager()
            assert isinstance(mgr, ModelManager)

    def test_returns_same_instance(self) -> None:
        with mock.patch("lilbee.config.cfg") as mock_cfg:
            mock_cfg.models_dir = Path("/tmp/models")
            mock_cfg.llm_base_url = "http://localhost:11434"
            mgr1 = get_model_manager()
            mgr2 = get_model_manager()
            assert mgr1 is mgr2

    def test_reset_creates_new_instance(self) -> None:
        with mock.patch("lilbee.config.cfg") as mock_cfg:
            mock_cfg.models_dir = Path("/tmp/models")
            mock_cfg.llm_base_url = "http://localhost:11434"
            mgr1 = get_model_manager()
            reset_model_manager()
            mgr2 = get_model_manager()
            assert mgr1 is not mgr2


class TestOllamaEdgeCases:
    def test_ollama_http_error(self, tmp_path: Path) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=mock.Mock(), response=mock_response
        )

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir=tmp_path, ollama_base_url="http://localhost:11434")
            result = mgr.list_installed(ModelSource.OLLAMA)

        assert result == []

    def test_ollama_timeout(self, tmp_path: Path) -> None:
        with mock.patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
            mgr = ModelManager(models_dir=tmp_path, ollama_base_url="http://localhost:11434")
            result = mgr.list_installed(ModelSource.OLLAMA)

        assert result == []
