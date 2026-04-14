"""Tests for model_manager.py — model lifecycle management across sources."""

from pathlib import Path
from unittest import mock

import httpx
import pytest

from lilbee.model_manager import (
    ModelManager,
    ModelSource,
    ModelTask,
    RemoteModel,
    _has_provider_key,
    detect_provider,
    discover_api_models,
    get_model_manager,
    reset_model_manager,
)


class TestModelSource:
    def test_native_value(self) -> None:
        assert ModelSource.NATIVE.value == "native"

    def test_litellm_value(self) -> None:
        assert ModelSource.LITELLM.value == "litellm"

    def test_members(self) -> None:
        assert set(ModelSource) == {ModelSource.NATIVE, ModelSource.LITELLM}

    def test_parse_none_and_empty_return_none(self) -> None:
        assert ModelSource.parse(None) is None
        assert ModelSource.parse("") is None

    def test_parse_valid_values(self) -> None:
        assert ModelSource.parse("native") is ModelSource.NATIVE
        assert ModelSource.parse("litellm") is ModelSource.LITELLM

    def test_parse_invalid_raises_value_error(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="invalid source 'bogus'"):
            ModelSource.parse("bogus")


def _install_registry_model(
    models_dir: Path,
    tmp_path: Path,
    name: str,
    data: bytes,
    repo: str = "org/repo",
) -> None:
    """Helper: create HF cache blob and install a model into the registry."""
    import hashlib

    from lilbee.registry import ModelManifest, ModelRef, ModelRegistry

    digest = hashlib.sha256(data).hexdigest()
    cache_dir = models_dir / f"models--{repo.replace('/', '--')}"
    blob_dir = cache_dir / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / digest).write_bytes(data)

    source = tmp_path / f"{name}.gguf"
    source.write_bytes(data)

    registry = ModelRegistry(models_dir)
    ref = ModelRef(name=name)
    manifest = ModelManifest(
        name=name,
        tag="latest",
        size_bytes=len(data),
        task="chat",
        source_repo=repo,
        source_filename=f"{name}.gguf",
        downloaded_at="2026-01-01T00:00:00+00:00",
    )
    registry.install(ref, source, manifest)


class TestModelManagerListInstalled:
    def test_native_lists_registered_models(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _install_registry_model(models_dir, tmp_path, "llama3-8b", b"llama3-data")
        _install_registry_model(models_dir, tmp_path, "mistral-7b", b"mistral-data")

        mgr = ModelManager(models_dir, "http://localhost:11434")
        result = mgr.list_installed(ModelSource.NATIVE)

        assert set(result) == {"llama3-8b:latest", "mistral-7b:latest"}

    def test_native_empty_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.list_installed(ModelSource.NATIVE) == []

    def test_native_missing_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "nonexistent"

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.list_installed(ModelSource.NATIVE) == []

    def test_litellm_lists_models(self) -> None:
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
            result = mgr.list_installed(ModelSource.LITELLM)

        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=30.0)
        assert set(result) == {"llama3:latest", "nomic-embed-text:latest"}

    def test_litellm_connection_error(self) -> None:
        with mock.patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.list_installed(ModelSource.LITELLM)

        assert result == []

    def test_litellm_empty_response(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.list_installed(ModelSource.LITELLM)

        assert result == []

    def test_none_source_lists_both(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _install_registry_model(models_dir, tmp_path, "native-model", b"native-data")

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "remote-model:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir, "http://localhost:11434")
            result = mgr.list_installed(None)

        assert set(result) == {"native-model:latest", "remote-model:latest"}

    def test_none_source_deduplicates(self, tmp_path: Path) -> None:
        """If same model appears in both sources, it should appear once."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _install_registry_model(models_dir, tmp_path, "llama3", b"llama3-dedup-data")

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

    def test_litellm_installed(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.is_installed("llama3:latest", ModelSource.LITELLM)

        assert result is True

    def test_litellm_not_installed(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.is_installed("missing:latest", ModelSource.LITELLM)

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
            assert mgr.is_installed("remote-model:latest", None) is False


class TestModelManagerGetSource:
    def test_native_model(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "my-model.gguf").touch()

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.get_source("my-model.gguf") == ModelSource.NATIVE

    def test_litellm_model(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.get_source("llama3:latest")

        assert result == ModelSource.LITELLM

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

        def fake_download(entry: object, *, on_progress: object = None) -> Path:
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
        mock_dl.assert_called_once_with(fake_entry, on_progress=None)
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

    def test_litellm_pull_success(self, tmp_path: Path) -> None:
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
            result = mgr.pull("llama3:latest", ModelSource.LITELLM, on_progress=on_progress)

        mock_client.stream.assert_called_once()
        call_args = mock_client.stream.call_args
        assert call_args[0] == ("POST", "http://localhost:11434/api/pull")
        assert call_args[1]["json"] == {"name": "llama3:latest", "stream": True}

        assert result is None  # litellm pull doesn't return a path
        assert len(progress_calls) > 0

    def test_litellm_pull_error(self, tmp_path: Path) -> None:
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
            mgr.pull("nonexistent:model", ModelSource.LITELLM)

    def test_litellm_connection_error_during_pull(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mock_client = mock.Mock()
        mock_client.stream.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__enter__ = mock.Mock(return_value=mock_client)
        mock_client.__exit__ = mock.Mock(return_value=False)

        mgr = ModelManager(models_dir, "http://localhost:11434")
        with (
            mock.patch("httpx.Client", return_value=mock_client),
            pytest.raises(RuntimeError, match="Cannot connect to litellm backend"),
        ):
            mgr.pull("llama3:latest", ModelSource.LITELLM)

    def test_litellm_pull_without_progress_callback(self, tmp_path: Path) -> None:
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
            result = mgr.pull("llama3:latest", ModelSource.LITELLM)

        assert result is None

    def test_litellm_pull_skips_empty_lines(self, tmp_path: Path) -> None:
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
            result = mgr.pull("llama3:latest", ModelSource.LITELLM)

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

    def test_native_remove_path_traversal_blocked(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr.remove("../../etc/passwd", ModelSource.NATIVE) is False

    def test_litellm_remove_success(self) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 200

        with mock.patch("httpx.request", return_value=mock_response) as mock_req:
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.remove("llama3:latest", ModelSource.LITELLM)

        mock_req.assert_called_once()
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs["content"] == b'{"model": "llama3:latest"}'
        assert call_kwargs["headers"]["Content-Type"] == "application/json"
        assert result is True

    def test_litellm_remove_not_found(self) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 404

        with mock.patch("httpx.request", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.remove("nonexistent:latest", ModelSource.LITELLM)

        assert result is False

    def test_litellm_connection_error_during_remove(self) -> None:
        with mock.patch("httpx.request", side_effect=httpx.ConnectError("Connection refused")):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            with pytest.raises(RuntimeError, match="Cannot connect to litellm backend"):
                mgr.remove("llama3:latest", ModelSource.LITELLM)

    def test_litellm_remove_unexpected_status(self) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 500

        with mock.patch("httpx.request", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"), "http://localhost:11434")
            result = mgr.remove("llama3:latest", ModelSource.LITELLM)

        assert result is False

    def test_none_source_removes_from_all(self, tmp_path: Path) -> None:
        """source=None tries native first, then litellm."""
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

    def test_creates_singleton(self, tmp_path: Path) -> None:
        from lilbee.config import cfg

        cfg.models_dir = tmp_path / "models"
        cfg.litellm_base_url = "http://localhost:11434"
        mgr = get_model_manager()
        assert isinstance(mgr, ModelManager)
        assert mgr._models_dir == tmp_path / "models"
        assert mgr._litellm_base_url == "http://localhost:11434"

    def test_returns_same_instance(self, tmp_path: Path) -> None:
        from lilbee.config import cfg

        cfg.models_dir = tmp_path / "models"
        cfg.litellm_base_url = "http://localhost:11434"
        mgr1 = get_model_manager()
        mgr2 = get_model_manager()
        assert mgr1 is mgr2

    def test_reset_creates_new_instance(self, tmp_path: Path) -> None:
        from lilbee.config import cfg

        cfg.models_dir = tmp_path / "models"
        cfg.litellm_base_url = "http://localhost:11434"
        mgr1 = get_model_manager()
        reset_model_manager()
        mgr2 = get_model_manager()
        assert mgr1 is not mgr2


class TestLitellmEdgeCases:
    def test_litellm_http_error(self, tmp_path: Path) -> None:
        mock_response = mock.Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=mock.Mock(), response=mock_response
        )

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir=tmp_path, litellm_base_url="http://localhost:11434")
            result = mgr.list_installed(ModelSource.LITELLM)

        assert result == []

    def test_litellm_timeout(self, tmp_path: Path) -> None:
        with mock.patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
            mgr = ModelManager(models_dir=tmp_path, litellm_base_url="http://localhost:11434")
            result = mgr.list_installed(ModelSource.LITELLM)

        assert result == []


class TestIsNativePathTraversal:
    def test_path_traversal_returns_false(self, tmp_path: Path) -> None:
        """_is_native returns False for path traversal attempts."""
        mgr = ModelManager(models_dir=tmp_path, litellm_base_url="http://localhost:11434")
        assert not mgr._is_native("../../etc/passwd")


class TestIsNativeRegistry:
    def test_is_native_true_when_in_registry(self, tmp_path: Path) -> None:
        """_is_native returns True when model exists in registry."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _install_registry_model(models_dir, tmp_path, "my-reg-model", b"registry-model-data")

        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr._is_native("my-reg-model") is True


class TestRemoveNativeRegistry:
    def test_remove_native_from_registry(self, tmp_path: Path) -> None:
        """_remove_native removes model from registry."""
        from lilbee.registry import ModelRegistry

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _install_registry_model(models_dir, tmp_path, "removable", b"registry-model-data")

        registry = ModelRegistry(models_dir)
        mgr = ModelManager(models_dir, "http://localhost:11434")
        assert mgr._remove_native("removable") is True
        assert not registry.is_installed("removable")


class TestDetectProvider:
    def test_localhost_ollama(self) -> None:
        assert detect_provider("http://localhost:11434") == "Ollama"

    def test_ollama_in_url(self) -> None:
        assert detect_provider("http://ollama.local:11434") == "Ollama"

    def test_openai_url(self) -> None:
        assert detect_provider("https://api.openai.com/v1") == "OpenAI"

    def test_anthropic_url(self) -> None:
        assert detect_provider("https://api.anthropic.com") == "Anthropic"

    def test_unknown_url(self) -> None:
        assert detect_provider("http://192.168.1.100:8080") == "Remote"

    def test_case_insensitive(self) -> None:
        assert detect_provider("http://LOCALHOST:11434") == "Ollama"


class TestRemoteModelProvider:
    def test_classify_remote_models_sets_provider(self) -> None:
        from lilbee.model_manager import classify_remote_models

        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest", "details": {"family": "llama", "parameter_size": "8B"}}
            ]
        }
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            result = classify_remote_models("http://localhost:11434")

        assert len(result) == 1
        assert result[0].provider == "Ollama"

    def test_classify_remote_models_openai_provider(self) -> None:
        from lilbee.model_manager import classify_remote_models

        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "models": [{"name": "gpt-4", "details": {"family": "gpt", "parameter_size": ""}}]
        }
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            result = classify_remote_models("https://api.openai.com/v1")

        assert len(result) == 1
        assert result[0].provider == "OpenAI"

    def test_remote_model_default_provider(self) -> None:
        model = RemoteModel(name="test", task="chat", family="llama", parameter_size="8B")
        assert model.provider == "Remote"


class TestIsVisionCapable:
    """Tests for is_vision_capable() — 4-tier detection."""

    def setup_method(self) -> None:
        from lilbee.model_manager import reset_vision_cache

        reset_vision_cache()

    def teardown_method(self) -> None:
        from lilbee.model_manager import reset_vision_cache

        reset_vision_cache()

    def test_empty_model_returns_false(self) -> None:
        from lilbee.model_manager import is_vision_capable

        assert is_vision_capable("") is False

    def test_provider_capabilities_vision(self) -> None:
        """Tier 1: provider reports vision capability."""
        from lilbee.model_manager import is_vision_capable

        mock_provider = mock.MagicMock()
        mock_provider.get_capabilities.return_value = ["completion", "vision"]
        mock_services = mock.MagicMock()
        mock_services.provider = mock_provider

        with mock.patch("lilbee.services.get_services", return_value=mock_services):
            assert is_vision_capable("llava:7b") is True

    def test_provider_capabilities_no_vision(self) -> None:
        """Tier 1: provider reports no vision, falls through to catalog/name."""
        from lilbee.model_manager import is_vision_capable

        mock_provider = mock.MagicMock()
        mock_provider.get_capabilities.return_value = ["completion"]
        mock_services = mock.MagicMock()
        mock_services.provider = mock_provider

        with mock.patch("lilbee.services.get_services", return_value=mock_services):
            # "qwen3:8b" has no vision name pattern and isn't in catalog
            assert is_vision_capable("qwen3:8b") is False

    def test_provider_error_falls_through(self) -> None:
        """Tier 1 failure falls through to tier 2 (catalog)."""
        from lilbee.model_manager import is_vision_capable

        mock_provider = mock.MagicMock()
        mock_provider.get_capabilities.side_effect = RuntimeError("no backend")
        mock_services = mock.MagicMock()
        mock_services.provider = mock_provider

        # "llava:7b" matches name pattern even without provider
        with mock.patch("lilbee.services.get_services", return_value=mock_services):
            assert is_vision_capable("llava:7b") is True

    def test_catalog_match(self) -> None:
        """Tier 2: model matches a FEATURED_VISION entry."""
        from lilbee.model_manager import is_vision_capable

        mock_entry = mock.MagicMock()
        mock_entry.name = "lightonocr"
        mock_entry.hf_repo = "noctrex/LightOnOCR-2-1B-GGUF"

        mock_provider = mock.MagicMock()
        mock_provider.get_capabilities.return_value = []
        mock_services = mock.MagicMock()
        mock_services.provider = mock_provider

        with (
            mock.patch("lilbee.services.get_services", return_value=mock_services),
            mock.patch("lilbee.catalog.FEATURED_VISION", (mock_entry,)),
        ):
            assert is_vision_capable("lightonocr") is True

    def test_name_pattern_fallback(self) -> None:
        """Tier 3: model name contains a known vision keyword."""
        from lilbee.model_manager import is_vision_capable

        mock_provider = mock.MagicMock()
        mock_provider.get_capabilities.return_value = []
        mock_services = mock.MagicMock()
        mock_services.provider = mock_provider

        with mock.patch("lilbee.services.get_services", return_value=mock_services):
            assert is_vision_capable("moondream:1.8b") is True
            # Reset cache between checks
            from lilbee.model_manager import reset_vision_cache

            reset_vision_cache()
            assert is_vision_capable("minicpm-v:8b") is True

    def test_no_match_returns_false(self) -> None:
        """Model matches no tier — returns False."""
        from lilbee.model_manager import is_vision_capable

        mock_provider = mock.MagicMock()
        mock_provider.get_capabilities.return_value = []
        mock_services = mock.MagicMock()
        mock_services.provider = mock_provider

        with mock.patch("lilbee.services.get_services", return_value=mock_services):
            assert is_vision_capable("mistral:7b") is False

    def test_result_is_cached(self) -> None:
        """Second call uses cache, not provider."""
        from lilbee.model_manager import is_vision_capable

        mock_provider = mock.MagicMock()
        mock_provider.get_capabilities.return_value = ["vision"]
        mock_services = mock.MagicMock()
        mock_services.provider = mock_provider

        with mock.patch("lilbee.services.get_services", return_value=mock_services):
            assert is_vision_capable("llava:7b") is True
            assert is_vision_capable("llava:7b") is True
            # Provider called only once due to cache
            mock_provider.get_capabilities.assert_called_once()

    def test_reset_clears_cache(self) -> None:
        """reset_vision_cache allows re-detection."""
        from lilbee.model_manager import is_vision_capable, reset_vision_cache

        mock_provider = mock.MagicMock()
        mock_provider.get_capabilities.return_value = ["vision"]
        mock_services = mock.MagicMock()
        mock_services.provider = mock_provider

        with mock.patch("lilbee.services.get_services", return_value=mock_services):
            is_vision_capable("llava:7b")
            reset_vision_cache()
            is_vision_capable("llava:7b")
            assert mock_provider.get_capabilities.call_count == 2


class TestHasProviderKey:
    def test_env_var_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert _has_provider_key("openai", "openai_api_key", "OPENAI_API_KEY") is True

    def test_env_var_absent_config_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.config import cfg

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg.openai_api_key = "sk-from-config"
        assert _has_provider_key("openai", "openai_api_key", "OPENAI_API_KEY") is True

    def test_neither_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.config import cfg

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg.anthropic_api_key = ""
        assert _has_provider_key("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY") is False

    def test_unknown_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOME_KEY", raising=False)
        assert _has_provider_key("unknown", "nonexistent_field", "SOME_KEY") is False


class TestDiscoverApiModels:
    def test_returns_empty_when_litellm_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with mock.patch.dict("sys.modules", {"litellm": None}):
            result = discover_api_models()
        assert result == {}

    def test_returns_models_for_configured_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_litellm = mock.MagicMock()
        mock_litellm.models_by_provider = {
            "openai": {"gpt-4o", "gpt-4o-mini", "dall-e-3"},
        }
        mock_litellm.model_cost = {
            "gpt-4o": {"mode": "chat"},
            "gpt-4o-mini": {"mode": "chat"},
            "dall-e-3": {"mode": "image_generation"},
        }
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from lilbee.config import cfg

        cfg.anthropic_api_key = ""
        cfg.gemini_api_key = ""

        with mock.patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = discover_api_models()

        assert "OpenAI" in result
        names = [m.name for m in result["OpenAI"]]
        assert "gpt-4o" in names
        assert "gpt-4o-mini" in names
        assert "dall-e-3" not in names

    def test_skips_providers_without_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_litellm = mock.MagicMock()
        mock_litellm.models_by_provider = {
            "openai": {"gpt-4o"},
            "anthropic": {"claude-sonnet-4-6"},
        }
        mock_litellm.model_cost = {
            "gpt-4o": {"mode": "chat"},
            "claude-sonnet-4-6": {"mode": "chat"},
        }
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from lilbee.config import cfg

        cfg.openai_api_key = ""
        cfg.anthropic_api_key = ""
        cfg.gemini_api_key = ""

        with mock.patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = discover_api_models()

        assert result == {}

    def test_multiple_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_litellm = mock.MagicMock()
        mock_litellm.models_by_provider = {
            "openai": {"gpt-4o"},
            "anthropic": {"claude-sonnet-4-6"},
        }
        mock_litellm.model_cost = {
            "gpt-4o": {"mode": "chat"},
            "claude-sonnet-4-6": {"mode": "chat"},
        }
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from lilbee.config import cfg

        cfg.gemini_api_key = ""

        with mock.patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = discover_api_models()

        assert "OpenAI" in result
        assert "Anthropic" in result
        assert all(m.task == ModelTask.CHAT for models in result.values() for m in models)

    def test_remote_model_has_correct_provider_label(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_litellm = mock.MagicMock()
        mock_litellm.models_by_provider = {"anthropic": {"claude-sonnet-4-6"}}
        mock_litellm.model_cost = {"claude-sonnet-4-6": {"mode": "chat"}}
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from lilbee.config import cfg

        cfg.openai_api_key = ""
        cfg.gemini_api_key = ""

        with mock.patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = discover_api_models()

        model = result["Anthropic"][0]
        assert model.provider == "Anthropic"
        assert model.name == "claude-sonnet-4-6"
