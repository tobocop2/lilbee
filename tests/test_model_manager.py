"""Tests for model_manager.py: model lifecycle management across sources."""

from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import httpx
import pytest

from lilbee.catalog.types import ModelSource, ModelTask
from lilbee.modelhub.model_manager import (
    ModelManager,
    RemoteModel,
    discover_api_models,
)
from lilbee.modelhub.model_manager.discovery import _has_provider_key
from lilbee.providers.sdk_backend import detect_backend_name


class TestNativeIdentitiesCache:
    """``list_native_identities`` memoizes against ``list_installed`` errors
    and within the TTL window. Ensures both branches execute."""

    def test_returns_cached_within_ttl(self) -> None:
        from lilbee.modelhub.model_manager.core import ModelManager as MM

        mgr = MM(Path("/nonexistent"))
        fake_registry = mock.MagicMock()
        m = mock.MagicMock()
        m.ref = "test/m"
        m.hf_repo = "test/m"
        fake_registry.list_installed.return_value = [m]
        mgr._registry = fake_registry  # type: ignore[assignment]
        first = mgr.list_native_identities()
        # Second call within TTL must hit the cache, not call list_installed again.
        second = mgr.list_native_identities()
        assert first is second
        assert fake_registry.list_installed.call_count == 1

    def test_swallows_registry_error(self) -> None:
        from lilbee.modelhub.model_manager.core import ModelManager as MM

        mgr = MM(Path("/nonexistent"))
        fake_registry = mock.MagicMock()
        fake_registry.list_installed.side_effect = OSError("permission denied")
        mgr._registry = fake_registry  # type: ignore[assignment]
        result = mgr.list_native_identities()
        assert result == frozenset()


class TestModelSource:
    def test_native_value(self) -> None:
        assert ModelSource.NATIVE.value == "native"

    def test_backend_value(self) -> None:
        assert ModelSource.REMOTE.value == "remote"

    def test_members(self) -> None:
        assert set(ModelSource) == {
            ModelSource.NATIVE,
            ModelSource.REMOTE,
            ModelSource.FRONTIER,
            ModelSource.OLLAMA,
            ModelSource.LM_STUDIO,
        }

    def test_parse_none_and_empty_return_none(self) -> None:
        assert ModelSource.parse(None) is None
        assert ModelSource.parse("") is None

    def test_parse_valid_values(self) -> None:
        assert ModelSource.parse("native") is ModelSource.NATIVE
        assert ModelSource.parse("remote") is ModelSource.REMOTE

    def test_parse_invalid_raises_value_error(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="invalid source 'bogus'"):
            ModelSource.parse("bogus")

    def test_has_frontier_and_ollama(self) -> None:
        assert ModelSource.FRONTIER == "frontier"
        assert ModelSource.OLLAMA == "ollama"
        assert ModelSource.parse("frontier") is ModelSource.FRONTIER
        assert ModelSource.parse("ollama") is ModelSource.OLLAMA

    def test_local_server_keys_map_to_sources(self) -> None:
        """Each local server's routing key is a valid ModelSource value.

        ``get_source`` and the catalog derive a model's source via
        ``ModelSource(spec.key)``; this contract keeps that mapping total.
        """
        from lilbee.providers.local_servers import LOCAL_SERVERS

        for spec in LOCAL_SERVERS:
            assert isinstance(ModelSource(spec.key), ModelSource)
        assert ModelSource("lm_studio") is ModelSource.LM_STUDIO


def _install_registry_model(
    models_dir: Path,
    tmp_path: Path,
    filename: str,
    data: bytes,
    repo: str = "org/repo-GGUF",
) -> str:
    """Install a model into the registry; return the canonical ref string."""
    from lilbee.modelhub.registry import ModelManifest, ModelRegistry

    source = tmp_path / filename
    source.write_bytes(data)

    registry = ModelRegistry(models_dir)
    manifest = ModelManifest(
        hf_repo=repo,
        gguf_filename=filename,
        size_bytes=len(data),
        task="chat",
        downloaded_at="2026-01-01T00:00:00+00:00",
    )
    registry.install(repo, filename, source, manifest)
    return f"{repo}/{filename}"


class TestModelManagerListInstalled:
    def test_native_lists_registered_models(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        ref_a = _install_registry_model(
            models_dir, tmp_path, "llama3-8b.gguf", b"llama3-data", repo="org/llama3-8b-GGUF"
        )
        ref_b = _install_registry_model(
            models_dir, tmp_path, "mistral-7b.gguf", b"mistral-data", repo="org/mistral-7b-GGUF"
        )

        mgr = ModelManager(models_dir)
        result = mgr.list_installed(ModelSource.NATIVE)

        assert set(result) == {ref_a, ref_b}

    def test_native_empty_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir)
        assert mgr.list_installed(ModelSource.NATIVE) == []

    def test_native_missing_dir(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "nonexistent"

        mgr = ModelManager(models_dir)
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
            mgr = ModelManager(Path("/tmp"))
            result = mgr.list_installed(ModelSource.REMOTE)

        # REMOTE now spans every configured server; Ollama is probed via /api/tags.
        called_urls = [call.args[0] for call in mock_get.call_args_list]
        assert "http://localhost:11434/api/tags" in called_urls
        assert set(result) == {"llama3:latest", "nomic-embed-text:latest"}

    def test_litellm_connection_error(self) -> None:
        with mock.patch("httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            mgr = ModelManager(Path("/tmp"))
            result = mgr.list_installed(ModelSource.REMOTE)

        assert result == []

    def test_litellm_empty_response(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"))
            result = mgr.list_installed(ModelSource.REMOTE)

        assert result == []

    def test_none_source_lists_both(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        native_ref = _install_registry_model(
            models_dir, tmp_path, "native.gguf", b"native-data", repo="org/native-GGUF"
        )

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "remote-model:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir)
            result = mgr.list_installed(None)

        assert set(result) == {native_ref, "remote-model:latest"}

    def test_none_source_deduplicates(self, tmp_path: Path) -> None:
        """If the same ref appears in both sources, it should appear once."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        native_ref = _install_registry_model(
            models_dir, tmp_path, "shared.gguf", b"shared-data", repo="org/shared-GGUF"
        )

        # The remote backend reports the same ref string verbatim.
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": native_ref}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir)
            result = mgr.list_installed(None)

        assert result.count(native_ref) == 1

    def test_second_call_within_ttl_uses_cache(self) -> None:
        """Two back-to-back calls should hit the HTTP endpoint only once."""
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response) as mock_get:
            mgr = ModelManager(Path("/tmp"))
            mgr.list_installed(ModelSource.REMOTE)
            after_first = mock_get.call_count
            mgr.list_installed(ModelSource.REMOTE)

        # The second call is served from cache: no additional network calls,
        # regardless of how many servers the first fetch probed.
        assert mock_get.call_count == after_first

    def test_cache_expires_after_ttl(self) -> None:
        """After the TTL window elapses, list_installed refetches."""
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        from lilbee.modelhub.model_manager import core as mm_core

        with (
            mock.patch("httpx.get", return_value=mock_response) as mock_get,
            mock.patch.object(mm_core.time, "monotonic") as mock_clock,
        ):
            # One clock tick per list_installed call: second tick is past TTL.
            mock_clock.side_effect = [0.0, 100.0]
            mgr = ModelManager(Path("/tmp"))
            mgr.list_installed(ModelSource.REMOTE)
            after_first = mock_get.call_count
            mgr.list_installed(ModelSource.REMOTE)

        # Past the TTL the second call refetches, doubling the per-fetch calls.
        assert mock_get.call_count == 2 * after_first

    def test_pull_invalidates_cache(self, tmp_path: Path) -> None:
        """After pull(), the next list_installed must refetch."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _install_registry_model(
            models_dir, tmp_path, "before.gguf", b"before-data", repo="org/before-GGUF"
        )

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir)
            mgr.list_installed(ModelSource.NATIVE)  # populate cache
            assert mgr._installed_cache

            # Swap the pull implementation so we don't actually fetch.
            with mock.patch.object(mgr, "_pull_native", return_value=Path("/tmp/x")):
                mgr.pull("Qwen/Qwen3-0.6B-GGUF", ModelSource.NATIVE)

            assert mgr._installed_cache == {}

    def test_remove_invalidates_cache(self, tmp_path: Path) -> None:
        """After remove(), the next list_installed must refetch."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        ref = _install_registry_model(
            models_dir, tmp_path, "doomed.gguf", b"doomed-data", repo="org/doomed-GGUF"
        )

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir)
            mgr.list_installed(None)  # populate cache
            assert mgr._installed_cache

            mgr.remove(ref, ModelSource.NATIVE)
            assert mgr._installed_cache == {}


class TestModelManagerIsInstalled:
    def test_native_installed(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "llama3-8b.gguf").touch()

        mgr = ModelManager(models_dir)
        assert mgr.is_installed("llama3-8b.gguf", ModelSource.NATIVE) is True

    def test_native_not_installed(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir)
        assert mgr.is_installed("missing.gguf", ModelSource.NATIVE) is False

    def test_litellm_installed(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"))
            result = mgr.is_installed("llama3:latest", ModelSource.REMOTE)

        assert result is True

    def test_litellm_not_installed(self) -> None:
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"))
            result = mgr.is_installed("missing:latest", ModelSource.REMOTE)

        assert result is False

    def test_none_source_checks_both(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "native-model.gguf").touch()

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir)
            assert mgr.is_installed("native-model.gguf", None) is True
            assert mgr.is_installed("remote-model:latest", None) is False


class TestModelManagerGetSource:
    def test_native_model(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "my-model.gguf").touch()

        mgr = ModelManager(models_dir)
        assert mgr.get_source("my-model.gguf") == ModelSource.NATIVE

    def test_bare_ollama_model_is_remote_source(self) -> None:
        """A bare name a backend reports installed is generic REMOTE.

        Granular source comes only from a provider prefix; a bare name names
        no specific server, so it stays REMOTE.
        """
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"))
            result = mgr.get_source("llama3:latest")

        assert result == ModelSource.REMOTE

    def test_ollama_prefixed_ref_is_ollama_source_without_network(self) -> None:
        """An ``ollama/`` ref classifies on the prefix, no /api/tags call."""
        mgr = ModelManager(Path("/tmp"))
        with mock.patch("httpx.get") as mock_get:
            result = mgr.get_source("ollama/llama3:latest")
        assert result == ModelSource.OLLAMA
        mock_get.assert_not_called()

    def test_api_prefixed_ref_is_frontier_source(self) -> None:
        """A hosted API ref classifies as FRONTIER without a network call."""
        mgr = ModelManager(Path("/tmp"))
        with mock.patch("httpx.get") as mock_get:
            result = mgr.get_source("gemini/gemini-2.5-pro")
        assert result == ModelSource.FRONTIER
        mock_get.assert_not_called()

    def test_lm_studio_prefixed_ref_is_lm_studio_source_without_network(self) -> None:
        """An ``lm_studio/`` ref classifies on the prefix, no network call."""
        mgr = ModelManager(Path("/tmp"))
        with mock.patch("httpx.get") as mock_get:
            result = mgr.get_source("lm_studio/qwen2.5-coder")
        assert result == ModelSource.LM_STUDIO
        mock_get.assert_not_called()

    def test_bare_model_on_lm_studio_backend_is_remote_source(self) -> None:
        """A bare name an LM Studio backend reports installed stays generic REMOTE.

        Without a provider prefix the source is not specialized to a server.
        """
        mock_response = mock.Mock()
        mock_response.json.return_value = {"data": [{"id": "qwen2.5-coder"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"))
            result = mgr.get_source("qwen2.5-coder")

        assert result == ModelSource.REMOTE

    def test_bare_model_on_unknown_backend_is_remote_source(self) -> None:
        """A bare name a non-Ollama/LM-Studio backend reports stays generic REMOTE."""
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "custom-model"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"))
            result = mgr.get_source("custom-model")

        assert result == ModelSource.REMOTE

    def test_native_takes_precedence(self, tmp_path: Path) -> None:
        """When model exists in both sources, NATIVE takes precedence."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "shared:latest.gguf").touch()

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "shared:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir)
            result = mgr.get_source("shared:latest.gguf")

        assert result == ModelSource.NATIVE

    def test_not_found_returns_none(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(models_dir)
            result = mgr.get_source("nonexistent.gguf")

        assert result is None


class TestModelManagerPull:
    def test_native_delegates_to_catalog(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        fake_entry = mock.Mock()
        fake_entry.name = "test-model"

        def fake_download(
            entry: object, *, on_progress: object = None, on_complete: object = None
        ) -> Path:
            path = models_dir / f"{entry.name}.gguf"
            path.write_text("fake model")
            return path

        mgr = ModelManager(models_dir)
        with (
            mock.patch(
                "lilbee.catalog.resolve_pull_target", return_value=fake_entry
            ) as mock_resolve,
            mock.patch("lilbee.catalog.download_model", side_effect=fake_download) as mock_dl,
        ):
            result = mgr.pull("test-model", ModelSource.NATIVE)

        mock_resolve.assert_called_once_with("test-model")
        mock_dl.assert_called_once()
        call = mock_dl.call_args
        assert call.args == (fake_entry,)
        assert call.kwargs["on_progress"] is None
        assert callable(call.kwargs["on_complete"])
        assert result is not None
        assert result.name == "test-model.gguf"

    def test_native_pull_succeeds_for_arbitrary_hf_repo(self, tmp_path: Path) -> None:
        """Non-featured HF repos round-trip through an ad-hoc catalog entry."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        captured: list[object] = []

        def fake_download(
            entry: object, *, on_progress: object = None, on_complete: object = None
        ) -> Path:
            captured.append(entry)
            path = models_dir / "adhoc.gguf"
            path.write_text("fake model")
            return path

        mgr = ModelManager(models_dir)
        with mock.patch("lilbee.catalog.download_model", side_effect=fake_download):
            mgr.pull("bartowski/gemma-2-2b-it-GGUF", ModelSource.NATIVE)

        assert len(captured) == 1
        entry = captured[0]
        assert entry.hf_repo == "bartowski/gemma-2-2b-it-GGUF"
        assert entry.gguf_filename == "*.gguf"
        assert entry.featured is False

    def test_native_pull_unknown_short_name_raises(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir)
        with (
            mock.patch("lilbee.catalog.resolve_pull_target", return_value=None),
            pytest.raises(RuntimeError, match="HuggingFace repo id"),
        ):
            mgr.pull("nonexistent-model", ModelSource.NATIVE)

    def test_remote_pull_refused_naming_the_server(self, tmp_path: Path) -> None:
        """Local servers are read-only; a remote pull is refused without any HTTP call."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir)
        with (
            mock.patch("httpx.Client") as mock_client,
            pytest.raises(ValueError, match="Ollama"),
        ):
            mgr.pull("llama3:latest", ModelSource.OLLAMA)
        mock_client.assert_not_called()

    def test_remote_pull_refused_for_lm_studio(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir)
        with pytest.raises(ValueError, match="LM Studio"):
            mgr.pull("qwen2.5-7b-instruct", ModelSource.LM_STUDIO)


class TestModelManagerRemove:
    def test_native_removes_file(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "llama3-8b.gguf"
        model_file.write_text("fake model data")

        mgr = ModelManager(models_dir)
        removed = mgr.remove("llama3-8b.gguf", ModelSource.NATIVE)
        assert removed is True
        assert not model_file.exists()

    def test_native_remove_nonexistent(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir)
        removed = mgr.remove("missing.gguf", ModelSource.NATIVE)
        assert removed is False

    def test_native_remove_path_traversal_blocked(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        mgr = ModelManager(models_dir)
        removed = mgr.remove("../../etc/passwd", ModelSource.NATIVE)
        assert removed is False

    def test_remove_refuses_ollama_source(self) -> None:
        """Ollama is read-only: an explicit OLLAMA source is refused, never deleted."""
        with mock.patch("httpx.request") as mock_req:
            mgr = ModelManager(Path("/tmp"))
            with pytest.raises(ValueError, match="Ollama models but doesn't remove them"):
                mgr.remove("ollama/llama3:latest", ModelSource.OLLAMA)
        mock_req.assert_not_called()

    def test_remove_refuses_lm_studio_source(self) -> None:
        """LM Studio is read-only: an explicit LM_STUDIO source is refused."""
        with mock.patch("httpx.request") as mock_req:
            mgr = ModelManager(Path("/tmp"))
            with pytest.raises(ValueError, match="LM Studio models but doesn't remove them"):
                mgr.remove("lm_studio/qwen2.5-coder", ModelSource.LM_STUDIO)
        mock_req.assert_not_called()

    def test_remove_refuses_generic_remote_source(self) -> None:
        """A generic REMOTE source on an undetected backend is refused too."""
        mgr = ModelManager(Path("/tmp"))
        with pytest.raises(ValueError, match="doesn't remove them"):
            mgr.remove("custom-model", ModelSource.REMOTE)

    def test_remove_refuses_ollama_prefixed_ref_without_network(self) -> None:
        """An ``ollama/`` ref with source=None resolves on the prefix, no network call."""
        with mock.patch("httpx.get") as mock_get, mock.patch("httpx.request") as mock_req:
            mgr = ModelManager(Path("/tmp"))
            with pytest.raises(ValueError, match="doesn't remove them"):
                mgr.remove("ollama/llama3:latest")
        mock_get.assert_not_called()
        mock_req.assert_not_called()

    def test_remove_refuses_bare_backend_model_with_source_none(self) -> None:
        """A bare name the backend reports installed resolves to a read-only source."""
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"))
            with pytest.raises(ValueError, match="doesn't remove them"):
                mgr.remove("llama3:latest")

    def test_none_source_removes_native(self, tmp_path: Path) -> None:
        """source=None removes a native model without touching any backend."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "my-model.gguf"
        model_file.write_text("fake")

        with mock.patch("httpx.request") as mock_req:
            mgr = ModelManager(models_dir)
            result = mgr.remove("my-model.gguf", None)

        assert result is True
        assert not model_file.exists()
        mock_req.assert_not_called()

    def test_remove_unknown_model_source_none_returns_false(self) -> None:
        """source=None on a model in no known source is a no-op, not a refusal."""
        mock_response = mock.Mock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            mgr = ModelManager(Path("/tmp"))
            result = mgr.remove("nonexistent.gguf")

        assert result is False


class TestServicesIntegration:
    """``ModelManager`` lifecycle inside the ``Services`` container."""

    def setup_method(self) -> None:
        from lilbee.app.services import reset_services

        reset_services()

    def teardown_method(self) -> None:
        from lilbee.app.services import reset_services

        reset_services()

    def test_services_holds_model_manager(self, tmp_path: Path) -> None:
        from lilbee.app.services import get_services
        from lilbee.core.config import cfg

        cfg.models_dir = tmp_path / "models"
        mgr = get_services().model_manager
        assert isinstance(mgr, ModelManager)
        assert mgr._models_dir == tmp_path / "models"

    def test_services_returns_same_model_manager(self, tmp_path: Path) -> None:
        from lilbee.app.services import get_services
        from lilbee.core.config import cfg

        cfg.models_dir = tmp_path / "models"
        mgr1 = get_services().model_manager
        mgr2 = get_services().model_manager
        assert mgr1 is mgr2

    def test_reset_services_creates_new_model_manager(self, tmp_path: Path) -> None:
        from lilbee.app.services import get_services, reset_services
        from lilbee.core.config import cfg

        cfg.models_dir = tmp_path / "models"
        mgr1 = get_services().model_manager
        reset_services()
        mgr2 = get_services().model_manager
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
            mgr = ModelManager(models_dir=tmp_path)
            result = mgr.list_installed(ModelSource.REMOTE)

        assert result == []

    def test_litellm_timeout(self, tmp_path: Path) -> None:
        with mock.patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
            mgr = ModelManager(models_dir=tmp_path)
            result = mgr.list_installed(ModelSource.REMOTE)

        assert result == []


class TestIsNativePathTraversal:
    def test_path_traversal_returns_false(self, tmp_path: Path) -> None:
        """_is_native returns False for path traversal attempts."""
        mgr = ModelManager(models_dir=tmp_path)
        assert not mgr._is_native("../../etc/passwd")


class TestIsNativeRegistry:
    def test_is_native_true_when_in_registry(self, tmp_path: Path) -> None:
        """_is_native returns True when the ref points at an installed manifest."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        ref = _install_registry_model(
            models_dir, tmp_path, "my-reg.gguf", b"data", repo="org/my-reg-GGUF"
        )

        mgr = ModelManager(models_dir)
        assert mgr._is_native(ref) is True


class TestRemoveNativeRegistry:
    def test_remove_native_from_registry(self, tmp_path: Path) -> None:
        """_remove_native removes the manifest from the registry."""
        from lilbee.modelhub.registry import ModelRegistry

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        ref = _install_registry_model(
            models_dir, tmp_path, "removable.gguf", b"data", repo="org/removable-GGUF"
        )

        registry = ModelRegistry(models_dir)
        mgr = ModelManager(models_dir)
        assert mgr._remove_native(ref) is True
        assert not registry.is_installed(ref)


class TestDetectProvider:
    def test_localhost_ollama(self) -> None:
        assert detect_backend_name("http://localhost:11434") == "Ollama"

    def test_ollama_in_url(self) -> None:
        assert detect_backend_name("http://ollama.local:11434") == "Ollama"

    def test_openai_url(self) -> None:
        assert detect_backend_name("https://api.openai.com/v1") == "OpenAI"

    def test_anthropic_url(self) -> None:
        assert detect_backend_name("https://api.anthropic.com") == "Anthropic"

    def test_gemini_url(self) -> None:
        assert detect_backend_name("https://generativelanguage.googleapis.com") == "Gemini"

    def test_gemini_substring_fallback(self) -> None:
        """Non-canonical URLs with 'gemini' in the path also match."""
        assert detect_backend_name("https://proxy.example.com/gemini/v1") == "Gemini"

    def test_unknown_url(self) -> None:
        assert detect_backend_name("http://192.168.1.100:8080") == "Remote"

    def test_case_insensitive(self) -> None:
        assert detect_backend_name("http://LOCALHOST:11434") == "Ollama"


class TestClassifyRemoteTask:
    def test_bge_reranker_classified_as_rerank(self) -> None:
        """bge-reranker-* classifies as rerank despite bge being in _EMBEDDING_FAMILIES."""
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.model_manager.discovery import _classify_remote_task

        assert _classify_remote_task("bge-reranker-base", "bge") == ModelTask.RERANK
        assert _classify_remote_task("bge-reranker-large:latest", "bge") == ModelTask.RERANK

    def test_bge_m3_classified_as_embedding(self) -> None:
        """Regular bge embedding models still classify as EMBEDDING."""
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.model_manager.discovery import _classify_remote_task

        assert _classify_remote_task("bge-m3:latest", "bge") == ModelTask.EMBEDDING

    def test_cross_encoder_classified_as_rerank(self) -> None:
        """cross-encoder/* sentence-transformers rerankers hit the reranker path."""
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.model_manager.discovery import _classify_remote_task

        assert _classify_remote_task("cross-encoder/ms-marco-MiniLM-L-6-v2", "") == ModelTask.RERANK

    def test_chat_model_classified_as_chat(self) -> None:
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.model_manager.discovery import _classify_remote_task

        assert _classify_remote_task("qwen3:8b", "qwen") == ModelTask.CHAT

    def test_vision_model_classified_as_vision(self) -> None:
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.model_manager.discovery import _classify_remote_task

        assert _classify_remote_task("llava:13b", "llama") == ModelTask.VISION


class TestRemoteModelProvider:
    def test_classify_remote_models_sets_provider(self) -> None:
        from lilbee.modelhub.model_manager import classify_remote_models
        from lilbee.providers.local_servers import OLLAMA

        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3:latest", "details": {"family": "llama", "parameter_size": "8B"}}
            ]
        }
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            result = classify_remote_models("http://localhost:11434", OLLAMA)

        assert len(result) == 1
        assert result[0].provider == "Ollama"

    def test_classify_remote_models_openai_compatible_endpoint(self) -> None:
        """An OpenAI-compatible ``/v1/models`` endpoint is parsed via the LM Studio spec."""
        from lilbee.modelhub.model_manager import classify_remote_models
        from lilbee.providers.local_servers import LM_STUDIO

        mock_response = mock.Mock()
        mock_response.json.return_value = {"data": [{"id": "gpt-4"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            result = classify_remote_models("https://api.openai.com/v1", LM_STUDIO)

        assert len(result) == 1
        assert result[0].name == "gpt-4"
        assert result[0].provider == "LM Studio"

    def test_classify_lm_studio_models_via_openai_endpoint(self) -> None:
        """LM Studio is listed via ``/v1/models`` and labeled ``LM Studio``."""
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.model_manager import classify_remote_models
        from lilbee.providers.local_servers import LM_STUDIO

        mock_response = mock.Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen2.5-7b-instruct"},
                {"id": "nomic-embed-text-v1.5"},
                {"id": "bge-reranker-v2-m3"},
            ]
        }
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response) as mock_get:
            result = classify_remote_models("http://localhost:1234/v1", LM_STUDIO)

        # Hit the OpenAI-compatible endpoint (no doubled /v1), not Ollama's /api/tags.
        assert mock_get.call_args.args[0] == "http://localhost:1234/v1/models"
        assert [m.name for m in result] == [
            "qwen2.5-7b-instruct",
            "nomic-embed-text-v1.5",
            "bge-reranker-v2-m3",
        ]
        assert all(m.provider == "LM Studio" for m in result)
        # Name-pattern classification still works without family metadata.
        by_name = {m.name: m.task for m in result}
        assert by_name["qwen2.5-7b-instruct"] == ModelTask.CHAT
        assert by_name["nomic-embed-text-v1.5"] == ModelTask.EMBEDDING
        assert by_name["bge-reranker-v2-m3"] == ModelTask.RERANK

    def test_classify_lm_studio_surfaces_remote_lm_link_models(self) -> None:
        """LM Link remote/cloud models appear in /v1/models and are not filtered out."""
        from lilbee.modelhub.model_manager import classify_remote_models
        from lilbee.providers.local_servers import LM_STUDIO

        mock_response = mock.Mock()
        # A cloud/remote id LM Studio exposes via LM Link, alongside a local one.
        mock_response.json.return_value = {
            "data": [{"id": "local-qwen2.5-7b"}, {"id": "openai/gpt-oss-120b"}]
        }
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            result = classify_remote_models("http://localhost:1234/v1", LM_STUDIO)

        assert {m.name for m in result} == {"local-qwen2.5-7b", "openai/gpt-oss-120b"}

    def test_classify_lm_studio_skips_entries_without_id(self) -> None:
        """``/v1/models`` rows lacking an ``id`` are skipped, not crashed on."""
        from lilbee.modelhub.model_manager import classify_remote_models
        from lilbee.providers.local_servers import LM_STUDIO

        mock_response = mock.Mock()
        mock_response.json.return_value = {"data": [{"id": ""}, {}, {"id": "qwen2.5-7b-instruct"}]}
        mock_response.raise_for_status = mock.Mock()

        with mock.patch("httpx.get", return_value=mock_response):
            result = classify_remote_models("http://localhost:1234/v1", LM_STUDIO)

        assert [m.name for m in result] == ["qwen2.5-7b-instruct"]

    def test_classify_lm_studio_returns_empty_on_http_error(self) -> None:
        """A down LM Studio server yields [] so read-only callers stay responsive."""
        import httpx

        from lilbee.modelhub.model_manager import classify_remote_models
        from lilbee.providers.local_servers import LM_STUDIO

        with mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = classify_remote_models("http://localhost:1234/v1", LM_STUDIO)

        assert result == []

    def test_remote_model_default_provider(self) -> None:
        model = RemoteModel(name="test", task="chat", family="llama", parameter_size="8B")
        assert model.provider == "Remote"

    def test_classify_all_merges_configured_servers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both servers are probed and results are merged, source-labeled per server."""
        from lilbee.core.config import cfg
        from lilbee.modelhub.model_manager import classify_all_remote_models

        monkeypatch.setattr(cfg, "ollama_base_url", "")
        monkeypatch.setattr(cfg, "lm_studio_base_url", "")

        def fake_get(url: str, timeout: float) -> mock.Mock:
            resp = mock.Mock()
            resp.raise_for_status = mock.Mock()
            if "/api/tags" in url:
                resp.json.return_value = {
                    "models": [
                        {"name": "llama3:latest", "details": {"family": "llama"}},
                    ]
                }
            else:
                resp.json.return_value = {"data": [{"id": "qwen2.5-7b"}]}
            return resp

        with mock.patch("httpx.get", side_effect=fake_get):
            result = classify_all_remote_models()

        assert {m.name: m.provider for m in result} == {
            "llama3:latest": "Ollama",
            "qwen2.5-7b": "LM Studio",
        }


class TestHasProviderKey:
    def test_env_var_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert _has_provider_key("openai_api_key", "OPENAI_API_KEY") is True

    def test_env_var_absent_config_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.core.config import cfg

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg.openai_api_key = "sk-from-config"
        assert _has_provider_key("openai_api_key", "OPENAI_API_KEY") is True

    def test_neither_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.core.config import cfg

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg.anthropic_api_key = ""
        assert _has_provider_key("anthropic_api_key", "ANTHROPIC_API_KEY") is False

    def test_unknown_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SOME_KEY", raising=False)
        assert _has_provider_key("nonexistent_field", "SOME_KEY") is False


class TestDiscoverApiModels:
    @pytest.fixture(autouse=True)
    def _force_auto_provider(self) -> Iterator[None]:
        # discover_api_models() calls get_services().provider.list_chat_models.
        # LlamaCppProvider.list_chat_models hard-codes []; only RoutingProvider
        # (cfg.llm_provider == "auto") delegates to the SDK backend where the
        # sys.modules["litellm"] patch these tests rely on can take effect.
        # Developers whose config.toml pins llm_provider="llama-cpp" would
        # otherwise see these tests fail locally while passing in CI.
        from lilbee.app.services import reset_services
        from lilbee.core.config import cfg

        cfg.llm_provider = "auto"
        reset_services()
        yield
        reset_services()

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
        from lilbee.core.config import cfg

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
        from lilbee.core.config import cfg

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
        from lilbee.core.config import cfg

        cfg.gemini_api_key = ""

        with mock.patch.dict("sys.modules", {"litellm": mock_litellm}):
            # Test arbitrary upstream ids; pin  so curation
            # doesn't filter them.
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
        from lilbee.core.config import cfg

        cfg.openai_api_key = ""
        cfg.gemini_api_key = ""

        with mock.patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = discover_api_models()

        model = result["Anthropic"][0]
        assert model.provider == "Anthropic"
        assert model.name == "claude-sonnet-4-6"
