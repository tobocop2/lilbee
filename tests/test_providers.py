"""Tests for the LLM provider abstraction layer (mocked — no live servers needed)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import httpx
import pytest

from lilbee.config import cfg

if TYPE_CHECKING:
    from lilbee.providers.routing_provider import RoutingProvider
    from lilbee.providers.sdk_llm_provider import SdkLLMProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_provider() -> None:
    """Reset provider singleton between tests."""
    import lilbee.providers.llama_cpp_provider as lcp
    from lilbee.services import reset_services

    reset_services()
    lcp._registry = None
    yield
    reset_services()
    lcp._registry = None


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory with a registered test model."""
    from lilbee.registry import ModelManifest, ModelRef, ModelRegistry

    models = tmp_path / "models"
    models.mkdir()
    registry = ModelRegistry(models)

    source = tmp_path / "test-model.gguf"
    source.write_bytes(b"fake-gguf")
    ref = ModelRef(name="test-model")
    manifest = ModelManifest(
        name="test-model",
        tag="latest",
        size_bytes=9,
        task="chat",
        source_repo="org/test-model-GGUF",
        source_filename="test-model.gguf",
        downloaded_at="2026-01-01T00:00:00+00:00",
    )
    registry.install(ref, source, manifest)
    return models


@pytest.fixture()
def mock_llama_cpp() -> mock.MagicMock:
    """Inject a mock llama_cpp module into sys.modules."""
    mod = mock.MagicMock()
    sys.modules["llama_cpp"] = mod
    yield mod
    sys.modules.pop("llama_cpp", None)


# ---------------------------------------------------------------------------
# ProviderError
# ---------------------------------------------------------------------------


class TestProviderError:
    def test_message(self) -> None:
        from lilbee.providers.base import ProviderError

        err = ProviderError("something broke")
        assert str(err) == "something broke"
        assert err.provider == ""

    def test_with_provider(self) -> None:
        from lilbee.providers.base import ProviderError

        err = ProviderError("fail", provider="test")
        assert err.provider == "test"


# ---------------------------------------------------------------------------
# LlamaCppProvider
# ---------------------------------------------------------------------------


class TestLlamaCppProvider:
    @pytest.fixture(autouse=True)
    def _shutdown_provider(self, models_dir: Path) -> None:
        """Ensure any LlamaCppProvider created in a test is shut down.
        Also patches resolve_model_path so the daemon embed thread
        doesn't block on registry lookups for test .gguf files.
        """
        cfg.models_dir = models_dir
        cfg.embedding_model = "test-model"
        cfg.chat_model = "test-model"
        cfg.subprocess_embed = False
        self._providers: list = []
        self._resolve_patcher = mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            side_effect=lambda m: models_dir / f"{m}.gguf",
        )
        self._resolve_patcher.start()
        yield
        for p in self._providers:
            p.shutdown()
        self._resolve_patcher.stop()

    def _make_provider(self) -> object:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        p = LlamaCppProvider()
        self._providers.append(p)
        return p

    def test_embed(self, mock_llama_cpp: mock.MagicMock) -> None:
        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_embedding.side_effect = [
            {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            {"data": [{"embedding": [0.4, 0.5, 0.6]}]},
        ]
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = self._make_provider()
        result = provider.embed(["hello", "world"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert mock_llama_instance.create_embedding.call_count == 2

    def test_chat_non_stream(self, mock_llama_cpp: mock.MagicMock) -> None:
        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello there"}}]
        }
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "hi"}])

        assert result == "Hello there"

    def test_chat_stream(self, mock_llama_cpp: mock.MagicMock) -> None:
        stream_chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}}]},
        ]
        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = iter(stream_chunks)
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "hi"}], stream=True)

        tokens = list(result)
        assert tokens == ["Hello", " world"]

    def test_chat_empty_content(self, mock_llama_cpp: mock.MagicMock) -> None:
        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": None}}]
        }
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "hi"}])

        assert result == ""

    def test_chat_with_options(self, mock_llama_cpp: mock.MagicMock) -> None:
        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = self._make_provider()
        provider.chat(
            [{"role": "user", "content": "hi"}],
            options={"temperature": 0.5, "seed": 42},
        )

        mock_llama_instance.create_chat_completion.assert_called_once()
        call_kwargs = mock_llama_instance.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["seed"] == 42

    def test_chat_model_override(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        (models_dir / "other-model.gguf").write_bytes(b"fake")

        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = self._make_provider()
        provider.chat([{"role": "user", "content": "hi"}], model="other-model")

        # Llama should have been called with a path for other-model
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert "other-model" in call_kwargs["model_path"]

    def test_list_models(self, models_dir: Path) -> None:
        provider = self._make_provider()
        result = provider.list_models()
        assert result == ["test-model:latest"]

    def test_list_models_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        cfg.models_dir = empty

        provider = self._make_provider()
        assert provider.list_models() == []

    def test_list_models_no_dir(self, tmp_path: Path) -> None:
        cfg.models_dir = tmp_path / "nonexistent"

        provider = self._make_provider()
        assert provider.list_models() == []

    def test_pull_model_raises(self) -> None:
        provider = self._make_provider()
        with pytest.raises(NotImplementedError, match="cannot pull"):
            provider.pull_model("some-model")

    def test_list_chat_models_empty(self) -> None:
        # Native llama-cpp has no frontier-provider catalog.
        provider = self._make_provider()
        assert provider.list_chat_models("openai") == []

    def test_show_model_returns_none(self) -> None:
        from lilbee.providers.base import ProviderError

        provider = self._make_provider()
        # Override the class-level resolve mock to raise for this test
        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            side_effect=ProviderError("not found"),
        ):
            assert provider.show_model("some-model") is None

    def testread_gguf_metadata(self, models_dir: Path) -> None:
        from unittest.mock import MagicMock, patch

        from lilbee.providers.llama_cpp_provider import read_gguf_metadata

        mock_llm = MagicMock()
        mock_llm.metadata = {
            "general.architecture": "qwen3",
            "general.name": "Qwen3 8B",
            "general.file_type": "15",
            "qwen3.context_length": "32768",
            "qwen3.embedding_length": "4096",
            "tokenizer.chat_template": "{% if messages %}...",
        }
        with patch("llama_cpp.Llama", return_value=mock_llm):
            result = read_gguf_metadata(models_dir / "test-model.gguf")
        assert result["architecture"] == "qwen3"
        assert result["context_length"] == "32768"
        assert result["embedding_length"] == "4096"
        assert result["chat_template"] == "{% if messages %}..."
        assert result["name"] == "Qwen3 8B"
        mock_llm.close.assert_called_once()

    def testread_gguf_metadata_empty(self, models_dir: Path) -> None:
        from unittest.mock import MagicMock, patch

        from lilbee.providers.llama_cpp_provider import read_gguf_metadata

        mock_llm = MagicMock()
        mock_llm.metadata = {}
        with patch("llama_cpp.Llama", return_value=mock_llm):
            result = read_gguf_metadata(models_dir / "test-model.gguf")
        assert result is None

    def testload_llama_sets_n_batch_for_embedding(self, models_dir: Path) -> None:
        from unittest.mock import patch

        from lilbee.providers.llama_cpp_provider import load_llama

        cfg.num_ctx = None
        with (
            patch("llama_cpp.Llama") as mock_llama_cls,
            patch(
                "lilbee.providers.llama_cpp_provider.read_gguf_metadata",
                return_value={"context_length": "2048"},
            ),
        ):
            load_llama(models_dir / "test-model.gguf", mode="embed")
            call_kwargs = mock_llama_cls.call_args[1]
            assert call_kwargs["n_batch"] == 2048
            assert call_kwargs["n_ubatch"] == 2048
            assert call_kwargs["embedding"] is True

    def testload_llama_no_n_batch_for_chat(self, models_dir: Path) -> None:
        from unittest.mock import patch

        from lilbee.providers.llama_cpp_provider import load_llama

        with patch("llama_cpp.Llama"):
            load_llama(models_dir / "test-model.gguf", mode="chat")
            import llama_cpp

            call_kwargs = llama_cpp.Llama.call_args[1]
            assert "n_batch" not in call_kwargs

    def testresolve_model_path_direct(self, models_dir: Path, tmp_path: Path) -> None:
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.llama_cpp_provider import resolve_model_path

            cfg.models_dir = models_dir
            abs_model = tmp_path / "standalone.gguf"
            abs_model.write_bytes(b"standalone-model")
            path = resolve_model_path(str(abs_model))
            assert path == abs_model
        finally:
            self._resolve_patcher.start()

    def testresolve_model_path_via_registry(self, models_dir: Path) -> None:
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.llama_cpp_provider import resolve_model_path

            cfg.models_dir = models_dir
            path = resolve_model_path("test-model")
            assert path.exists()
        finally:
            self._resolve_patcher.start()

    def testresolve_model_path_registry_with_tag(self, models_dir: Path) -> None:
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.llama_cpp_provider import resolve_model_path

            cfg.models_dir = models_dir
            path = resolve_model_path("test-model:latest")
            assert path.exists()
        finally:
            self._resolve_patcher.start()

    def testresolve_model_path_not_found(self, models_dir: Path) -> None:
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.base import ProviderError
            from lilbee.providers.llama_cpp_provider import resolve_model_path

            cfg.models_dir = models_dir
            with pytest.raises(ProviderError, match="not found"):
                resolve_model_path("missing-model")
        finally:
            self._resolve_patcher.start()

    def testresolve_model_path_direct_not_exists(self, models_dir: Path, tmp_path: Path) -> None:
        self._resolve_patcher.stop()
        try:
            from lilbee.providers.base import ProviderError
            from lilbee.providers.llama_cpp_provider import resolve_model_path

            cfg.models_dir = models_dir
            # Use a real absolute path that doesn't exist (works on all platforms)
            fake_path = str(tmp_path / "nonexistent" / "model.gguf")
            with pytest.raises(ProviderError, match="Model file not found"):
                resolve_model_path(fake_path)
        finally:
            self._resolve_patcher.start()

    def test_embed_caches_llm(self, mock_llama_cpp: mock.MagicMock) -> None:
        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_embedding.return_value = {"data": [{"embedding": [0.1] * 3}]}
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        cfg.num_ctx = 4096  # Explicit ctx skips metadata read
        provider = self._make_provider()
        provider.embed(["a"])
        provider.embed(["b"])

        # With explicit num_ctx, no metadata read needed — only 1 Llama call.
        # Second embed reuses the cached instance.
        assert mock_llama_cpp.Llama.call_count == 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_default_provider_is_routing(self) -> None:
        from lilbee.providers.factory import create_provider
        from lilbee.providers.routing_provider import RoutingProvider

        cfg.llm_provider = "auto"
        provider = create_provider(cfg)
        assert isinstance(provider, RoutingProvider)

    def test_explicit_llama_cpp(self) -> None:
        from lilbee.providers.factory import create_provider
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.llm_provider = "llama-cpp"
        provider = create_provider(cfg)
        assert isinstance(provider, LlamaCppProvider)

    def test_unknown_provider_raises(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.factory import create_provider

        cfg.llm_provider = "unknown"
        with pytest.raises(ProviderError, match="Unknown LLM provider"):
            create_provider(cfg)

    def test_services_singleton(self) -> None:
        from lilbee.services import get_services, reset_services

        reset_services()
        cfg.llm_provider = "llama-cpp"
        p1 = get_services().provider
        p2 = get_services().provider
        assert p1 is p2
        reset_services()

    def test_services_reset_clears_singleton(self) -> None:
        from lilbee.services import get_services, reset_services

        reset_services()
        cfg.llm_provider = "llama-cpp"
        p1 = get_services().provider
        reset_services()
        p2 = get_services().provider
        assert p1 is not p2
        reset_services()


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigProvider:
    def test_default_llm_provider(self, tmp_path) -> None:
        env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("LILBEE_")}
        # Point LILBEE_DATA at a clean tmp dir so the test doesn't pick up a
        # user-local config.toml whose model slots predate strict catalog
        # validation (round 4).
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_SKIP_MODEL_TASK_VALIDATION"] = "1"
        with (
            mock.patch.dict(__import__("os").environ, env, clear=True),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            from lilbee.config import Config

            c = Config()
            assert c.llm_provider == "auto"
            assert c.backend_base_url == "http://localhost:11434"
            assert c.llm_api_key == ""

    def test_provider_env_override(self) -> None:
        import os

        with mock.patch.dict(
            os.environ,
            {
                "LILBEE_LLM_PROVIDER": "backend",
                "LILBEE_BACKEND_BASE_URL": "http://myhost:11434",
                "LILBEE_LLM_API_KEY": "sk-key",
            },
        ):
            from lilbee.config import Config

            c = Config()
            assert c.llm_provider == "backend"
            assert c.backend_base_url == "http://myhost:11434"
            assert c.llm_api_key == "sk-key"

    def test_models_dir_uses_canonical_location(self, tmp_path: Path) -> None:
        """models_dir always uses the canonical shared location, not data_root."""
        import os

        from lilbee.platform import canonical_models_dir

        with mock.patch.dict(os.environ, {"LILBEE_DATA": str(tmp_path / "test-lilbee")}):
            from lilbee.config import Config

            c = Config()
            assert c.models_dir == canonical_models_dir()


# ---------------------------------------------------------------------------
# RoutingProvider
# ---------------------------------------------------------------------------


class TestRoutingProvider:
    @pytest.fixture(autouse=True)
    def _shutdown_provider(self):
        """Ensure all LlamaCppProvider background threads are stopped."""
        self._to_shutdown: list = []
        yield
        for p in self._to_shutdown:
            p.shutdown()

    def _make_provider(self) -> RoutingProvider:
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        # Track the real llama-cpp provider for shutdown (tests replace it with mocks)
        if rp._llama_cpp is not None:
            self._to_shutdown.append(rp._llama_cpp)
        self._to_shutdown.append(rp)
        return rp

    def test_routes_chat_to_litellm_for_api_model(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.chat.return_value = "hello"
        rp._sdk_provider = mock_litellm

        result = rp.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")
        assert result == "hello"
        mock_litellm.chat.assert_called_once()

    def test_routes_chat_to_litellm_for_ollama_model(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.chat.return_value = "hello"
        rp._sdk_provider = mock_litellm

        result = rp.chat([{"role": "user", "content": "hi"}], model="ollama/qwen3:8b")
        assert result == "hello"
        mock_litellm.chat.assert_called_once()

    def test_routes_chat_to_llama_cpp_for_bare_ref(self) -> None:
        """Bare refs dispatch to llama-cpp regardless of registry contents.

        The new routing is strict: no prefix means native. If the native
        registry doesn't have the model, llama-cpp raises its own
        'not installed' error; routing never falls through to litellm.
        """
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.chat.return_value = "local"
        rp._llama_cpp = mock_llama

        cfg.chat_model = "local-model:latest"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "local"
        mock_llama.chat.assert_called_once()

    def test_routes_embed_to_litellm_for_ollama_model(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.embed.return_value = [[0.1, 0.2]]
        rp._sdk_provider = mock_litellm

        cfg.embedding_model = "ollama/nomic-embed-text:latest"
        result = rp.embed(["test"])
        assert result == [[0.1, 0.2]]
        mock_litellm.embed.assert_called_once()

    def test_routes_embed_to_llama_cpp_for_bare_ref(self) -> None:
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.embed.return_value = [[0.3, 0.4]]
        rp._llama_cpp = mock_llama

        cfg.embedding_model = "nomic-embed-text:latest"
        result = rp.embed(["test"])
        assert result == [[0.3, 0.4]]

    def test_bare_ref_never_falls_through_to_litellm(self) -> None:
        """Bare refs stay on llama-cpp even when litellm is installed.

        This is the behaviour change from bb-0ud4's probe-based routing:
        prefix is the single source of truth, so users who want Ollama
        must say so explicitly with 'ollama/<name>'.
        """
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_llama = mock.MagicMock()
        mock_llama.embed.return_value = [[0.9, 1.0]]
        rp._sdk_provider = mock_litellm
        rp._llama_cpp = mock_llama

        cfg.embedding_model = "mistral:latest"
        result = rp.embed(["test"])
        assert result == [[0.9, 1.0]]
        mock_llama.embed.assert_called_once()
        mock_litellm.embed.assert_not_called()

    def test_list_models_native_only_when_sdk_unavailable(self) -> None:
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.list_models.return_value = ["local.gguf"]
        rp._llama_cpp = mock_llama

        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = False
        rp._sdk_provider = mock_sdk

        result = rp.list_models()
        assert result == ["local.gguf"]
        mock_sdk.list_models.assert_not_called()

    def test_list_models_union_when_both_available(self) -> None:
        """Native and remote listings are merged when the SDK backend is installed."""
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.list_models.return_value = ["local.gguf"]
        rp._llama_cpp = mock_llama

        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = True
        mock_sdk.list_models.return_value = ["ollama/qwen3:8b"]
        rp._sdk_provider = mock_sdk

        result = rp.list_models()
        assert result == ["local.gguf", "ollama/qwen3:8b"]

    def test_list_models_remote_error_returns_native_only(self) -> None:
        """A failing SDK backend doesn't mask the native registry listing."""
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.list_models.return_value = ["local.gguf"]
        rp._llama_cpp = mock_llama

        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = True
        mock_sdk.list_models.side_effect = RuntimeError("remote down")
        rp._sdk_provider = mock_sdk

        result = rp.list_models()
        assert result == ["local.gguf"]

    def test_get_llama_cpp_caches_instance(self) -> None:
        """``_get_llama_cpp`` memoizes the LlamaCppProvider on first call."""
        from lilbee.providers.routing_provider import RoutingProvider

        rp = RoutingProvider()
        self._to_shutdown.append(rp)
        first = rp._get_llama_cpp()
        self._to_shutdown.append(first)
        second = rp._get_llama_cpp()
        assert first is second

    def test_get_sdk_provider_caches_instance(self) -> None:
        """``_get_sdk_provider`` memoizes the SdkLLMProvider on first call."""
        rp = self._make_provider()
        first = rp._get_sdk_provider()
        second = rp._get_sdk_provider()
        assert first is second

    def test_list_chat_models_empty_when_sdk_unavailable(self) -> None:
        # list_chat_models must skip the SDK backend entirely when the SDK
        # is not installed; native llama-cpp never has a frontier catalog.
        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = False
        rp._sdk_provider = mock_sdk

        assert rp.list_chat_models("openai") == []
        mock_sdk.list_chat_models.assert_not_called()

    def test_list_chat_models_delegates_through_sdk_provider(self) -> None:
        # Pins the suppression chain: list_chat_models must reach the SDK
        # provider (not the adapter directly), so SdkLLMProvider._ensure_initialized
        # can apply cfg.json_mode before any SDK import inside the backend.
        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = True
        mock_sdk.list_chat_models.return_value = ["openai/gpt-4o", "openai/gpt-4o-mini"]
        rp._sdk_provider = mock_sdk

        result = rp.list_chat_models("openai")

        assert result == ["openai/gpt-4o", "openai/gpt-4o-mini"]
        mock_sdk.list_chat_models.assert_called_once_with("openai")

    def test_show_model_delegates_by_prefix(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.show_model.return_value = {"parameters": "temp 0.7"}
        rp._sdk_provider = mock_litellm

        result = rp.show_model("ollama/qwen3:8b")
        assert result == {"parameters": "temp 0.7"}
        mock_litellm.show_model.assert_called_once_with("ollama/qwen3:8b")

    def test_show_model_bare_ref_uses_llama_cpp(self) -> None:
        rp = self._make_provider()

        mock_llama = mock.MagicMock()
        mock_llama.show_model.return_value = None
        rp._llama_cpp = mock_llama

        result = rp.show_model("local.gguf")
        assert result is None
        mock_llama.show_model.assert_called_once_with("local.gguf")

    def test_pull_model_raises_when_sdk_unavailable(self) -> None:
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = False
        rp._sdk_provider = mock_sdk

        with pytest.raises(ProviderError, match="no pull-capable backend"):
            rp.pull_model("bad-model")
        mock_sdk.pull_model.assert_not_called()

    def test_pull_model_delegates_to_sdk_when_available(self) -> None:
        """With the SDK backend available, pull_model forwards to the SDK provider."""
        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        mock_sdk.available.return_value = True
        rp._sdk_provider = mock_sdk
        captured: dict[str, object] = {}

        def _on_progress(evt: dict[str, object]) -> None:
            captured["saw"] = evt

        rp.pull_model("ollama/llama3:8b", on_progress=_on_progress)

        mock_sdk.pull_model.assert_called_once_with("ollama/llama3:8b", on_progress=_on_progress)

    def test_chat_with_explicit_api_model_override(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.chat.return_value = "saw it"
        rp._sdk_provider = mock_litellm

        cfg.chat_model = "local-model:latest"
        result = rp.chat(
            [{"role": "user", "content": "describe"}],
            model="openai/gpt-4o",
        )
        assert result == "saw it"
        mock_litellm.chat.assert_called_once()

    def test_get_capabilities_delegates_by_prefix(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.get_capabilities.return_value = ["completion", "vision"]
        rp._sdk_provider = mock_litellm

        caps = rp.get_capabilities("ollama/llava:7b")
        assert caps == ["completion", "vision"]
        mock_litellm.get_capabilities.assert_called_once_with("ollama/llava:7b")


# ---------------------------------------------------------------------------
# litellm_available guard
# ---------------------------------------------------------------------------


class TestLitellmAvailable:
    def test_returns_false_when_not_installed(self) -> None:
        from lilbee.providers.litellm_sdk import litellm_available

        with mock.patch.dict("sys.modules", {"litellm": None}):
            assert litellm_available() is False

    def test_factory_raises_when_litellm_unavailable(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.factory import create_provider
        from lilbee.providers.litellm_sdk import LitellmSdkBackend

        cfg.llm_provider = "backend"
        with (
            mock.patch.object(LitellmSdkBackend, "available", return_value=False),
            pytest.raises(ProviderError, match="SDK backend adapter is not installed"),
        ):
            create_provider(cfg)


class TestLiteLLMShowModelCapabilities:
    """Tests for SdkLLMProvider.show_model capabilities parsing via litellm backend."""

    def _make_provider(self) -> SdkLLMProvider:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        return SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")

    def test_show_model_returns_capabilities(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {
            "capabilities": ["completion", "vision"],
            "parameters": "temperature 0.7",
        }
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("llava:7b")

        assert result is not None
        assert result["capabilities"] == ["completion", "vision"]
        assert result["parameters"] == "temperature 0.7"

    def test_show_model_no_capabilities_field(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": "temperature 0.7"}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("qwen3:8b")

        assert result is not None
        assert "capabilities" not in result
        assert result["parameters"] == "temperature 0.7"

    def test_show_model_only_capabilities_no_params(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"capabilities": ["completion"]}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("some-model")

        assert result is not None
        assert result["capabilities"] == ["completion"]

    def test_show_model_empty_returns_none(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            result = provider.show_model("empty-model")

        assert result is None

    def test_show_model_http_error(self) -> None:
        provider = self._make_provider()
        with mock.patch("httpx.post", side_effect=httpx.HTTPError("fail")):
            result = provider.show_model("bad-model")

        assert result is None

    def test_get_capabilities_returns_list(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"capabilities": ["completion", "vision", "tools"]}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            caps = provider.get_capabilities("llava:7b")

        assert caps == ["completion", "vision", "tools"]

    def test_get_capabilities_returns_empty_on_error(self) -> None:
        provider = self._make_provider()
        with mock.patch("httpx.post", side_effect=httpx.HTTPError("fail")):
            caps = provider.get_capabilities("bad-model")

        assert caps == []

    def test_get_capabilities_no_capabilities_field(self) -> None:
        provider = self._make_provider()
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": "temp 0.7"}
        mock_resp.raise_for_status = mock.Mock()

        with mock.patch("httpx.post", return_value=mock_resp):
            caps = provider.get_capabilities("qwen3:8b")

        assert caps == []


# ---------------------------------------------------------------------------
# Phase 2: _dispatch_batch, embed fallback, vision_ocr, chat stream,
# show_model None, shutdown, _LockedStreamIterator, GGUF helpers,
# vision handler resolution, WorkerProcess None-response paths
# ---------------------------------------------------------------------------


class TestDispatchBatch:
    def testembed_one_at_a_time(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_dispatch_batch embeds one text at a time and resolves the future."""
        from concurrent.futures import Future

        from lilbee.providers.llama_cpp_provider import LlamaCppProvider, _EmbedRequest

        mock_llm = mock.MagicMock()
        mock_llm.create_embedding.side_effect = [
            {"data": [{"embedding": [0.1]}]},
            {"data": [{"embedding": [0.2]}]},
        ]

        provider = LlamaCppProvider()
        fut: Future[list[list[float]]] = Future()
        with mock.patch.object(provider, "_get_embed_llm", return_value=mock_llm):
            provider._dispatch_batch([_EmbedRequest(texts=["a", "b"], future=fut)])
        assert fut.result() == [[0.1], [0.2]]

    def test_exception_sets_future_exception(self, mock_llama_cpp: mock.MagicMock) -> None:
        """When embedding fails, the future receives the exception."""
        from concurrent.futures import Future

        from lilbee.providers.llama_cpp_provider import LlamaCppProvider, _EmbedRequest

        mock_llm = mock.MagicMock()
        mock_llm.create_embedding.side_effect = RuntimeError("GPU OOM")

        provider = LlamaCppProvider()
        fut: Future[list[list[float]]] = Future()
        with mock.patch.object(provider, "_get_embed_llm", return_value=mock_llm):
            provider._dispatch_batch([_EmbedRequest(texts=["a"], future=fut)])
        with pytest.raises(RuntimeError, match="GPU OOM"):
            fut.result()


class TestEmbedSubprocessFallback:
    def test_oserror_disables_subprocess(self, mock_llama_cpp: mock.MagicMock) -> None:
        """OSError from subprocess worker falls back to in-process embedding."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider, _EmbedRequest

        provider = LlamaCppProvider()
        provider._subprocess_enabled = True
        mock_worker = mock.MagicMock()
        mock_worker.embed.side_effect = OSError("No child processes")
        provider._subprocess_worker = mock_worker

        # The in-process fallback puts a request on the embed queue.
        # Mock the queue to capture the request and resolve the future.
        original_put = provider._embed_queue.put

        def _intercept_put(item):
            if isinstance(item, _EmbedRequest):
                item.future.set_result([[0.5]])
            else:
                original_put(item)

        with mock.patch.object(provider._embed_queue, "put", side_effect=_intercept_put):
            result = provider.embed(["test"])
        assert result == [[0.5]]
        assert provider._subprocess_enabled is False


class TestVisionOcr:
    def test_delegates_to_subprocess(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        provider = LlamaCppProvider()
        mock_worker = mock.MagicMock()
        mock_worker.vision_ocr.return_value = "extracted text"
        provider._subprocess_worker = mock_worker

        result = provider.vision_ocr(b"\x89PNG", "vision-model", "describe")
        assert result == "extracted text"
        mock_worker.vision_ocr.assert_called_once_with(b"\x89PNG", "vision-model", "describe")


class TestChatStreamReturnsLockedIterator:
    def test_stream_returns_locked_iterator(self, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider, _LockedStreamIterator

        mock_llm = mock.MagicMock()
        mock_llm.create_chat_completion.return_value = iter([])

        provider = LlamaCppProvider()
        with mock.patch.object(provider, "_get_chat_llm", return_value=mock_llm):
            result = provider.chat([{"role": "user", "content": "hi"}], stream=True)
        assert isinstance(result, _LockedStreamIterator)
        # Exhaust the iterator to release the lock
        list(result)


class TestShowModelNotFound:
    def test_returns_none_for_missing_model(self) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        provider = LlamaCppProvider()
        assert provider.show_model("nonexistent-model-xyz") is None


class TestShutdown:
    def test_stops_subprocess_worker(self) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        provider = LlamaCppProvider()
        mock_worker = mock.MagicMock()
        provider._subprocess_worker = mock_worker

        provider.shutdown()
        mock_worker.stop.assert_called_once()
        assert provider._subprocess_worker is None


class TestLockedStreamIterator:
    def test_next_and_close(self) -> None:
        """__next__ yields content, close() releases the lock."""
        import threading

        from lilbee.providers.llama_cpp_provider import _LockedStreamIterator

        lock = threading.Lock()
        lock.acquire()
        chunks = iter(
            [
                {"choices": [{"delta": {"content": "hi"}}]},
            ]
        )
        it = _LockedStreamIterator(chunks, lock)
        assert next(it) == "hi"
        it.close()
        assert lock.acquire(blocking=False)  # lock was released
        lock.release()

    def test_close_releases_lock_early(self) -> None:
        import threading

        from lilbee.providers.llama_cpp_provider import _LockedStreamIterator

        lock = threading.Lock()
        lock.acquire()
        it = _LockedStreamIterator(iter([]), lock)
        it.close()
        # lock should be released
        assert lock.acquire(blocking=False)
        lock.release()

    def test_close_drains_remaining_tokens(self) -> None:
        """close() exhausts the underlying iterator before releasing the lock."""
        import threading

        from lilbee.providers.llama_cpp_provider import _LockedStreamIterator

        lock = threading.Lock()
        lock.acquire()
        chunks = [
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}}]},
        ]
        it = _LockedStreamIterator(iter(chunks), lock)
        # Don't consume any tokens, just close
        it.close()
        assert lock.acquire(blocking=False)
        lock.release()

    def test_close_handles_drain_exception(self) -> None:
        """close() releases lock even if draining the iterator raises."""
        import threading

        from lilbee.providers.llama_cpp_provider import _LockedStreamIterator

        def _exploding():
            yield {"choices": [{"delta": {"content": "a"}}]}
            raise RuntimeError("boom")

        lock = threading.Lock()
        lock.acquire()
        it = _LockedStreamIterator(_exploding(), lock)
        it.close()  # should not raise
        assert lock.acquire(blocking=False)
        lock.release()


class TestReadMmprojProjectorType:
    def test_reads_projector_type(self, tmp_path: Path) -> None:
        import struct

        from lilbee.providers.llama_cpp_provider import read_mmproj_projector_type

        # Build a minimal GGUF file with clip.projector_type = "ldp"
        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)  # version
        buf += struct.pack("<Q", 0)  # tensor_count
        buf += struct.pack("<Q", 1)  # kv_count = 1
        key = b"clip.projector_type"
        buf += struct.pack("<Q", len(key))
        buf += key
        buf += struct.pack("<I", 8)  # value_type = string
        value = b"ldp"
        buf += struct.pack("<Q", len(value))
        buf += value

        gguf_file = tmp_path / "test_mmproj.gguf"
        gguf_file.write_bytes(bytes(buf))
        assert read_mmproj_projector_type(gguf_file) == "ldp"

    def test_exception_returns_none(self) -> None:
        from lilbee.providers.llama_cpp_provider import read_mmproj_projector_type

        assert read_mmproj_projector_type(Path("/nonexistent/file.gguf")) is None

    def test_non_string_projector_type_returns_none(self, tmp_path: Path) -> None:
        """If clip.projector_type is present but not a string (someone wrote it
        as an int or bool), the reader returns None instead of decoding bytes."""
        import struct

        from lilbee.providers.llama_cpp_provider import read_mmproj_projector_type

        # Build a GGUF where clip.projector_type is value_type=4 (uint32), not string.
        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)  # version
        buf += struct.pack("<Q", 0)  # tensor_count
        buf += struct.pack("<Q", 1)  # kv_count = 1
        key = b"clip.projector_type"
        buf += struct.pack("<Q", len(key)) + key
        buf += struct.pack("<I", 4)  # value_type = uint32
        buf += struct.pack("<I", 42)  # bogus integer payload

        gguf_file = tmp_path / "wrong_type_mmproj.gguf"
        gguf_file.write_bytes(bytes(buf))
        assert read_mmproj_projector_type(gguf_file) is None

    def test_reads_projector_type_past_bool_kv(self, tmp_path: Path) -> None:
        """Parser must skip bool KV pairs (1 byte each) to reach projector_type.
        LightOn OCR2's mmproj has ``clip.has_vision_encoder`` (bool) preceding
        ``clip.projector_type``. A bool skip-size regression would over-advance
        the stream and parse_header of the next key would raise.
        """
        import struct

        from lilbee.providers.llama_cpp_provider import read_mmproj_projector_type

        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<I", 3)
        buf += struct.pack("<Q", 0)
        buf += struct.pack("<Q", 2)  # 2 KV pairs
        bool_key = b"clip.has_vision_encoder"
        buf += struct.pack("<Q", len(bool_key)) + bool_key
        buf += struct.pack("<I", 7)  # bool
        buf += b"\x01"  # single byte
        proj_key = b"clip.projector_type"
        buf += struct.pack("<Q", len(proj_key)) + proj_key
        buf += struct.pack("<I", 8)  # string
        proj_val = b"lightonocr"  # the real regression case
        buf += struct.pack("<Q", len(proj_val)) + proj_val

        f = tmp_path / "mmproj.gguf"
        f.write_bytes(bytes(buf))
        assert read_mmproj_projector_type(f) == "lightonocr"


class TestResolveVisionHandler:
    def test_known_projector(self, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import _resolve_vision_handler

        handler_cls = mock.MagicMock()
        mock_llama_cpp.llama_chat_format.Llava15ChatHandler = handler_cls
        mock_llama_cpp.llama_chat_format.MiniCPMv26ChatHandler = mock.MagicMock()

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.read_mmproj_projector_type",
            return_value="minicpmv",
        ):
            result = _resolve_vision_handler(Path("test.gguf"))
        assert result is mock_llama_cpp.llama_chat_format.MiniCPMv26ChatHandler

    def test_unknown_projector_falls_back(self, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import _resolve_vision_handler

        fallback = mock.MagicMock()
        mock_llama_cpp.llama_chat_format.Llava15ChatHandler = fallback
        # Register the submodule so `from llama_cpp.llama_chat_format import ...` works
        sys.modules["llama_cpp.llama_chat_format"] = mock_llama_cpp.llama_chat_format

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.read_mmproj_projector_type",
            return_value="totally_unknown_projector",
        ):
            result = _resolve_vision_handler(Path("test.gguf"))
        assert result is fallback
        sys.modules.pop("llama_cpp.llama_chat_format", None)

    def test_handler_not_found_falls_back(self, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import _resolve_vision_handler

        fallback = mock.MagicMock()
        # Use a module mock where getattr returns None for the handler
        fake_chat_format = mock.MagicMock(spec=[])
        fake_chat_format.Llava15ChatHandler = fallback
        mock_llama_cpp.llama_chat_format = fake_chat_format
        sys.modules["llama_cpp.llama_chat_format"] = fake_chat_format

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.read_mmproj_projector_type",
            return_value="minicpmv",
        ):
            result = _resolve_vision_handler(Path("test.gguf"))
        assert result is fallback
        sys.modules.pop("llama_cpp.llama_chat_format", None)

    def test_no_projector_falls_back(self, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import _resolve_vision_handler

        fallback = mock.MagicMock()
        mock_llama_cpp.llama_chat_format.Llava15ChatHandler = fallback
        sys.modules["llama_cpp.llama_chat_format"] = mock_llama_cpp.llama_chat_format

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.read_mmproj_projector_type",
            return_value=None,
        ):
            result = _resolve_vision_handler(Path("test.gguf"))
        assert result is fallback
        sys.modules.pop("llama_cpp.llama_chat_format", None)


class TestLoadVisionLlama:
    def test_with_mmproj_and_num_ctx(self, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import load_vision_llama

        handler_cls = mock.MagicMock()
        mock_llama_cpp.llama_chat_format.Llava15ChatHandler = handler_cls
        cfg.num_ctx = 4096

        with mock.patch(
            "lilbee.providers.llama_cpp_provider._resolve_vision_handler",
            return_value=handler_cls,
        ):
            load_vision_llama(Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 4096

    def test_without_num_ctx(self, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import load_vision_llama

        handler_cls = mock.MagicMock()
        mock_llama_cpp.llama_chat_format.Llava15ChatHandler = handler_cls
        cfg.num_ctx = None

        with mock.patch(
            "lilbee.providers.llama_cpp_provider._resolve_vision_handler",
            return_value=handler_cls,
        ):
            load_vision_llama(Path("model.gguf"), mmproj_path=Path("mmproj.gguf"))
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 0

    def test_without_mmproj_calls_find(self, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import load_vision_llama

        handler_cls = mock.MagicMock()
        mock_llama_cpp.llama_chat_format.Llava15ChatHandler = handler_cls
        cfg.num_ctx = None

        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.find_mmproj_for_model",
                return_value=Path("found_mmproj.gguf"),
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider._resolve_vision_handler",
                return_value=handler_cls,
            ),
        ):
            load_vision_llama(Path("model.gguf"))
        assert mock_llama_cpp.Llama.called


# ---------------------------------------------------------------------------
# WorkerProcess None-response paths
# ---------------------------------------------------------------------------


class TestWorkerProcessNoneResponses:
    def test_embed_round_trip_none_retries(self) -> None:
        from lilbee.providers.worker_process import EmbedResponse, WorkerProcess

        wp = WorkerProcess()
        wp._request_queue = mock.MagicMock()
        wp._response_queue = mock.MagicMock()
        wp._process = mock.MagicMock()
        wp._started = True
        wp._next_id = 0

        # First put_and_get returns None (worker died), retry returns valid response.
        with (
            mock.patch.object(
                wp,
                "_put_and_get",
                side_effect=[None, EmbedResponse(vectors=[[0.1]], request_id=1)],
            ),
            mock.patch.object(wp, "restart"),
        ):
            result = wp.embed(["hello"], model="test")
        assert result == [[0.1]]

    def test_embed_round_trip_retry_still_none_raises(self) -> None:
        from lilbee.providers.worker_process import WorkerProcess

        wp = WorkerProcess()
        wp._request_queue = mock.MagicMock()
        wp._response_queue = mock.MagicMock()
        wp._process = mock.MagicMock()
        wp._started = True

        with (
            mock.patch.object(wp, "_put_and_get", return_value=None),
            mock.patch.object(wp, "restart"),
            pytest.raises(RuntimeError, match="crashed again"),
        ):
            wp.embed(["hello"], model="test")

    def test_vision_round_trip_none_retries(self) -> None:
        from lilbee.providers.worker_process import VisionResponse, WorkerProcess

        wp = WorkerProcess()
        wp._request_queue = mock.MagicMock()
        wp._response_queue = mock.MagicMock()
        wp._process = mock.MagicMock()
        wp._started = True
        wp._next_id = 0

        with (
            mock.patch.object(
                wp,
                "_put_and_get",
                side_effect=[None, VisionResponse(text="ocr result", request_id=1)],
            ),
            mock.patch.object(wp, "restart"),
        ):
            result = wp.vision_ocr(b"\x89PNG", model="vis")
        assert result == "ocr result"

    def test_vision_round_trip_retry_still_none_raises(self) -> None:
        from lilbee.providers.worker_process import WorkerProcess

        wp = WorkerProcess()
        wp._request_queue = mock.MagicMock()
        wp._response_queue = mock.MagicMock()
        wp._process = mock.MagicMock()
        wp._started = True

        with (
            mock.patch.object(wp, "_put_and_get", return_value=None),
            mock.patch.object(wp, "restart"),
            pytest.raises(RuntimeError, match="crashed again"),
        ):
            wp.vision_ocr(b"\x89PNG", model="vis")

    def test_get_response_dead_worker_returns_none(self) -> None:
        from lilbee.providers.worker_process import WorkerProcess

        wp = WorkerProcess()
        wp._response_queue = mock.MagicMock()
        wp._response_queue.get.side_effect = Exception("empty")
        wp._process = mock.MagicMock()
        wp._process.is_alive.return_value = False

        result = wp._get_response(timeout=0.5)
        assert result is None

    def test_load_model_sends_request(self) -> None:
        from lilbee.providers.worker_process import LoadModelRequest, WorkerProcess

        wp = WorkerProcess()
        wp._request_queue = mock.MagicMock()
        wp._started = True
        wp._process = mock.MagicMock()
        wp._process.is_alive.return_value = True

        with mock.patch.object(wp, "_ensure_started"):
            wp.load_model("test-model", "embed")
        args = wp._request_queue.put.call_args[0][0]
        assert isinstance(args, LoadModelRequest)
        assert args.model == "test-model"


# ---------------------------------------------------------------------------
# LLMOptions / filter_options
# ---------------------------------------------------------------------------


class TestLLMOptions:
    def test_to_dict_omits_none(self) -> None:
        from lilbee.providers.base import LLMOptions

        opts = LLMOptions(temperature=0.7, top_p=None)
        result = opts.to_dict()
        assert result == {"temperature": 0.7}
        assert "top_p" not in result

    def test_to_dict_all_set(self) -> None:
        from lilbee.providers.base import LLMOptions

        opts = LLMOptions(temperature=0.5, seed=42)
        result = opts.to_dict()
        assert result["temperature"] == 0.5
        assert result["seed"] == 42


class TestFilterOptions:
    def test_filters_valid_options(self) -> None:
        from lilbee.providers.base import filter_options

        result = filter_options({"temperature": 0.8, "seed": 42})
        assert result == {"temperature": 0.8, "seed": 42}

    def test_strips_none_values(self) -> None:
        from lilbee.providers.base import filter_options

        result = filter_options({"temperature": 0.5})
        assert "top_p" not in result


# ---------------------------------------------------------------------------
# LlamaCppProvider methods (bypassing __init__ daemon thread)
# ---------------------------------------------------------------------------


def _make_provider_no_thread() -> object:
    """Create a LlamaCppProvider without starting the embed/rerank thread."""
    from lilbee.providers.llama_cpp_provider import LlamaCppProvider

    with mock.patch("threading.Thread.start"):
        provider = LlamaCppProvider()
    provider._cache = mock.MagicMock()
    provider._embed_thread = mock.MagicMock()
    provider._rerank_thread = mock.MagicMock()
    return provider


class TestLlamaCppProviderMethods:
    def test_get_chat_llm_non_vision(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_get_chat_llm loads via cache for non-vision models."""
        provider = _make_provider_no_thread()
        cfg.chat_model = "test-model"

        mock_cache_model = mock.MagicMock()
        provider._cache.load_model.return_value = mock_cache_model

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            return_value=Path("/models/test.gguf"),
        ):
            result = provider._get_chat_llm()

        assert result == mock_cache_model
        provider._cache.load_model.assert_called_once_with(Path("/models/test.gguf"), mode="chat")

    def test_get_chat_llm_does_not_route_vision_models(
        self, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """_get_chat_llm loads the chat model directly; vision path is separate."""
        provider = _make_provider_no_thread()
        cfg.chat_model = "vision-model"

        mock_cache_model = mock.MagicMock()
        provider._cache.load_model.return_value = mock_cache_model

        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=Path("/models/vision.gguf"),
            ),
            mock.patch.object(provider, "_get_vision_llm") as mock_vis,
        ):
            result = provider._get_chat_llm()

        mock_vis.assert_not_called()
        assert result == mock_cache_model
        provider._cache.load_model.assert_called_once_with(Path("/models/vision.gguf"), mode="chat")

    def test_get_chat_llm_with_override_model(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_get_chat_llm uses the override model when provided."""
        provider = _make_provider_no_thread()
        cfg.chat_model = "default-model"

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            return_value=Path("/models/override.gguf"),
        ):
            provider._get_chat_llm(model="override-model")

        provider._cache.load_model.assert_called_once_with(
            Path("/models/override.gguf"), mode="chat"
        )

    def test_get_vision_llm_caches(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_get_vision_llm caches the vision model."""
        provider = _make_provider_no_thread()

        mock_vis = mock.MagicMock()
        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=Path("/models/vis.gguf"),
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider.load_vision_llama",
                return_value=mock_vis,
            ),
        ):
            result = provider._get_vision_llm("vis-model")

        assert result == mock_vis
        assert provider._vision_llm == mock_vis

    def test_get_vision_llm_reuses_cache(
        self, mock_llama_cpp: mock.MagicMock, tmp_path: Path
    ) -> None:
        """_get_vision_llm reuses cached model for same path."""
        provider = _make_provider_no_thread()
        vis_path = tmp_path / "models" / "vis.gguf"
        existing_vis = mock.MagicMock()
        provider._vision_llm = existing_vis
        provider._vision_model_path = str(vis_path)

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            return_value=vis_path,
        ):
            result = provider._get_vision_llm("vis-model")

        assert result is existing_vis

    def test_get_embed_llm(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_get_embed_llm loads embedding model via cache."""
        provider = _make_provider_no_thread()
        cfg.embedding_model = "embed-model"

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            return_value=Path("/models/embed.gguf"),
        ):
            provider._get_embed_llm()

        provider._cache.load_model.assert_called_once_with(Path("/models/embed.gguf"), mode="embed")

    def test_get_subprocess_worker(self) -> None:
        """_get_subprocess_worker lazy-creates a WorkerProcess."""
        provider = _make_provider_no_thread()

        with mock.patch("lilbee.providers.worker_process.WorkerProcess") as mock_wp_cls:
            result = provider._get_subprocess_worker()

        assert result == mock_wp_cls.return_value
        assert provider._subprocess_worker == mock_wp_cls.return_value

    def test_chat_non_stream_with_options(self, mock_llama_cpp: mock.MagicMock) -> None:
        """chat() with options filters and renames num_predict to max_tokens."""
        provider = _make_provider_no_thread()

        mock_llm = mock.MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }

        with mock.patch.object(provider, "_get_chat_llm", return_value=mock_llm):
            result = provider.chat(
                [{"role": "user", "content": "hi"}],
                stream=False,
                options={"temperature": 0.5, "num_predict": 100},
            )

        assert result == "response"
        call_kwargs = mock_llm.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert "num_predict" not in call_kwargs

    def test_chat_strips_num_ctx(self, mock_llama_cpp: mock.MagicMock) -> None:
        """chat() strips num_ctx since it is a model-load param, not per-call."""
        provider = _make_provider_no_thread()

        mock_llm = mock.MagicMock()
        mock_llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.object(provider, "_get_chat_llm", return_value=mock_llm):
            result = provider.chat(
                [{"role": "user", "content": "hi"}],
                stream=False,
                options={"temperature": 0.7, "num_ctx": 2048},
            )

        assert result == "ok"
        call_kwargs = mock_llm.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert "num_ctx" not in call_kwargs

    def test_chat_non_stream_no_options(self, mock_llama_cpp: mock.MagicMock) -> None:
        """chat() without options passes no extra kwargs."""
        provider = _make_provider_no_thread()

        mock_llm = mock.MagicMock()
        mock_llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.object(provider, "_get_chat_llm", return_value=mock_llm):
            result = provider.chat(
                [{"role": "user", "content": "hi"}],
                stream=False,
            )

        assert result == "ok"

    def test_chat_stream(self, mock_llama_cpp: mock.MagicMock) -> None:
        """chat() with stream=True returns a _LockedStreamIterator."""
        from lilbee.providers.llama_cpp_provider import _LockedStreamIterator

        provider = _make_provider_no_thread()

        mock_llm = mock.MagicMock()
        mock_response = iter([])
        mock_llm.create_chat_completion.return_value = mock_response

        with mock.patch.object(provider, "_get_chat_llm", return_value=mock_llm):
            result = provider.chat(
                [{"role": "user", "content": "hi"}],
                stream=True,
            )

        assert isinstance(result, _LockedStreamIterator)
        # Lock should still be held (released by iterator)
        result.close()

    def test_pull_model_raises(self) -> None:
        """pull_model always raises NotImplementedError."""
        provider = _make_provider_no_thread()
        with pytest.raises(NotImplementedError, match="cannot pull model"):
            provider.pull_model("some-model")

    def test_show_model_returns_metadata(self, mock_llama_cpp: mock.MagicMock) -> None:
        """show_model returns metadata from read_gguf_metadata."""
        provider = _make_provider_no_thread()

        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=Path("/models/test.gguf"),
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider.read_gguf_metadata",
                return_value={"architecture": "llama"},
            ),
        ):
            result = provider.show_model("test-model")

        assert result == {"architecture": "llama"}

    def test_show_model_returns_none_on_error(self) -> None:
        """show_model returns None when model not found."""
        from lilbee.providers.base import ProviderError

        provider = _make_provider_no_thread()

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            side_effect=ProviderError("not found"),
        ):
            result = provider.show_model("missing-model")

        assert result is None

    def test_get_capabilities_with_mmproj(self) -> None:
        """get_capabilities returns ['completion', 'vision'] when mmproj found."""
        provider = _make_provider_no_thread()

        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=Path("/models/llava.gguf"),
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider.find_mmproj_for_model",
                return_value=Path("/models/llava-mmproj.gguf"),
            ),
        ):
            caps = provider.get_capabilities("llava:7b")

        assert "completion" in caps
        assert "vision" in caps

    def test_get_capabilities_rerank_model_short_circuits(self) -> None:
        """A rerank catalog ref returns ``["rerank"]`` without reaching resolve_model_path."""
        from lilbee.catalog import FEATURED_RERANK

        provider = _make_provider_no_thread()
        assert FEATURED_RERANK, "catalog must have at least one rerank entry"
        ref = FEATURED_RERANK[0].name

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            side_effect=AssertionError("resolve_model_path must not be called for a rerank ref"),
        ):
            caps = provider.get_capabilities(ref)

        assert caps == ["rerank"]

    def test_get_capabilities_no_mmproj(self) -> None:
        """get_capabilities returns ['completion'] when no mmproj found."""
        from lilbee.providers.base import ProviderError

        provider = _make_provider_no_thread()

        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=Path("/models/qwen.gguf"),
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider.find_mmproj_for_model",
                side_effect=ProviderError("no mmproj"),
            ),
        ):
            caps = provider.get_capabilities("qwen:8b")

        assert caps == ["completion"]

    def test_get_capabilities_resolve_error(self) -> None:
        """get_capabilities returns ['completion'] when model path not found."""
        from lilbee.providers.base import ProviderError

        provider = _make_provider_no_thread()

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            side_effect=ProviderError("not found"),
        ):
            caps = provider.get_capabilities("missing-model")

        assert caps == ["completion"]

    def test_list_models(self) -> None:
        """list_models returns sorted registry models."""
        provider = _make_provider_no_thread()

        mock_manifest1 = mock.MagicMock()
        mock_manifest1.name = "beta"
        mock_manifest1.tag = "latest"
        mock_manifest2 = mock.MagicMock()
        mock_manifest2.name = "alpha"
        mock_manifest2.tag = "latest"

        mock_registry = mock.MagicMock()
        mock_registry.list_installed.return_value = [mock_manifest1, mock_manifest2]
        mock_services = mock.MagicMock()
        mock_services.registry = mock_registry

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.get_services", return_value=mock_services
        ):
            result = provider.list_models()

        assert result == ["alpha:latest", "beta:latest"]

    def test_shutdown(self) -> None:
        """shutdown stops embed thread, subprocess worker, and cache."""
        provider = _make_provider_no_thread()
        mock_subprocess = mock.MagicMock()
        provider._subprocess_worker = mock_subprocess

        provider.shutdown()

        provider._embed_thread.join.assert_called_once_with(timeout=2)
        mock_subprocess.stop.assert_called_once()
        assert provider._subprocess_worker is None
        provider._cache.unload_all.assert_called_once()

    def test_embed_subprocess_enabled(self) -> None:
        """embed delegates to subprocess worker when enabled."""
        provider = _make_provider_no_thread()
        provider._subprocess_enabled = True

        mock_worker = mock.MagicMock()
        mock_worker.embed.return_value = [[0.1, 0.2]]

        with mock.patch.object(provider, "_get_subprocess_worker", return_value=mock_worker):
            result = provider.embed(["hello"])

        assert result == [[0.1, 0.2]]

    def test_embed_subprocess_fallback(self) -> None:
        """embed falls back to in-process on subprocess failure."""
        from concurrent.futures import Future

        provider = _make_provider_no_thread()
        provider._subprocess_enabled = True

        mock_worker = mock.MagicMock()
        mock_worker.embed.side_effect = OSError("worker crashed")

        fut: Future[list[list[float]]] = Future()
        fut.set_result([[0.3, 0.4]])

        with mock.patch.object(provider, "_get_subprocess_worker", return_value=mock_worker):
            # The embed will try subprocess, fail, then queue in-process
            # We need to handle the queue - put a pre-resolved future

            def intercept_put(req: object) -> None:
                req.future.set_result([[0.3, 0.4]])

            provider._embed_queue.put = intercept_put
            result = provider.embed(["hello"])

        assert result == [[0.3, 0.4]]
        assert provider._subprocess_enabled is False

    def test_vision_ocr(self) -> None:
        """vision_ocr delegates to subprocess worker."""
        provider = _make_provider_no_thread()

        mock_worker = mock.MagicMock()
        mock_worker.vision_ocr.return_value = "OCR result"

        with mock.patch.object(provider, "_get_subprocess_worker", return_value=mock_worker):
            result = provider.vision_ocr(b"\x89PNG", "vis-model", "extract text")

        mock_worker.vision_ocr.assert_called_once_with(b"\x89PNG", "vis-model", "extract text")
        assert result == "OCR result"


class TestEmbedWorker:
    def test_embed_worker_dispatches_batch(self) -> None:
        """_embed_worker processes items and dispatches them."""
        from concurrent.futures import Future

        from lilbee.providers.llama_cpp_provider import LlamaCppProvider, _EmbedRequest

        with mock.patch("threading.Thread.start"):
            provider = LlamaCppProvider()
        provider._cache = mock.MagicMock()

        # Clear the queue and put a request + shutdown sentinel
        while not provider._embed_queue.empty():
            provider._embed_queue.get_nowait()

        fut: Future[list[list[float]]] = Future()
        provider._embed_queue.put(_EmbedRequest(texts=["hello"], future=fut))
        provider._embed_queue.put(None)  # shutdown signal

        with mock.patch.object(provider, "_dispatch_batch") as mock_dispatch:
            provider._embed_worker()

        assert mock_dispatch.called
        batch = mock_dispatch.call_args[0][0]
        assert len(batch) == 1
        assert batch[0].texts == ["hello"]

    def test_embed_worker_shutdown_during_batch(self) -> None:
        """_embed_worker exits when None received during batching."""
        from concurrent.futures import Future

        from lilbee.providers.llama_cpp_provider import LlamaCppProvider, _EmbedRequest

        with mock.patch("threading.Thread.start"):
            provider = LlamaCppProvider()
        provider._cache = mock.MagicMock()

        # Clear the queue and put a request + shutdown
        while not provider._embed_queue.empty():
            provider._embed_queue.get_nowait()

        fut: Future[list[list[float]]] = Future()
        provider._embed_queue.put(_EmbedRequest(texts=["a"], future=fut))
        # After first item, put shutdown while batching
        provider._embed_queue.put(None)

        with mock.patch.object(provider, "_dispatch_batch") as mock_dispatch:
            provider._embed_worker()
        mock_dispatch.assert_called_once()

    def test_dispatch_batch_success(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_dispatch_batch resolves futures with embedding vectors."""
        from concurrent.futures import Future

        from lilbee.providers.llama_cpp_provider import _EmbedRequest

        provider = _make_provider_no_thread()
        mock_llm = mock.MagicMock()
        mock_llm.create_embedding.return_value = {"data": [{"embedding": [0.1]}]}
        provider._cache.load_model.return_value = mock_llm

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.resolve_model_path",
            return_value=Path("/test.gguf"),
        ):
            cfg.embedding_model = "test"
            fut: Future[list[list[float]]] = Future()
            batch = [_EmbedRequest(texts=["hello"], future=fut)]
            provider._dispatch_batch(batch)

        assert fut.result() == [[0.1]]

    def test_dispatch_batch_error(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_dispatch_batch sets exception on future when embed fails."""
        from concurrent.futures import Future

        from lilbee.providers.llama_cpp_provider import _EmbedRequest

        provider = _make_provider_no_thread()
        mock_llm = mock.MagicMock()
        provider._cache.load_model.return_value = mock_llm

        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=Path("/test.gguf"),
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider.embed_one",
                side_effect=RuntimeError("embed broken"),
            ),
        ):
            cfg.embedding_model = "test"
            fut: Future[list[list[float]]] = Future()
            batch = [_EmbedRequest(texts=["hello"], future=fut)]
            provider._dispatch_batch(batch)

        with pytest.raises(RuntimeError, match="embed broken"):
            fut.result()


class TestDispatchBatchGetEmbedLlmError:
    def test_get_embed_llm_failure_sets_exception_on_all_futures(
        self, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """When _get_embed_llm raises, all futures in the batch get the exception."""
        from concurrent.futures import Future

        from lilbee.providers.llama_cpp_provider import _EmbedRequest

        provider = _make_provider_no_thread()

        with mock.patch.object(
            provider, "_get_embed_llm", side_effect=RuntimeError("model not found")
        ):
            fut1: Future[list[list[float]]] = Future()
            fut2: Future[list[list[float]]] = Future()
            batch = [
                _EmbedRequest(texts=["a"], future=fut1),
                _EmbedRequest(texts=["b"], future=fut2),
            ]
            provider._dispatch_batch(batch)

        with pytest.raises(RuntimeError, match="model not found"):
            fut1.result()
        with pytest.raises(RuntimeError, match="model not found"):
            fut2.result()


class TestLockedStreamIteratorException:
    def test_next_releases_lock_on_exception(self) -> None:
        """_LockedStreamIterator releases lock when inner stream raises."""
        import threading

        from lilbee.providers.llama_cpp_provider import _LockedStreamIterator

        lock = threading.Lock()
        lock.acquire()

        def bad_stream() -> Iterator[str]:
            """Generator that raises immediately."""
            yield ""  # make it a generator
            raise RuntimeError("stream error")

        gen = bad_stream()
        next(gen)  # advance past the yield to prime the generator
        it = _LockedStreamIterator(gen, lock)
        with pytest.raises(RuntimeError, match="stream error"):
            next(it)

        # Lock should be released
        assert lock.acquire(timeout=0.1)
        lock.release()


class TestReadGgufMetadata:
    def test_reads_all_fields(self, mock_llama_cpp: mock.MagicMock) -> None:
        """read_gguf_metadata returns parsed fields."""
        from lilbee.providers.llama_cpp_provider import read_gguf_metadata

        mock_llm = mock.MagicMock()
        mock_llm.metadata = {
            "general.architecture": "llama",
            "llama.context_length": 4096,
            "llama.embedding_length": 4096,
            "tokenizer.chat_template": "template",
            "general.file_type": "7",
            "general.name": "Test Model",
        }
        mock_llama_cpp.Llama.return_value = mock_llm

        result = read_gguf_metadata(Path("/test.gguf"))

        assert result == {
            "architecture": "llama",
            "context_length": "4096",
            "embedding_length": "4096",
            "chat_template": "template",
            "file_type": "7",
            "name": "Test Model",
        }
        mock_llm.close.assert_called_once()

    def test_returns_none_for_empty_metadata(self, mock_llama_cpp: mock.MagicMock) -> None:
        """read_gguf_metadata returns None when no fields found."""
        from lilbee.providers.llama_cpp_provider import read_gguf_metadata

        mock_llm = mock.MagicMock()
        mock_llm.metadata = {}
        mock_llama_cpp.Llama.return_value = mock_llm

        result = read_gguf_metadata(Path("/test.gguf"))
        assert result is None

    def test_handles_none_metadata(self, mock_llama_cpp: mock.MagicMock) -> None:
        """read_gguf_metadata handles None metadata."""
        from lilbee.providers.llama_cpp_provider import read_gguf_metadata

        mock_llm = mock.MagicMock()
        mock_llm.metadata = None
        mock_llama_cpp.Llama.return_value = mock_llm

        result = read_gguf_metadata(Path("/test.gguf"))
        assert result is None


class TestLoadLlama:
    def test_embedding_with_ctx0_reads_metadata(self, mock_llama_cpp: mock.MagicMock) -> None:
        """load_llama for embeddings reads context_length from GGUF metadata."""
        from lilbee.providers.llama_cpp_provider import load_llama

        cfg.num_ctx = None

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.read_gguf_metadata",
            return_value={"context_length": "2048"},
        ):
            load_llama(Path("/test.gguf"), mode="embed")

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_batch"] == 2048
        assert call_kwargs["n_ubatch"] == 2048
        assert call_kwargs["n_ctx"] == 0
        assert call_kwargs["embedding"] is True

    def test_embedding_no_metadata(self, mock_llama_cpp: mock.MagicMock) -> None:
        """load_llama defaults to 2048 when no metadata."""
        from lilbee.providers.llama_cpp_provider import load_llama

        cfg.num_ctx = None

        with mock.patch(
            "lilbee.providers.llama_cpp_provider.read_gguf_metadata",
            return_value=None,
        ):
            load_llama(Path("/test.gguf"), mode="embed")

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_batch"] == 2048

    def test_embedding_with_explicit_ctx(self, mock_llama_cpp: mock.MagicMock) -> None:
        """load_llama with explicit num_ctx uses it for n_batch."""
        from lilbee.providers.llama_cpp_provider import load_llama

        cfg.num_ctx = 4096

        load_llama(Path("/test.gguf"), mode="embed")

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 4096
        assert call_kwargs["n_batch"] == 4096

    def test_chat_mode(self, mock_llama_cpp: mock.MagicMock) -> None:
        """load_llama for chat does not set n_batch."""
        from lilbee.providers.llama_cpp_provider import load_llama

        cfg.num_ctx = None

        load_llama(Path("/test.gguf"), mode="chat")

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["embedding"] is False
        assert "n_batch" not in call_kwargs


class TestFindMmprojForModel:
    def test_catalog_lookup(self) -> None:
        """find_mmproj_for_model uses catalog lookup first."""
        from lilbee.providers.llama_cpp_provider import find_mmproj_for_model

        with mock.patch(
            "lilbee.catalog.find_mmproj_file",
            return_value=Path("/found.gguf"),
        ):
            result = find_mmproj_for_model(Path("/models/model.gguf"))

        assert result == Path("/found.gguf")

    def test_directory_fallback(self, tmp_path: Path) -> None:
        """find_mmproj_for_model falls back to directory scan."""
        from lilbee.providers.llama_cpp_provider import find_mmproj_for_model

        model_path = tmp_path / "model.gguf"
        model_path.touch()
        mmproj = tmp_path / "model-mmproj-fp16.gguf"
        mmproj.touch()

        with mock.patch(
            "lilbee.catalog.find_mmproj_file",
            return_value=None,
        ):
            result = find_mmproj_for_model(model_path)

        assert result == mmproj

    def test_raises_when_not_found(self, tmp_path: Path) -> None:
        """find_mmproj_for_model raises ProviderError when no mmproj found."""
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import find_mmproj_for_model

        model_path = tmp_path / "model.gguf"
        model_path.touch()

        with (
            mock.patch(
                "lilbee.catalog.find_mmproj_file",
                return_value=None,
            ),
            pytest.raises(ProviderError, match="No mmproj"),
        ):
            find_mmproj_for_model(model_path)

    def test_hf_cache_blob_walks_to_snapshots(self, tmp_path: Path) -> None:
        """HF cache resolves main GGUFs to blob paths; the mmproj lives next to the
        snapshot symlink, not in blobs/. find_mmproj_for_model must walk up to the
        sibling snapshots/ tree when the model path lives under blobs/.
        """
        from lilbee.providers.llama_cpp_provider import find_mmproj_for_model

        model_root = tmp_path / "models--org--Repo-GGUF"
        blobs = model_root / "blobs"
        snap = model_root / "snapshots" / "abc123"
        blobs.mkdir(parents=True)
        snap.mkdir(parents=True)
        blob_path = blobs / "deadbeef"
        blob_path.touch()
        (snap / "mmproj-f16.gguf").touch()

        with mock.patch("lilbee.catalog.find_mmproj_file", return_value=None):
            result = find_mmproj_for_model(blob_path)
        assert result == snap / "mmproj-f16.gguf"

    def test_hf_cache_blob_without_snapshots_falls_through(self, tmp_path: Path) -> None:
        """blobs/ dir with no sibling snapshots/ tree returns None from the HF
        helper, allowing the flat-dir fallback to take over."""
        from lilbee.providers.llama_cpp_provider import find_mmproj_for_model

        model_root = tmp_path / "models--org--Repo-GGUF"
        blobs = model_root / "blobs"
        blobs.mkdir(parents=True)
        blob_path = blobs / "deadbeef"
        blob_path.touch()
        # No snapshots/ sibling and no mmproj in blobs/.

        from lilbee.providers.base import ProviderError

        with (
            mock.patch("lilbee.catalog.find_mmproj_file", return_value=None),
            pytest.raises(ProviderError, match="No mmproj"),
        ):
            find_mmproj_for_model(blob_path)

    def test_hf_cache_snapshots_without_mmproj_falls_through(self, tmp_path: Path) -> None:
        """snapshots/ tree exists but no mmproj GGUF lives in any snapshot —
        the HF helper returns None and the flat-dir fallback takes over."""
        from lilbee.providers.llama_cpp_provider import find_mmproj_for_model

        model_root = tmp_path / "models--org--Repo-GGUF"
        blobs = model_root / "blobs"
        snap = model_root / "snapshots" / "abc123"
        blobs.mkdir(parents=True)
        snap.mkdir(parents=True)
        blob_path = blobs / "deadbeef"
        blob_path.touch()
        # snapshot dir is present but contains no *mmproj*.gguf files.

        from lilbee.providers.base import ProviderError

        with (
            mock.patch("lilbee.catalog.find_mmproj_file", return_value=None),
            pytest.raises(ProviderError, match="No mmproj"),
        ):
            find_mmproj_for_model(blob_path)


class TestReadMmprojProjectorTypePartial:
    def test_returns_projector_type(self, tmp_path: Path) -> None:
        """read_mmproj_projector_type reads clip.projector_type from GGUF."""
        import struct

        from lilbee.providers.llama_cpp_provider import read_mmproj_projector_type

        # Build a minimal GGUF with one KV pair: clip.projector_type = "ldp"
        f = tmp_path / "test.gguf"
        with open(f, "wb") as fp:
            fp.write(b"GGUF")  # magic
            fp.write(struct.pack("<I", 3))  # version
            fp.write(struct.pack("<Q", 0))  # tensor count
            fp.write(struct.pack("<Q", 1))  # kv count
            key = b"clip.projector_type"
            fp.write(struct.pack("<Q", len(key)))
            fp.write(key)
            fp.write(struct.pack("<I", 8))  # type 8 = string
            value = b"ldp"
            fp.write(struct.pack("<Q", len(value)))
            fp.write(value)

        result = read_mmproj_projector_type(f)
        assert result == "ldp"

    def test_skips_non_matching_keys(self, tmp_path: Path) -> None:
        """read_mmproj_projector_type skips unrelated keys."""
        import struct

        from lilbee.providers.llama_cpp_provider import read_mmproj_projector_type

        f = tmp_path / "test.gguf"
        with open(f, "wb") as fp:
            fp.write(b"GGUF")
            fp.write(struct.pack("<I", 3))
            fp.write(struct.pack("<Q", 0))
            fp.write(struct.pack("<Q", 2))  # 2 kv pairs
            # First KV: other.key = "value" (string)
            key1 = b"other.key"
            fp.write(struct.pack("<Q", len(key1)))
            fp.write(key1)
            fp.write(struct.pack("<I", 8))  # string type
            val1 = b"value"
            fp.write(struct.pack("<Q", len(val1)))
            fp.write(val1)
            # Second KV: clip.projector_type = "resampler"
            key2 = b"clip.projector_type"
            fp.write(struct.pack("<Q", len(key2)))
            fp.write(key2)
            fp.write(struct.pack("<I", 8))
            val2 = b"resampler"
            fp.write(struct.pack("<Q", len(val2)))
            fp.write(val2)

        result = read_mmproj_projector_type(f)
        assert result == "resampler"


class TestIsOllama:
    def test_localhost_default_port(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("http://localhost:11434") is True

    def test_127_default_port(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("http://127.0.0.1:11434") is True

    def test_ollama_in_url(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("https://ollama.example.com") is True

    def test_openai_url(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("https://api.openai.com") is False

    def test_custom_url(self) -> None:
        from lilbee.providers.litellm_sdk import _is_ollama

        assert _is_ollama("http://myserver:8080") is False


class TestRouteModel:
    """Wire-format routing lives in ``LitellmSdkBackend._route_model``.

    These tests exercise the helper directly so we do not depend on the
    internal composition of ``SdkLLMProvider`` + backend.
    """

    def test_ollama_url_adds_prefix(self) -> None:
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("qwen3:8b")
        assert _route_model(ref, "http://localhost:11434") == "ollama/qwen3:8b"

    def test_non_ollama_url_no_prefix(self) -> None:
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("gpt-4o")
        # No Ollama hint and no API provider prefix: preserve raw name.
        assert _route_model(ref, "https://api.openai.com") == "gpt-4o:latest"

    def test_already_prefixed_passes_through(self) -> None:
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("openai/gpt-4o")
        assert _route_model(ref, "http://localhost:11434") == "openai/gpt-4o"

    def test_provider_prefixed_on_non_ollama(self) -> None:
        from lilbee.providers.litellm_sdk import _route_model
        from lilbee.providers.model_ref import parse_model_ref

        ref = parse_model_ref("anthropic/claude-sonnet-4-6")
        assert _route_model(ref, "https://api.anthropic.com") == "anthropic/claude-sonnet-4-6"


class TestInjectProviderKeys:
    def test_injects_keys_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.sdk_llm_provider import inject_provider_keys

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg.openai_api_key = "sk-test-openai"
        cfg.anthropic_api_key = "sk-test-anthropic"
        cfg.gemini_api_key = ""

        inject_provider_keys()

        import os

        assert os.environ.get("OPENAI_API_KEY") == "sk-test-openai"
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-test-anthropic"
        assert os.environ.get("GEMINI_API_KEY") is None

    def test_does_not_override_existing_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.providers.sdk_llm_provider import inject_provider_keys

        monkeypatch.setenv("OPENAI_API_KEY", "sk-existing")
        cfg.openai_api_key = "sk-from-config"

        inject_provider_keys()

        import os

        assert os.environ["OPENAI_API_KEY"] == "sk-existing"


class TestLiteLLMListModelsRouting:
    def test_ollama_url_uses_api_tags(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.get", return_value=mock_resp) as mock_get:
            result = provider.list_models()

        mock_get.assert_called_once()
        assert "api/tags" in mock_get.call_args[0][0]
        assert result == ["llama3:8b"]

    def test_non_ollama_url_uses_v1_models(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(
            LitellmSdkBackend(), base_url="https://api.openai.com", api_key="sk-test"
        )
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.get", return_value=mock_resp) as mock_get:
            result = provider.list_models()

        mock_get.assert_called_once()
        assert "v1/models" in mock_get.call_args[0][0]
        assert result == ["gpt-4o", "gpt-4o-mini"]

    def test_non_ollama_returns_empty_on_error(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="https://api.openai.com")

        with mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            result = provider.list_models()

        assert result == []

    def test_v1_models_sends_auth_header(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(
            LitellmSdkBackend(), base_url="https://api.openai.com", api_key="sk-secret"
        )
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.get", return_value=mock_resp) as mock_get:
            provider.list_models()

        headers = mock_get.call_args[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-secret"


class TestNeedsApiBase:
    def test_ollama_prefixed_model_needs_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("ollama/qwen3:8b").needs_api_base is True

    def test_bare_model_needs_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("qwen3:8b").needs_api_base is True

    def test_openai_prefixed_model_skips_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("openai/gpt-4o").needs_api_base is False

    def test_anthropic_prefixed_model_skips_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("anthropic/claude-sonnet-4-6").needs_api_base is False

    def test_gemini_prefixed_model_skips_api_base(self) -> None:
        from lilbee.providers.model_ref import parse_model_ref

        assert parse_model_ref("gemini/gemini-pro").needs_api_base is False


class TestChatApiBaseRouting:
    """Verify that chat() omits api_base for non-Ollama provider-prefixed models."""

    def _make_fake_litellm(self) -> mock.MagicMock:
        fake = mock.MagicMock()
        resp = mock.MagicMock()
        resp.choices = [mock.MagicMock()]
        resp.choices[0].message.content = "hello"
        fake.completion.return_value = resp
        return fake

    def test_ollama_model_passes_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        fake = self._make_fake_litellm()

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.chat([{"role": "user", "content": "hi"}], model="qwen3:0.6b")

        call_kwargs = fake.completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"
        assert call_kwargs["model"] == "ollama/qwen3:0.6b"

    def test_frontier_model_omits_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        fake = self._make_fake_litellm()

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")

        call_kwargs = fake.completion.call_args[1]
        assert "api_base" not in call_kwargs
        assert call_kwargs["model"] == "openai/gpt-4o"

    def test_anthropic_model_omits_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        fake = self._make_fake_litellm()

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.chat([{"role": "user", "content": "hi"}], model="anthropic/claude-sonnet-4-6")

        call_kwargs = fake.completion.call_args[1]
        assert "api_base" not in call_kwargs

    def test_chat_calls_inject_provider_keys(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        fake = self._make_fake_litellm()

        with (
            mock.patch.dict("sys.modules", {"litellm": fake}),
            mock.patch("lilbee.providers.sdk_llm_provider.inject_provider_keys") as mock_inject,
        ):
            provider.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")

        mock_inject.assert_called_once()


class TestEmbedApiBaseRouting:
    """Verify that embed() omits api_base for non-Ollama provider-prefixed models."""

    def test_ollama_embed_passes_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        cfg.embedding_model = "nomic-embed-text"
        fake = mock.MagicMock()
        fake.embedding.return_value = {"data": [{"embedding": [0.1, 0.2]}]}

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.embed(["hello"])

        call_kwargs = fake.embedding.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"

    def test_prefixed_embed_omits_api_base(self) -> None:
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        provider = SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        cfg.embedding_model = "openai/text-embedding-3-small"
        fake = mock.MagicMock()
        fake.embedding.return_value = {"data": [{"embedding": [0.1, 0.2]}]}

        with mock.patch.dict("sys.modules", {"litellm": fake}):
            provider.embed(["hello"])

        call_kwargs = fake.embedding.call_args[1]
        assert "api_base" not in call_kwargs


class TestSdkRerank:
    """Coverage for SdkLLMProvider.rerank + LitellmSdkBackend.rerank."""

    def _make_sdk_provider(self):
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider

        return SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")

    def test_rerank_returns_scores_in_candidate_order(self) -> None:
        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = self._make_sdk_provider()
        fake_response = {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.3},
                {"index": 1, "relevance_score": 0.7},
            ]
        }
        fake_litellm = mock.MagicMock()
        fake_litellm.rerank.return_value = fake_response
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            scores = provider.rerank("q", ["a", "b", "c"])
        assert scores == [0.3, 0.7, 0.9]
        kwargs = fake_litellm.rerank.call_args.kwargs
        assert kwargs["query"] == "q"
        assert kwargs["documents"] == ["a", "b", "c"]
        assert "cohere" in kwargs["model"]

    def test_rerank_empty_candidates_short_circuits(self) -> None:
        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = self._make_sdk_provider()
        assert provider.rerank("q", []) == []

    def test_supports_rerank_matches_backend_available(self) -> None:
        provider = self._make_sdk_provider()
        with mock.patch(
            "lilbee.providers.litellm_sdk.litellm_available",
            return_value=True,
        ):
            assert provider.supports_rerank() is True
        with mock.patch(
            "lilbee.providers.litellm_sdk.litellm_available",
            return_value=False,
        ):
            assert provider.supports_rerank() is False

    def test_rerank_maps_backend_error_to_provider_error(self) -> None:
        from lilbee.providers.base import ProviderError

        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = self._make_sdk_provider()
        fake_litellm = mock.MagicMock()
        fake_litellm.rerank.side_effect = RuntimeError("boom")
        with (
            mock.patch.dict(sys.modules, {"litellm": fake_litellm}),
            pytest.raises(ProviderError, match="Rerank failed"),
        ):
            provider.rerank("q", ["a"])

    def test_rerank_accepts_object_style_response(self) -> None:
        """Pydantic-style responses with attribute access also parse."""
        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = self._make_sdk_provider()
        item = mock.MagicMock(index=0, relevance_score=0.42)
        response = mock.MagicMock(results=[item], model="cohere/rerank-english-v3.0")
        fake_litellm = mock.MagicMock()
        fake_litellm.rerank.return_value = response
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            scores = provider.rerank("q", ["a"])
        assert scores == [0.42]

    def test_backend_empty_candidates_skip_sdk(self) -> None:
        """Backend-level empty-candidates guard avoids importing litellm."""
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.model_ref import parse_model_ref
        from lilbee.providers.sdk_backend import RerankRequest

        backend = LitellmSdkBackend()
        request = RerankRequest(
            ref=parse_model_ref("cohere/rerank-english-v3.0"),
            query="q",
            candidates=[],
        )
        result = backend.rerank(request)
        assert result.scores == []

    def test_backend_forwards_api_key(self) -> None:
        """``api_key`` on the request is forwarded to litellm.rerank kwargs."""
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.model_ref import parse_model_ref
        from lilbee.providers.sdk_backend import RerankRequest

        backend = LitellmSdkBackend()
        request = RerankRequest(
            ref=parse_model_ref("cohere/rerank-english-v3.0"),
            query="q",
            candidates=["a"],
            api_base="http://localhost:11434",
            api_key="sk-test",
        )
        fake_litellm = mock.MagicMock()
        fake_litellm.rerank.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend.rerank(request)
        assert fake_litellm.rerank.call_args.kwargs["api_key"] == "sk-test"

    def test_provider_wraps_non_provider_error(self) -> None:
        """A non-``ProviderError`` raised inside the backend is re-wrapped."""
        from lilbee.providers.base import ProviderError
        from lilbee.providers.litellm_sdk import LitellmSdkBackend
        from lilbee.providers.sdk_llm_provider import SdkLLMProvider as _SdkLLMProvider

        cfg.reranker_model = "cohere/rerank-english-v3.0"
        provider = _SdkLLMProvider(LitellmSdkBackend(), base_url="http://localhost:11434")
        with (
            mock.patch.object(provider._backend, "rerank", side_effect=RuntimeError("wire error")),
            pytest.raises(ProviderError, match="Rerank failed: wire error"),
        ):
            provider.rerank("q", ["a"])


class TestIsRerankModel:
    def test_empty_model_returns_false(self) -> None:
        from lilbee.providers.llama_cpp_provider import _is_rerank_model

        assert _is_rerank_model("") is False

    def test_matches_featured_rerank_entry(self) -> None:
        from lilbee.catalog import FEATURED_RERANK
        from lilbee.providers.llama_cpp_provider import _is_rerank_model

        assert FEATURED_RERANK, "catalog must have at least one rerank entry"
        assert _is_rerank_model(FEATURED_RERANK[0].name) is True

    def test_non_rerank_model_returns_false(self) -> None:
        from lilbee.providers.llama_cpp_provider import _is_rerank_model

        assert _is_rerank_model("definitely-not-a-rerank-model") is False

    def test_substring_of_catalog_name_does_not_match(self) -> None:
        from lilbee.providers.llama_cpp_provider import _is_rerank_model

        assert _is_rerank_model("base") is False
        assert _is_rerank_model("reranker") is False

    def test_hf_repo_with_tag_suffix_matches(self) -> None:
        from lilbee.catalog import FEATURED_RERANK
        from lilbee.providers.llama_cpp_provider import _is_rerank_model

        assert FEATURED_RERANK, "catalog must have at least one rerank entry"
        entry = FEATURED_RERANK[0]
        assert _is_rerank_model(f"{entry.hf_repo}:latest") is True
        assert _is_rerank_model(entry.hf_repo) is True


class TestExtractRerankScore:
    def test_raises_when_data_empty(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import _extract_rerank_score

        with pytest.raises(ProviderError, match="no data"):
            _extract_rerank_score({"data": []})

    def test_flat_list_embedding_returns_first_element(self) -> None:
        """llama-cpp-python 0.3.x returns ``list[float]`` with length n_embd=1."""
        from lilbee.providers.llama_cpp_provider import _extract_rerank_score

        assert _extract_rerank_score({"data": [{"embedding": [0.73]}]}) == 0.73

    def test_scalar_embedding_is_unexpected(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import _extract_rerank_score

        with pytest.raises(ProviderError, match=r"unexpected score shape.*float"):
            _extract_rerank_score({"data": [{"embedding": 0.73}]})

    def test_nested_list_embedding_is_unexpected(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import _extract_rerank_score

        with pytest.raises(ProviderError, match=r"unexpected score shape.*list"):
            _extract_rerank_score({"data": [{"embedding": [[0.42]]}]})

    def test_empty_embedding_list_is_unexpected(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import _extract_rerank_score

        with pytest.raises(ProviderError, match=r"unexpected score shape.*list: \[\]"):
            _extract_rerank_score({"data": [{"embedding": []}]})

    def test_non_numeric_embedding_is_unexpected(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import _extract_rerank_score

        with pytest.raises(ProviderError, match="unexpected score shape"):
            _extract_rerank_score({"data": [{"embedding": "not-a-number"}]})


class TestRoutingProviderRerank:
    """Routing-level rerank dispatch between native llama-cpp and hosted SDK."""

    def _make_provider(self):
        from lilbee.providers.routing_provider import RoutingProvider

        return RoutingProvider()

    def test_rerank_routes_hosted_ref_to_sdk(self) -> None:
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_sdk = mock.MagicMock()
        mock_sdk.supports_rerank.return_value = True
        mock_sdk.rerank.return_value = [0.9, 0.1]
        rp._llama_cpp = mock_llama
        rp._sdk_provider = mock_sdk

        cfg.reranker_model = "cohere/rerank-english-v3.0"
        scores = rp.rerank("q", ["a", "b"])
        assert scores == [0.9, 0.1]
        mock_sdk.rerank.assert_called_once_with("q", ["a", "b"])
        mock_llama.rerank.assert_not_called()

    def test_rerank_raises_when_hosted_ref_and_sdk_backend_missing(self) -> None:
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_sdk = mock.MagicMock()
        mock_sdk.supports_rerank.return_value = False
        rp._llama_cpp = mock_llama
        rp._sdk_provider = mock_sdk

        cfg.reranker_model = "cohere/rerank-english-v3.0"
        with pytest.raises(ProviderError, match="hosted rerank backend not available"):
            rp.rerank("q", ["a", "b"])
        mock_sdk.rerank.assert_not_called()

    def test_supports_rerank_disabled_model_always_true(self) -> None:
        rp = self._make_provider()
        cfg.reranker_model = ""
        assert rp.supports_rerank() is True

    def test_supports_rerank_native_delegates_to_llama(self) -> None:
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_llama.supports_rerank.return_value = True
        rp._llama_cpp = mock_llama

        cfg.reranker_model = "bge-reranker-v2-m3:latest"
        assert rp.supports_rerank() is True
        mock_llama.supports_rerank.assert_called_once()

    def test_supports_rerank_hosted_delegates_to_sdk(self) -> None:
        rp = self._make_provider()
        mock_sdk = mock.MagicMock()
        rp._sdk_provider = mock_sdk
        cfg.reranker_model = "cohere/rerank-english-v3.0"

        mock_sdk.supports_rerank.return_value = False
        assert rp.supports_rerank() is False
        mock_sdk.supports_rerank.return_value = True
        assert rp.supports_rerank() is True

    def test_rerank_routes_bare_gguf_to_llama_cpp(self) -> None:
        rp = self._make_provider()
        mock_llama = mock.MagicMock()
        mock_sdk = mock.MagicMock()
        mock_llama.rerank.return_value = [0.5, 0.5]
        rp._llama_cpp = mock_llama
        rp._sdk_provider = mock_sdk

        cfg.reranker_model = "bge-reranker-v2-m3:latest"
        scores = rp.rerank("q", ["a", "b"])
        assert scores == [0.5, 0.5]
        mock_llama.rerank.assert_called_once_with("q", ["a", "b"])
        mock_sdk.rerank.assert_not_called()

    def test_empty_reranker_model_routes_to_litellm(self) -> None:
        """An empty ``cfg.reranker_model`` is treated as non-native (hosted)."""
        from lilbee.providers.routing_provider import _is_native_rerank_ref

        assert _is_native_rerank_ref("") is False

    def test_rerank_with_empty_model_raises_provider_error(self) -> None:
        """rerank() raises ProviderError when cfg.reranker_model is empty."""
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        cfg.reranker_model = ""
        with pytest.raises(ProviderError, match="No reranker configured"):
            rp.rerank("q", ["a", "b"])


class TestLlamaCppRerankDispatchError:
    def test_scoring_error_is_propagated_to_future(
        self, models_dir: Path, mock_llama_cpp: mock.MagicMock
    ) -> None:
        """A failure inside ``compute_rerank_scores`` resolves the future with the error."""
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.reranker_model = "test-model"
        instance = mock.MagicMock()
        instance.create_embedding.side_effect = RuntimeError("boom")
        mock_llama_cpp.Llama.return_value = instance

        provider = LlamaCppProvider()
        try:
            with pytest.raises(RuntimeError, match="boom"):
                provider.rerank("q", ["candidate"])
        finally:
            provider.shutdown()


class TestLlamaCppHasRankPooling:
    def test_has_rank_pooling_reports_import_status(self) -> None:
        from lilbee.providers.llama_cpp_provider import _llama_cpp_has_rank_pooling

        fake_mod = mock.MagicMock()
        fake_mod.LLAMA_POOLING_TYPE_RANK = 4
        with mock.patch.dict(sys.modules, {"llama_cpp": fake_mod}):
            assert _llama_cpp_has_rank_pooling() is True
        with mock.patch.dict("sys.modules", {"llama_cpp": None}):
            assert _llama_cpp_has_rank_pooling() is False

    def test_supports_rerank_requires_rank_pooling(self) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        with mock.patch("threading.Thread.start"):
            provider = LlamaCppProvider()
        try:
            with mock.patch(
                "lilbee.providers.llama_cpp_provider._llama_cpp_has_rank_pooling",
                return_value=True,
            ):
                assert provider.supports_rerank() is True
            with mock.patch(
                "lilbee.providers.llama_cpp_provider._llama_cpp_has_rank_pooling",
                return_value=False,
            ):
                assert provider.supports_rerank() is False
        finally:
            provider._embed_thread = mock.MagicMock()
            provider._rerank_thread = mock.MagicMock()
            provider.shutdown()
