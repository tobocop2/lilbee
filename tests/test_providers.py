"""Tests for the LLM provider abstraction layer (mocked — no live servers needed)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from lilbee.config import cfg

if TYPE_CHECKING:
    from lilbee.providers.routing_provider import RoutingProvider

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
            load_llama(models_dir / "test-model.gguf", embedding=True)
            call_kwargs = mock_llama_cls.call_args[1]
            assert call_kwargs["n_batch"] == 2048
            assert call_kwargs["n_ubatch"] == 2048
            assert call_kwargs["embedding"] is True

    def testload_llama_no_n_batch_for_chat(self, models_dir: Path) -> None:
        from unittest.mock import patch

        from lilbee.providers.llama_cpp_provider import load_llama

        with patch("llama_cpp.Llama"):
            load_llama(models_dir / "test-model.gguf", embedding=False)
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
    def test_default_llm_provider(self) -> None:
        env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("LILBEE_")}
        with (
            mock.patch.dict(__import__("os").environ, env, clear=True),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            from lilbee.config import Config

            c = Config()
            assert c.llm_provider == "auto"
            assert c.litellm_base_url == "http://localhost:11434"
            assert c.llm_api_key == ""

    def test_provider_env_override(self) -> None:
        import os

        with mock.patch.dict(
            os.environ,
            {
                "LILBEE_LLM_PROVIDER": "litellm",
                "LILBEE_LITELLM_BASE_URL": "http://myhost:11434",
                "LILBEE_LLM_API_KEY": "sk-key",
            },
        ):
            from lilbee.config import Config

            c = Config()
            assert c.llm_provider == "litellm"
            assert c.litellm_base_url == "http://myhost:11434"
            assert c.llm_api_key == "sk-key"

    def test_models_dir_is_canonical(self, tmp_path: Path) -> None:
        """models_dir uses canonical system path, not per-project data_root."""
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

    def test_routes_chat_to_litellm_when_available(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.chat.return_value = "hello"
        rp._litellm = mock_litellm
        rp._use_litellm = True

        cfg.chat_model = "qwen3:8b"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "hello"
        mock_litellm.chat.assert_called_once()

    def test_routes_chat_to_llama_cpp_when_litellm_unavailable(self) -> None:
        rp = self._make_provider()
        rp._use_litellm = False

        mock_llama = mock.MagicMock()
        mock_llama.chat.return_value = "local"
        rp._llama_cpp = mock_llama

        cfg.chat_model = "local-model.gguf"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "local"
        mock_llama.chat.assert_called_once()

    def test_routes_embed_to_litellm_when_available(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.embed.return_value = [[0.1, 0.2]]
        rp._litellm = mock_litellm
        rp._use_litellm = True

        result = rp.embed(["test"])
        assert result == [[0.1, 0.2]]
        mock_litellm.embed.assert_called_once()

    def test_routes_embed_to_llama_cpp_when_litellm_unavailable(self) -> None:
        rp = self._make_provider()
        rp._use_litellm = False

        mock_llama = mock.MagicMock()
        mock_llama.embed.return_value = [[0.3, 0.4]]
        rp._llama_cpp = mock_llama

        result = rp.embed(["test"])
        assert result == [[0.3, 0.4]]

    def test_list_models_native_only_when_litellm_unavailable(self) -> None:
        rp = self._make_provider()
        rp._use_litellm = False

        mock_llama = mock.MagicMock()
        mock_llama.list_models.return_value = ["local.gguf"]
        rp._llama_cpp = mock_llama

        result = rp.list_models()
        assert result == ["local.gguf"]

    def test_litellm_unreachable_falls_back_to_llama_cpp(self) -> None:
        rp = self._make_provider()
        rp._use_litellm = False

        mock_llama = mock.MagicMock()
        mock_llama.chat.return_value = "fallback"
        rp._llama_cpp = mock_llama

        cfg.chat_model = "local.gguf"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "fallback"

    def test_show_model_delegates_to_litellm_when_available(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.show_model.return_value = {"parameters": "temp 0.7"}
        rp._litellm = mock_litellm
        rp._use_litellm = True

        result = rp.show_model("qwen3:8b")
        assert result == {"parameters": "temp 0.7"}
        mock_litellm.show_model.assert_called_once_with("qwen3:8b")

    def test_show_model_uses_llama_cpp_when_litellm_unavailable(self) -> None:
        rp = self._make_provider()
        rp._use_litellm = False

        mock_llama = mock.MagicMock()
        mock_llama.show_model.return_value = None
        rp._llama_cpp = mock_llama

        result = rp.show_model("local.gguf")
        assert result is None

    def test_invalidate_cache_clears_detection(self) -> None:
        rp = self._make_provider()
        rp._use_litellm = True

        rp.invalidate_cache()
        assert rp._use_litellm is None

    def test_pull_model_raises_when_litellm_unavailable(self) -> None:
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        rp._use_litellm = False

        with pytest.raises(ProviderError, match="no pull-capable backend"):
            rp.pull_model("bad-model")

    def test_chat_with_explicit_model_override(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.chat.return_value = "saw it"
        rp._litellm = mock_litellm
        rp._use_litellm = True

        cfg.chat_model = "local.gguf"
        result = rp.chat(
            [{"role": "user", "content": "describe"}],
            model="vision:7b",
        )
        assert result == "saw it"
        mock_litellm.chat.assert_called_once()

    def test_litellm_not_installed_skips_remote(self) -> None:
        """When litellm is not installed, routing provider uses llama-cpp."""
        rp = self._make_provider()

        with mock.patch("lilbee.providers.litellm_provider.litellm_available", return_value=False):
            assert rp._should_use_litellm() is False


# ---------------------------------------------------------------------------
# litellm_available guard
# ---------------------------------------------------------------------------


class TestLitellmAvailable:
    def test_returns_false_when_not_installed(self) -> None:
        from lilbee.providers.litellm_provider import litellm_available

        with mock.patch.dict("sys.modules", {"litellm": None}):
            assert litellm_available() is False

    def test_factory_raises_when_litellm_unavailable(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.factory import create_provider
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.llm_provider = "litellm"
        with (
            mock.patch.object(LiteLLMProvider, "available", return_value=False),
            pytest.raises(ProviderError, match="litellm is not installed"),
        ):
            create_provider(cfg)


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


class TestSkipGgufValue:
    def test_string_type(self) -> None:
        import io
        import struct

        from lilbee.providers.llama_cpp_provider import _skip_gguf_value

        data = struct.pack("<Q", 5) + b"hello"
        f = io.BytesIO(data)
        _skip_gguf_value(f, 8)
        assert f.tell() == len(data)

    def test_array_type(self) -> None:
        import io
        import struct

        from lilbee.providers.llama_cpp_provider import _skip_gguf_value

        # array of 2 uint32 elements (type 4, size 4 bytes each)
        data = struct.pack("<I", 4) + struct.pack("<Q", 2) + b"\x00" * 8
        f = io.BytesIO(data)
        _skip_gguf_value(f, 9)
        assert f.tell() == len(data)

    def test_known_type(self) -> None:
        import io

        from lilbee.providers.llama_cpp_provider import _skip_gguf_value

        f = io.BytesIO(b"\x00\x00\x00\x00")
        _skip_gguf_value(f, 4)  # uint32 = 4 bytes
        assert f.tell() == 4

    def test_unknown_type_skips_8_bytes(self) -> None:
        import io

        from lilbee.providers.llama_cpp_provider import _skip_gguf_value

        f = io.BytesIO(b"\x00" * 8)
        _skip_gguf_value(f, 255)  # unknown type
        assert f.tell() == 8


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
    def test_send_and_receive_embed_none_retries(self) -> None:
        from lilbee.providers.worker_process import EmbedRequest, EmbedResponse, WorkerProcess

        wp = WorkerProcess()
        wp._request_queue = mock.MagicMock()
        wp._response_queue = mock.MagicMock()
        wp._process = mock.MagicMock()
        wp._started = True
        wp._next_id = 0

        req = EmbedRequest(texts=["hello"], model="test", request_id=1)
        # First call returns None (worker died), retry returns valid response
        with (
            mock.patch.object(
                wp,
                "_get_response",
                side_effect=[None, EmbedResponse(vectors=[[0.1]], request_id=1)],
            ),
            mock.patch.object(wp, "restart"),
        ):
            result = wp._send_and_receive_embed(req)
        assert result == [[0.1]]

    def test_retry_embed_none_raises(self) -> None:
        from lilbee.providers.worker_process import EmbedRequest, WorkerProcess

        wp = WorkerProcess()
        wp._request_queue = mock.MagicMock()
        wp._response_queue = mock.MagicMock()
        wp._process = mock.MagicMock()
        wp._started = True

        req = EmbedRequest(texts=["hello"], model="test", request_id=1)
        with (
            mock.patch.object(wp, "_get_response", return_value=None),
            mock.patch.object(wp, "restart"),
            pytest.raises(RuntimeError, match="crashed again"),
        ):
            wp._retry_embed(req)

    def test_send_and_receive_vision_none_retries(self) -> None:
        from lilbee.providers.worker_process import VisionRequest, VisionResponse, WorkerProcess

        wp = WorkerProcess()
        wp._request_queue = mock.MagicMock()
        wp._response_queue = mock.MagicMock()
        wp._process = mock.MagicMock()
        wp._started = True
        wp._next_id = 0

        req = VisionRequest(png_bytes=b"\x89PNG", model="vis", prompt="", request_id=1)
        with (
            mock.patch.object(
                wp,
                "_get_response",
                side_effect=[None, VisionResponse(text="ocr result", request_id=1)],
            ),
            mock.patch.object(wp, "restart"),
        ):
            result = wp._send_and_receive_vision(req)
        assert result == "ocr result"

    def test_retry_vision_none_raises(self) -> None:
        from lilbee.providers.worker_process import VisionRequest, WorkerProcess

        wp = WorkerProcess()
        wp._request_queue = mock.MagicMock()
        wp._response_queue = mock.MagicMock()
        wp._process = mock.MagicMock()
        wp._started = True

        req = VisionRequest(png_bytes=b"\x89PNG", model="vis", prompt="", request_id=1)
        with (
            mock.patch.object(wp, "_get_response", return_value=None),
            mock.patch.object(wp, "restart"),
            pytest.raises(RuntimeError, match="crashed again"),
        ):
            wp._retry_vision(req)

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
    """Create a LlamaCppProvider without starting the embed thread."""
    from lilbee.providers.llama_cpp_provider import LlamaCppProvider

    with mock.patch("threading.Thread.start"):
        provider = LlamaCppProvider()
    provider._cache = mock.MagicMock()
    provider._embed_thread = mock.MagicMock()
    return provider


class TestLlamaCppProviderMethods:
    def test_get_chat_llm_non_vision(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_get_chat_llm loads via cache for non-vision models."""
        provider = _make_provider_no_thread()
        cfg.chat_model = "test-model"

        mock_cache_model = mock.MagicMock()
        provider._cache.load_model.return_value = mock_cache_model

        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=Path("/models/test.gguf"),
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider._is_vision_model",
                return_value=False,
            ),
        ):
            result = provider._get_chat_llm()

        assert result == mock_cache_model
        provider._cache.load_model.assert_called_once_with(
            Path("/models/test.gguf"), embedding=False
        )

    def test_get_chat_llm_vision(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_get_chat_llm delegates to _get_vision_llm for vision models."""
        provider = _make_provider_no_thread()
        cfg.chat_model = "vision-model"

        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider._is_vision_model",
                return_value=True,
            ),
            mock.patch.object(
                provider, "_get_vision_llm", return_value=mock.MagicMock()
            ) as mock_vis,
        ):
            result = provider._get_chat_llm()

        mock_vis.assert_called_once_with("vision-model")
        assert result == mock_vis.return_value

    def test_get_chat_llm_with_override_model(self, mock_llama_cpp: mock.MagicMock) -> None:
        """_get_chat_llm uses the override model when provided."""
        provider = _make_provider_no_thread()
        cfg.chat_model = "default-model"

        with (
            mock.patch(
                "lilbee.providers.llama_cpp_provider.resolve_model_path",
                return_value=Path("/models/override.gguf"),
            ),
            mock.patch(
                "lilbee.providers.llama_cpp_provider._is_vision_model",
                return_value=False,
            ),
        ):
            provider._get_chat_llm(model="override-model")

        provider._cache.load_model.assert_called_once_with(
            Path("/models/override.gguf"), embedding=False
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
        existing_vis._model_path = str(vis_path)
        provider._vision_llm = existing_vis

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

        provider._cache.load_model.assert_called_once_with(
            Path("/models/embed.gguf"), embedding=True
        )

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

        with mock.patch("lilbee.services.get_services", return_value=mock_services):
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
            load_llama(Path("/test.gguf"), embedding=True)

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
            load_llama(Path("/test.gguf"), embedding=True)

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_batch"] == 2048

    def test_embedding_with_explicit_ctx(self, mock_llama_cpp: mock.MagicMock) -> None:
        """load_llama with explicit num_ctx uses it for n_batch."""
        from lilbee.providers.llama_cpp_provider import load_llama

        cfg.num_ctx = 4096

        load_llama(Path("/test.gguf"), embedding=True)

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["n_ctx"] == 4096
        assert call_kwargs["n_batch"] == 4096

    def test_chat_mode(self, mock_llama_cpp: mock.MagicMock) -> None:
        """load_llama for chat does not set n_batch."""
        from lilbee.providers.llama_cpp_provider import load_llama

        cfg.num_ctx = None

        load_llama(Path("/test.gguf"), embedding=False)

        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert call_kwargs["embedding"] is False
        assert "n_batch" not in call_kwargs


class TestIsVisionModel:
    def test_matches_config_vision_model(self) -> None:
        """_is_vision_model matches cfg.vision_model."""
        from lilbee.providers.llama_cpp_provider import _is_vision_model

        cfg.vision_model = "my-vision"

        assert _is_vision_model("my-vision") is True

    def test_no_match_when_empty(self) -> None:
        """_is_vision_model returns False for empty vision_model."""
        from lilbee.providers.llama_cpp_provider import _is_vision_model

        cfg.vision_model = ""

        # Only matches featured catalog entries
        assert _is_vision_model("random-model") is False

    def test_matches_featured_vision(self) -> None:
        """_is_vision_model matches FEATURED_VISION entries."""
        from lilbee.providers.llama_cpp_provider import _is_vision_model

        cfg.vision_model = ""

        mock_entry = mock.MagicMock()
        mock_entry.name = "LightOnOCR"
        mock_entry.hf_repo = "noctrex/LightOnOCR-2-1B-GGUF"

        with mock.patch(
            "lilbee.catalog.FEATURED_VISION",
            [mock_entry],
        ):
            assert _is_vision_model("lightonocr") is True
            assert _is_vision_model("no-match-here") is False


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
