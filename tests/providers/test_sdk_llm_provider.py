"""Tests for ``SdkLLMProvider``. The SDK-agnostic semantic layer.

These tests inject an inline ``FakeBackend`` satisfying the
``LlmSdkBackend`` Protocol, so they never touch ``litellm`` or any
other third-party SDK. They verify message formatting, auth-key
injection, option translation, error wrapping, streaming, and the
"optional method returns not-supported" paths.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from lilbee.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.sdk_backend import (
    CompletionRequest,
    CompletionResult,
    EmbeddingRequest,
    EmbeddingResult,
    StreamChunk,
)
from lilbee.providers.sdk_llm_provider import (
    SdkLLMProvider,
    inject_provider_keys,
)


@dataclass
class FakeBackend:
    """Inline ``LlmSdkBackend`` implementation for tests."""

    complete_result: CompletionResult = field(
        default_factory=lambda: CompletionResult(content="hello", model="fake-model")
    )
    stream_chunks: list[StreamChunk] = field(default_factory=list)
    embed_result: EmbeddingResult = field(
        default_factory=lambda: EmbeddingResult(vectors=[[0.1, 0.2]])
    )
    list_models_result: list[str] = field(default_factory=list)
    show_model_result: dict[str, Any] | None = None
    raise_complete: Exception | None = None
    raise_embed: Exception | None = None
    raise_pull: Exception | None = None
    raise_configure_logging: Exception | None = None
    raise_mid_stream: Exception | None = None
    pull_not_supported: bool = False
    list_not_supported: bool = False
    show_not_supported: bool = False
    list_chat_models_not_supported: bool = False
    list_chat_models_result: list[str] = field(default_factory=list)
    provider_name: str = "fake"
    complete_calls: list[CompletionRequest] = field(default_factory=list)
    embed_calls: list[EmbeddingRequest] = field(default_factory=list)
    logging_calls: list[bool] = field(default_factory=list)
    pull_calls: list[tuple[str, str]] = field(default_factory=list)
    list_models_calls: list[tuple[str, str]] = field(default_factory=list)
    list_chat_models_calls: list[str] = field(default_factory=list)

    def available(self) -> bool:
        return True

    def configure_logging(self, *, suppress_debug: bool) -> None:
        if self.raise_configure_logging is not None:
            raise self.raise_configure_logging
        self.logging_calls.append(suppress_debug)

    def complete(self, request: CompletionRequest) -> CompletionResult:
        self.complete_calls.append(request)
        if self.raise_complete is not None:
            raise self.raise_complete
        return self.complete_result

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        self.complete_calls.append(request)
        if self.raise_complete is not None:
            raise self.raise_complete
        for idx, chunk in enumerate(self.stream_chunks):
            if self.raise_mid_stream is not None and idx > 0:
                raise self.raise_mid_stream
            yield chunk

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.embed_calls.append(request)
        if self.raise_embed is not None:
            raise self.raise_embed
        return self.embed_result

    def list_models(self, *, base_url: str, api_key: str) -> list[str]:
        self.list_models_calls.append((base_url, api_key))
        if self.list_not_supported:
            raise NotImplementedError
        return self.list_models_result

    def list_chat_models(self, provider: str) -> list[str]:
        self.list_chat_models_calls.append(provider)
        if self.list_chat_models_not_supported:
            raise NotImplementedError
        return self.list_chat_models_result

    def pull_model(
        self,
        model: str,
        *,
        base_url: str,
        on_progress: Callable[..., Any] | None = None,
    ) -> None:
        self.pull_calls.append((model, base_url))
        if self.pull_not_supported:
            raise NotImplementedError
        if self.raise_pull is not None:
            raise self.raise_pull

    def show_model(self, model: str, *, base_url: str) -> dict[str, Any] | None:
        if self.show_not_supported:
            raise NotImplementedError
        return self.show_model_result


@pytest.fixture(autouse=True)
def _reset_chat_model() -> Iterator[None]:
    """Preserve ``cfg.chat_model``, ``cfg.embedding_model``, and ``cfg.json_mode`` per test."""
    chat_snapshot = cfg.chat_model
    embed_snapshot = cfg.embedding_model
    json_snapshot = cfg.json_mode
    yield
    cfg.chat_model = chat_snapshot
    cfg.embedding_model = embed_snapshot
    cfg.json_mode = json_snapshot


class TestInjectProviderKeys:
    def test_copies_config_values_into_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg.openai_api_key = "sk-abc"
        inject_provider_keys()
        import os

        assert os.environ["OPENAI_API_KEY"] == "sk-abc"

    def test_does_not_overwrite_existing_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-existing")
        cfg.openai_api_key = "sk-from-config"
        inject_provider_keys()
        import os

        assert os.environ["OPENAI_API_KEY"] == "sk-existing"


class TestChatNonStream:
    def test_returns_content_string(self) -> None:
        backend = FakeBackend(complete_result=CompletionResult(content="hi"))
        provider = SdkLLMProvider(backend, base_url="http://localhost:11434")
        assert provider.chat([{"role": "user", "content": "hey"}]) == "hi"

    def test_builds_completion_request_with_parsed_ref(self) -> None:
        backend = FakeBackend()
        provider = SdkLLMProvider(backend, base_url="http://localhost:11434")
        provider.chat([{"role": "user", "content": "hey"}], model="qwen3:8b")
        req = backend.complete_calls[-1]
        # Semantic layer passes the parsed ref; the adapter handles the
        # wire format (ollama/ prefix) when converting to SDK kwargs.
        assert req.ref.raw == "qwen3:8b"
        assert req.ref.name == "qwen3:8b"
        assert req.ref.provider == "local"
        assert req.api_base == "http://localhost:11434"

    def test_api_model_omits_api_base(self) -> None:
        backend = FakeBackend()
        provider = SdkLLMProvider(backend, base_url="http://localhost:11434")
        provider.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")
        req = backend.complete_calls[-1]
        assert req.ref.provider == "openai"
        assert req.ref.name == "gpt-4o"
        assert req.api_base is None

    def test_passes_api_key_when_configured(self) -> None:
        backend = FakeBackend()
        provider = SdkLLMProvider(backend, base_url="https://api.openai.com", api_key="sk-x")
        provider.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")
        assert backend.complete_calls[-1].api_key == "sk-x"

    def test_options_are_translated_per_ref(self) -> None:
        backend = FakeBackend()
        provider = SdkLLMProvider(backend, base_url="https://api.openai.com")
        provider.chat(
            [{"role": "user", "content": "hi"}],
            model="openai/gpt-4o",
            options={"num_predict": 100, "top_k": 40, "temperature": 0.5},
        )
        req = backend.complete_calls[-1]
        assert req.options == {"max_tokens": 100, "temperature": 0.5}

    def test_wraps_unexpected_errors_in_provider_error(self) -> None:
        backend = FakeBackend(raise_complete=RuntimeError("boom"))
        provider = SdkLLMProvider(backend)
        with pytest.raises(ProviderError, match="Chat failed: boom"):
            provider.chat([{"role": "user", "content": "hi"}])

    def test_passes_through_provider_error_unchanged(self) -> None:
        backend = FakeBackend(raise_complete=ProviderError("original", provider="fake"))
        provider = SdkLLMProvider(backend)
        with pytest.raises(ProviderError, match="original"):
            provider.chat([{"role": "user", "content": "hi"}])

    def test_calls_inject_provider_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg.openai_api_key = "sk-injected"
        backend = FakeBackend()
        provider = SdkLLMProvider(backend)
        provider.chat([{"role": "user", "content": "hi"}], model="openai/gpt-4o")
        import os

        assert os.environ.get("OPENAI_API_KEY") == "sk-injected"

    def test_configure_logging_called_with_false_when_json_mode_off(self) -> None:
        cfg.json_mode = False
        backend = FakeBackend()
        provider = SdkLLMProvider(backend)
        assert backend.logging_calls == []
        provider.chat([{"role": "user", "content": "hi"}])
        # Pin the VALUE, not just the length: cfg.json_mode=False must
        # propagate as suppress_debug=False so normal runs keep debug on.
        assert backend.logging_calls == [False]
        # Second call is a no-op (lazy one-shot initialization).
        provider.chat([{"role": "user", "content": "hi"}])
        assert backend.logging_calls == [False]

    def test_configure_logging_called_with_true_when_json_mode_on(self) -> None:
        cfg.json_mode = True
        backend = FakeBackend()
        provider = SdkLLMProvider(backend)
        provider.chat([{"role": "user", "content": "hi"}])
        # Pins cfg.json_mode=True -> backend.configure_logging(True).
        assert backend.logging_calls == [True]

    def test_configure_logging_error_is_swallowed(self) -> None:
        backend = FakeBackend(raise_configure_logging=ImportError("no sdk"))
        provider = SdkLLMProvider(backend)
        # Must not raise; the request should still reach complete().
        provider.chat([{"role": "user", "content": "hi"}])
        assert len(backend.complete_calls) == 1

    def test_initialization_shared_across_chat_and_embed(self) -> None:
        # Pins _ensure_initialized idempotence across methods on the same
        # instance: a chat followed by an embed must only invoke
        # configure_logging and inject_provider_keys once each.
        backend = FakeBackend()
        provider = SdkLLMProvider(backend)
        provider.chat([{"role": "user", "content": "hi"}])
        provider.embed(["text"])
        assert len(backend.logging_calls) == 1


class TestChatStream:
    def test_yields_content_tokens(self) -> None:
        backend = FakeBackend(stream_chunks=[StreamChunk(content="a"), StreamChunk(content="b")])
        provider = SdkLLMProvider(backend)
        result = provider.chat([{"role": "user", "content": "hi"}], stream=True)
        assert list(result) == ["a", "b"]

    def test_skips_empty_chunks(self) -> None:
        backend = FakeBackend(stream_chunks=[StreamChunk(content=""), StreamChunk(content="ok")])
        provider = SdkLLMProvider(backend)
        assert list(provider.chat([{"role": "user", "content": "hi"}], stream=True)) == ["ok"]

    def test_wraps_errors_when_opening_stream(self) -> None:
        backend = FakeBackend(raise_complete=RuntimeError("stream boom"))
        provider = SdkLLMProvider(backend)
        stream = provider.chat([{"role": "user", "content": "hi"}], stream=True)
        with pytest.raises(ProviderError, match="Chat failed: stream boom"):
            list(stream)

    def test_stream_propagates_provider_error(self) -> None:
        backend = FakeBackend(raise_complete=ProviderError("already wrapped"))
        provider = SdkLLMProvider(backend)
        stream = provider.chat([{"role": "user", "content": "hi"}], stream=True)
        with pytest.raises(ProviderError, match="already wrapped"):
            list(stream)

    def test_wraps_mid_iteration_errors(self) -> None:
        backend = FakeBackend(
            stream_chunks=[StreamChunk(content="a"), StreamChunk(content="b")],
            raise_mid_stream=RuntimeError("stream died"),
        )
        provider = SdkLLMProvider(backend)
        stream = provider.chat([{"role": "user", "content": "hi"}], stream=True)
        with pytest.raises(ProviderError, match="Chat failed: stream died"):
            list(stream)


class TestEmbed:
    def test_returns_vectors(self) -> None:
        backend = FakeBackend(embed_result=EmbeddingResult(vectors=[[0.0, 1.0]]))
        provider = SdkLLMProvider(backend)
        assert provider.embed(["hi"]) == [[0.0, 1.0]]

    def test_request_includes_api_base_for_ollama(self) -> None:
        backend = FakeBackend()
        cfg.embedding_model = "nomic-embed-text"
        provider = SdkLLMProvider(backend, base_url="http://localhost:11434")
        provider.embed(["hello"])
        assert backend.embed_calls[-1].api_base == "http://localhost:11434"

    def test_request_omits_api_base_for_api_model(self) -> None:
        backend = FakeBackend()
        cfg.embedding_model = "openai/text-embedding-3-small"
        provider = SdkLLMProvider(backend, base_url="http://localhost:11434")
        provider.embed(["hello"])
        assert backend.embed_calls[-1].api_base is None

    def test_bare_embed_model_with_non_ollama_base_url(self) -> None:
        backend = FakeBackend()
        cfg.embedding_model = "custom-embed"
        provider = SdkLLMProvider(backend, base_url="https://self-hosted:8000")
        provider.embed(["hello"])
        # Non-API, non-Ollama URL keeps the raw name and still passes api_base.
        req = backend.embed_calls[-1]
        assert req.ref.name == "custom-embed:latest"
        assert req.ref.provider == "local"
        assert req.api_base == "https://self-hosted:8000"

    def test_wraps_backend_errors(self) -> None:
        backend = FakeBackend(raise_embed=RuntimeError("oops"))
        provider = SdkLLMProvider(backend)
        with pytest.raises(ProviderError, match="Embedding failed: oops"):
            provider.embed(["hi"])

    def test_provider_error_passes_through(self) -> None:
        backend = FakeBackend(raise_embed=ProviderError("already"))
        provider = SdkLLMProvider(backend)
        with pytest.raises(ProviderError, match="already"):
            provider.embed(["hi"])

    def test_embed_injects_provider_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Parity with chat(): a remote embedding model must see the
        # configured API key in the environment on first use, or hosted
        # embeddings will fail with a 401.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg.openai_api_key = "sk-embed"
        cfg.embedding_model = "openai/text-embedding-3-small"
        backend = FakeBackend()
        provider = SdkLLMProvider(backend)
        provider.embed(["hi"])
        import os

        assert os.environ.get("OPENAI_API_KEY") == "sk-embed"


class TestListModels:
    def test_returns_backend_models(self) -> None:
        backend = FakeBackend(list_models_result=["a", "b"])
        provider = SdkLLMProvider(backend)
        assert provider.list_models() == ["a", "b"]

    def test_returns_empty_when_backend_says_not_supported(self) -> None:
        backend = FakeBackend(list_not_supported=True)
        provider = SdkLLMProvider(backend)
        assert provider.list_models() == []

    def test_wraps_unexpected_errors(self) -> None:
        class _FailingBackend(FakeBackend):
            def list_models(self, *, base_url: str, api_key: str) -> list[str]:
                raise RuntimeError("boom")

        provider = SdkLLMProvider(_FailingBackend())
        with pytest.raises(ProviderError, match="Listing models failed"):
            provider.list_models()

    def test_propagates_provider_error_unchanged(self) -> None:
        class _WrappedError(FakeBackend):
            def list_models(self, *, base_url: str, api_key: str) -> list[str]:
                raise ProviderError("already-typed", provider="fake")

        provider = SdkLLMProvider(_WrappedError())
        with pytest.raises(ProviderError, match="already-typed"):
            provider.list_models()


class TestListChatModels:
    def test_returns_backend_catalog(self) -> None:
        backend = FakeBackend(list_chat_models_result=["openai/gpt-4o"])
        provider = SdkLLMProvider(backend)
        assert provider.list_chat_models("openai") == ["openai/gpt-4o"]
        assert backend.list_chat_models_calls == ["openai"]

    def test_not_supported_returns_empty(self) -> None:
        backend = FakeBackend(list_chat_models_not_supported=True)
        provider = SdkLLMProvider(backend)
        assert provider.list_chat_models("openai") == []

    def test_initializes_backend_before_query(self) -> None:
        # The provider must apply cfg.json_mode suppression before the
        # catalog call, otherwise the litellm banner can leak.
        cfg.json_mode = True
        backend = FakeBackend(list_chat_models_result=["m"])
        provider = SdkLLMProvider(backend)
        provider.list_chat_models("openai")
        assert backend.logging_calls == [True]

    def test_wraps_unexpected_errors(self) -> None:
        class _FailingBackend(FakeBackend):
            def list_chat_models(self, provider: str) -> list[str]:
                raise RuntimeError("catalog boom")

        provider = SdkLLMProvider(_FailingBackend())
        with pytest.raises(ProviderError, match="Listing chat models failed"):
            provider.list_chat_models("openai")

    def test_propagates_provider_error_unchanged(self) -> None:
        class _WrappedError(FakeBackend):
            def list_chat_models(self, provider: str) -> list[str]:
                raise ProviderError("already-typed", provider="fake")

        provider = SdkLLMProvider(_WrappedError())
        with pytest.raises(ProviderError, match="already-typed"):
            provider.list_chat_models("openai")


class TestPullModel:
    def test_forwards_to_backend(self) -> None:
        backend = FakeBackend()
        provider = SdkLLMProvider(backend, base_url="http://localhost:11434")
        provider.pull_model("m")
        assert backend.pull_calls[-1] == ("m", "http://localhost:11434")

    def test_not_supported_raises_provider_error(self) -> None:
        backend = FakeBackend(pull_not_supported=True)
        provider = SdkLLMProvider(backend)
        with pytest.raises(ProviderError, match="does not support pulling"):
            provider.pull_model("m")

    def test_wraps_unexpected_errors(self) -> None:
        class _FailingBackend(FakeBackend):
            def pull_model(
                self,
                model: str,
                *,
                base_url: str,
                on_progress: Any = None,
            ) -> None:
                raise RuntimeError("network boom")

        provider = SdkLLMProvider(_FailingBackend())
        with pytest.raises(ProviderError, match="Cannot pull model 'm': network boom"):
            provider.pull_model("m")

    def test_propagates_provider_error_unchanged(self) -> None:
        class _WrappedError(FakeBackend):
            def pull_model(
                self,
                model: str,
                *,
                base_url: str,
                on_progress: Any = None,
            ) -> None:
                raise ProviderError("already-typed", provider="fake")

        provider = SdkLLMProvider(_WrappedError())
        with pytest.raises(ProviderError, match="already-typed"):
            provider.pull_model("m")


class TestShowModel:
    def test_forwards_result(self) -> None:
        backend = FakeBackend(show_model_result={"parameters": "t 0.1"})
        provider = SdkLLMProvider(backend)
        assert provider.show_model("m") == {"parameters": "t 0.1"}

    def test_not_supported_returns_none(self) -> None:
        backend = FakeBackend(show_not_supported=True)
        provider = SdkLLMProvider(backend)
        assert provider.show_model("m") is None

    def test_wraps_unexpected_errors(self) -> None:
        class _FailingBackend(FakeBackend):
            def show_model(self, model: str, *, base_url: str) -> dict[str, Any] | None:
                raise RuntimeError("metadata boom")

        provider = SdkLLMProvider(_FailingBackend())
        with pytest.raises(ProviderError, match="Showing model 'm' failed: metadata boom"):
            provider.show_model("m")

    def test_propagates_provider_error_unchanged(self) -> None:
        class _WrappedError(FakeBackend):
            def show_model(self, model: str, *, base_url: str) -> dict[str, Any] | None:
                raise ProviderError("already-typed", provider="fake")

        provider = SdkLLMProvider(_WrappedError())
        with pytest.raises(ProviderError, match="already-typed"):
            provider.show_model("m")


class TestGetCapabilities:
    def test_returns_backend_capabilities_list(self) -> None:
        backend = FakeBackend(show_model_result={"capabilities": ["completion", "vision"]})
        provider = SdkLLMProvider(backend)
        assert provider.get_capabilities("m") == ["completion", "vision"]

    def test_empty_when_show_model_returns_none(self) -> None:
        backend = FakeBackend(show_model_result=None)
        provider = SdkLLMProvider(backend)
        assert provider.get_capabilities("m") == []

    def test_empty_when_capabilities_not_a_list(self) -> None:
        backend = FakeBackend(show_model_result={"capabilities": "something"})
        provider = SdkLLMProvider(backend)
        assert provider.get_capabilities("m") == []

    def test_missing_capabilities_key_returns_empty(self) -> None:
        backend = FakeBackend(show_model_result={"parameters": "x"})
        provider = SdkLLMProvider(backend)
        assert provider.get_capabilities("m") == []


class TestShutdown:
    def test_idempotent_and_preserves_backend_state(self) -> None:
        backend = FakeBackend()
        provider = SdkLLMProvider(backend)
        # Calling shutdown twice must not raise and must not trigger any
        # backend calls (SDK providers hold no lilbee-side resources).
        provider.shutdown()
        provider.shutdown()
        assert backend.complete_calls == []
        assert backend.embed_calls == []
