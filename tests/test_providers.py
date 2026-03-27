"""Tests for the LLM provider abstraction layer (mocked — no live servers needed)."""

from __future__ import annotations

import json
import sys
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
    from lilbee.providers.factory import reset_provider

    reset_provider()
    yield
    reset_provider()


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory with test .gguf files."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "test-model.gguf").write_bytes(b"fake-gguf")
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
    def test_embed(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.embedding_model = "test-model"
        cfg.models_dir = models_dir

        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_embedding.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
        }
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = LlamaCppProvider()
        result = provider.embed(["hello", "world"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_llama_instance.create_embedding.assert_called_once_with(input=["hello", "world"])

    def test_chat_non_stream(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.chat_model = "test-model"
        cfg.models_dir = models_dir

        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello there"}}]
        }
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = LlamaCppProvider()
        result = provider.chat([{"role": "user", "content": "hi"}])

        assert result == "Hello there"

    def test_chat_stream(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.chat_model = "test-model"
        cfg.models_dir = models_dir

        stream_chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}}]},
        ]
        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = iter(stream_chunks)
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = LlamaCppProvider()
        result = provider.chat([{"role": "user", "content": "hi"}], stream=True)

        tokens = list(result)
        assert tokens == ["Hello", " world"]

    def test_chat_empty_content(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.chat_model = "test-model"
        cfg.models_dir = models_dir

        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": None}}]
        }
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = LlamaCppProvider()
        result = provider.chat([{"role": "user", "content": "hi"}])

        assert result == ""

    def test_chat_with_options(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.chat_model = "test-model"
        cfg.models_dir = models_dir

        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = LlamaCppProvider()
        provider.chat(
            [{"role": "user", "content": "hi"}],
            options={"temperature": 0.5, "seed": 42},
        )

        mock_llama_instance.create_chat_completion.assert_called_once()
        call_kwargs = mock_llama_instance.create_chat_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["seed"] == 42

    def test_chat_model_override(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.chat_model = "test-model"
        cfg.models_dir = models_dir
        (models_dir / "other-model.gguf").write_bytes(b"fake")

        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = LlamaCppProvider()
        provider.chat([{"role": "user", "content": "hi"}], model="other-model")

        # Llama should have been called with the overridden model path
        call_kwargs = mock_llama_cpp.Llama.call_args[1]
        assert "other-model" in call_kwargs["model_path"]

    def test_list_models(self, models_dir: Path) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.models_dir = models_dir

        provider = LlamaCppProvider()
        result = provider.list_models()
        assert result == ["test-model.gguf"]

    def test_list_models_empty_dir(self, tmp_path: Path) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        empty = tmp_path / "empty"
        empty.mkdir()
        cfg.models_dir = empty

        provider = LlamaCppProvider()
        assert provider.list_models() == []

    def test_list_models_no_dir(self, tmp_path: Path) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.models_dir = tmp_path / "nonexistent"

        provider = LlamaCppProvider()
        assert provider.list_models() == []

    def test_pull_model_raises(self) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        provider = LlamaCppProvider()
        with pytest.raises(NotImplementedError, match="cannot pull"):
            provider.pull_model("some-model")

    def test_show_model_returns_none(self) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        provider = LlamaCppProvider()
        assert provider.show_model("some-model") is None

    def test_resolve_model_path_direct(self, models_dir: Path) -> None:
        from lilbee.providers.llama_cpp_provider import _resolve_model_path

        cfg.models_dir = models_dir
        path = _resolve_model_path(str(models_dir / "test-model.gguf"))
        assert path.name == "test-model.gguf"

    def test_resolve_model_path_in_models_dir(self, models_dir: Path) -> None:
        from lilbee.providers.llama_cpp_provider import _resolve_model_path

        cfg.models_dir = models_dir
        path = _resolve_model_path("test-model")
        assert path.name == "test-model.gguf"

    def test_resolve_model_path_by_filename(self, models_dir: Path) -> None:
        from lilbee.providers.llama_cpp_provider import _resolve_model_path

        cfg.models_dir = models_dir
        path = _resolve_model_path("test-model.gguf")
        assert path.name == "test-model.gguf"

    def test_resolve_model_path_not_found(self, models_dir: Path) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import _resolve_model_path

        cfg.models_dir = models_dir
        with pytest.raises(ProviderError, match="not found"):
            _resolve_model_path("missing-model")

    def test_resolve_model_path_direct_not_exists(self, models_dir: Path) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.llama_cpp_provider import _resolve_model_path

        cfg.models_dir = models_dir
        with pytest.raises(ProviderError, match="Model file not found"):
            _resolve_model_path("/nonexistent/model.gguf")

    def test_embed_caches_llm(self, models_dir: Path, mock_llama_cpp: mock.MagicMock) -> None:
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.embedding_model = "test-model"
        cfg.models_dir = models_dir

        mock_llama_instance = mock.MagicMock()
        mock_llama_instance.create_embedding.return_value = {"data": [{"embedding": [0.1] * 3}]}
        mock_llama_cpp.Llama.return_value = mock_llama_instance

        provider = LlamaCppProvider()
        provider.embed(["a"])
        provider.embed(["b"])

        # Llama should only be instantiated once for embeddings
        assert mock_llama_cpp.Llama.call_count == 1


# ---------------------------------------------------------------------------
# LiteLLMProvider
# ---------------------------------------------------------------------------


class TestLiteLLMProvider:
    def test_embed(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.embedding_model = "nomic-embed-text"

        mock_response = {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]}
        with mock.patch("litellm.embedding", return_value=mock_response) as mock_embed:
            provider = LiteLLMProvider()
            result = provider.embed(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args[1]
        assert call_kwargs["model"] == "ollama/nomic-embed-text"

    def test_embed_preserves_ollama_prefix(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.embedding_model = "ollama/my-model"

        mock_response = {"data": [{"embedding": [0.1]}]}
        with mock.patch("litellm.embedding", return_value=mock_response) as mock_embed:
            provider = LiteLLMProvider()
            provider.embed(["test"])

        assert mock_embed.call_args[1]["model"] == "ollama/my-model"

    def test_model_name_preserves_ollama_prefix(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "ollama/prefixed-model"
        provider = LiteLLMProvider()
        result = provider._model_name()
        assert result == "ollama/prefixed-model"

    def test_model_name_adds_ollama_prefix(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "plain-model"
        provider = LiteLLMProvider()
        result = provider._model_name()
        assert result == "ollama/plain-model"

    def test_embed_error_wrapped(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.embedding_model = "test"

        with mock.patch("litellm.embedding", side_effect=RuntimeError("connection lost")):
            provider = LiteLLMProvider()
            with pytest.raises(ProviderError, match="Embedding failed"):
                provider.embed(["test"])

    def test_chat_non_stream(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "qwen3:8b"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="Hello!"))]

        with mock.patch("litellm.completion", return_value=mock_response) as mock_chat:
            provider = LiteLLMProvider()
            result = provider.chat([{"role": "user", "content": "hi"}])

        assert result == "Hello!"
        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["model"] == "ollama/qwen3:8b"
        assert call_kwargs["stream"] is False

    def test_chat_stream(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "qwen3:8b"

        chunk1 = mock.MagicMock()
        chunk1.choices = [mock.MagicMock(delta=mock.MagicMock(content="Hello"))]
        chunk2 = mock.MagicMock()
        chunk2.choices = [mock.MagicMock(delta=mock.MagicMock(content=" world"))]
        chunk3 = mock.MagicMock()
        chunk3.choices = [mock.MagicMock(delta=mock.MagicMock(content=None))]

        with mock.patch("litellm.completion", return_value=iter([chunk1, chunk2, chunk3])):
            provider = LiteLLMProvider()
            result = provider.chat([{"role": "user", "content": "hi"}], stream=True)

        tokens = list(result)
        assert tokens == ["Hello", " world"]

    def test_chat_stream_empty_choices(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "test"

        chunk = mock.MagicMock()
        chunk.choices = []

        with mock.patch("litellm.completion", return_value=iter([chunk])):
            provider = LiteLLMProvider()
            result = provider.chat([{"role": "user", "content": "hi"}], stream=True)

        assert list(result) == []

    def test_chat_with_options(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "test"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="ok"))]

        with mock.patch("litellm.completion", return_value=mock_response) as mock_chat:
            provider = LiteLLMProvider()
            provider.chat(
                [{"role": "user", "content": "hi"}],
                options={"temperature": 0.7},
            )

        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    def test_chat_model_override(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "default-model"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="ok"))]

        with mock.patch("litellm.completion", return_value=mock_response) as mock_chat:
            provider = LiteLLMProvider()
            provider.chat([{"role": "user", "content": "hi"}], model="other-model")

        assert mock_chat.call_args[1]["model"] == "ollama/other-model"

    def test_chat_error_wrapped(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "test"

        with mock.patch("litellm.completion", side_effect=RuntimeError("timeout")):
            provider = LiteLLMProvider()
            with pytest.raises(ProviderError, match="Chat failed"):
                provider.chat([{"role": "user", "content": "hi"}])

    def test_chat_with_api_key(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "test"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="ok"))]

        with mock.patch("litellm.completion", return_value=mock_response) as mock_chat:
            provider = LiteLLMProvider(api_key="sk-test123")
            provider.chat([{"role": "user", "content": "hi"}])

        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["api_key"] == "sk-test123"

    def test_chat_without_api_key(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.chat_model = "test"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="ok"))]

        with mock.patch("litellm.completion", return_value=mock_response) as mock_chat:
            provider = LiteLLMProvider()
            provider.chat([{"role": "user", "content": "hi"}])

        call_kwargs = mock_chat.call_args[1]
        assert "api_key" not in call_kwargs

    def test_list_models(self) -> None:

        from lilbee.providers.litellm_provider import LiteLLMProvider

        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:latest"}, {"name": "qwen3:8b"}]}

        with mock.patch("httpx.get", return_value=mock_resp) as mock_get:
            provider = LiteLLMProvider()
            result = provider.list_models()

        assert result == ["llama3:latest", "qwen3:8b"]
        mock_get.assert_called_once()

    def test_list_models_error(self) -> None:
        import httpx

        from lilbee.providers.base import ProviderError
        from lilbee.providers.litellm_provider import LiteLLMProvider

        with mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            provider = LiteLLMProvider()
            with pytest.raises(ProviderError, match="Cannot list models"):
                provider.list_models()

    def test_pull_model(self) -> None:

        from lilbee.providers.litellm_provider import LiteLLMProvider

        events = [
            json.dumps({"status": "pulling", "completed": 50, "total": 100}).encode(),
            json.dumps({"status": "success"}).encode(),
        ]

        mock_stream_ctx = mock.MagicMock()
        mock_stream_ctx.__enter__ = mock.MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = mock.MagicMock(return_value=False)
        mock_stream_ctx.iter_lines.return_value = iter(events)

        mock_client = mock.MagicMock()
        mock_client.stream.return_value = mock_stream_ctx
        mock_client.__enter__ = mock.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mock.MagicMock(return_value=False)

        progress_events: list[dict] = []

        def on_progress(event: dict) -> None:
            progress_events.append(event)

        with mock.patch("httpx.Client", return_value=mock_client):
            provider = LiteLLMProvider()
            provider.pull_model("llama3", on_progress=on_progress)

        assert len(progress_events) == 2

    def test_pull_model_error(self) -> None:
        import httpx

        from lilbee.providers.base import ProviderError
        from lilbee.providers.litellm_provider import LiteLLMProvider

        mock_client = mock.MagicMock()
        mock_client.stream.side_effect = httpx.ConnectError("refused")
        mock_client.__enter__ = mock.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("httpx.Client", return_value=mock_client):
            provider = LiteLLMProvider()
            with pytest.raises(ProviderError, match="Cannot pull model"):
                provider.pull_model("llama3")

    def test_pull_model_skips_empty_lines(self) -> None:

        from lilbee.providers.litellm_provider import LiteLLMProvider

        events = [
            b"",
            json.dumps({"status": "success"}).encode(),
        ]

        mock_stream_ctx = mock.MagicMock()
        mock_stream_ctx.__enter__ = mock.MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = mock.MagicMock(return_value=False)
        mock_stream_ctx.iter_lines.return_value = iter(events)

        mock_client = mock.MagicMock()
        mock_client.stream.return_value = mock_stream_ctx
        mock_client.__enter__ = mock.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("httpx.Client", return_value=mock_client):
            provider = LiteLLMProvider()
            provider.pull_model("llama3")

    def test_show_model(self) -> None:

        from lilbee.providers.litellm_provider import LiteLLMProvider

        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": "4.7B"}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result == {"parameters": "4.7B"}

    def test_show_model_no_params(self) -> None:

        from lilbee.providers.litellm_provider import LiteLLMProvider

        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result is None

    def test_show_model_error_returns_none(self) -> None:
        import httpx

        from lilbee.providers.litellm_provider import LiteLLMProvider

        with mock.patch("httpx.post", side_effect=httpx.HTTPError("fail")):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result is None

    def test_show_model_non_string_params(self) -> None:

        from lilbee.providers.litellm_provider import LiteLLMProvider

        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": {"key": "value"}}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result == {"parameters": "{'key': 'value'}"}

    def test_show_model_empty_params_returns_none(self) -> None:

        from lilbee.providers.litellm_provider import LiteLLMProvider

        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": ""}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result is None

    def test_show_model_falsy_non_string_params(self) -> None:

        from lilbee.providers.litellm_provider import LiteLLMProvider

        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": 0}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result is None

    def test_format_messages_with_images(self) -> None:
        from lilbee.providers.litellm_provider import _format_messages

        messages = [{"role": "user", "content": "describe this", "images": [b"\x89PNG fake"]}]
        result = _format_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in result[0]["content"][1]["image_url"]["url"]

    def test_format_messages_without_images(self) -> None:
        from lilbee.providers.litellm_provider import _format_messages

        messages = [{"role": "user", "content": "hello"}]
        result = _format_messages(messages)
        assert result == [{"role": "user", "content": "hello"}]

    def test_format_messages_empty_images(self) -> None:
        from lilbee.providers.litellm_provider import _format_messages

        messages = [{"role": "user", "content": "test", "images": []}]
        result = _format_messages(messages)
        assert result[0]["content"][0] == {"type": "text", "text": "test"}

    def test_format_messages_empty_content_with_images(self) -> None:
        from lilbee.providers.litellm_provider import _format_messages

        messages = [{"role": "user", "images": [b"img"]}]
        result = _format_messages(messages)
        assert result[0]["content"][0] == {"type": "text", "text": ""}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_default_provider_is_routing(self) -> None:
        from lilbee.providers.factory import get_provider
        from lilbee.providers.routing_provider import RoutingProvider

        cfg.llm_provider = "auto"
        provider = get_provider()
        assert isinstance(provider, RoutingProvider)

    def test_explicit_llama_cpp(self) -> None:
        from lilbee.providers.factory import get_provider
        from lilbee.providers.llama_cpp_provider import LlamaCppProvider

        cfg.llm_provider = "llama-cpp"
        provider = get_provider()
        assert isinstance(provider, LlamaCppProvider)

    def test_ollama_alias_provider(self) -> None:
        from lilbee.providers.factory import get_provider
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.llm_provider = "ollama"
        provider = get_provider()
        assert isinstance(provider, LiteLLMProvider)
        assert provider._base_url == "http://localhost:11434"

    def test_litellm_provider(self) -> None:
        from lilbee.providers.factory import get_provider
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.llm_provider = "litellm"
        cfg.llm_api_key = "sk-test"
        provider = get_provider()
        assert isinstance(provider, LiteLLMProvider)
        assert provider._api_key == "sk-test"

    def test_unknown_provider_raises(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.factory import get_provider

        cfg.llm_provider = "unknown"
        with pytest.raises(ProviderError, match="Unknown LLM provider"):
            get_provider()

    def test_singleton(self) -> None:
        from lilbee.providers.factory import get_provider

        cfg.llm_provider = "llama-cpp"
        p1 = get_provider()
        p2 = get_provider()
        assert p1 is p2

    def test_reset_clears_singleton(self) -> None:
        from lilbee.providers.factory import get_provider, reset_provider

        cfg.llm_provider = "llama-cpp"
        p1 = get_provider()
        reset_provider()
        p2 = get_provider()
        assert p1 is not p2

    def test_custom_base_url(self) -> None:
        from lilbee.providers.factory import get_provider
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.llm_provider = "litellm"
        cfg.litellm_base_url = "http://custom:11434"
        provider = get_provider()
        assert isinstance(provider, LiteLLMProvider)
        assert provider._base_url == "http://custom:11434"


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

            c = Config.from_env()
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

            c = Config.from_env()
            assert c.llm_provider == "litellm"
            assert c.litellm_base_url == "http://myhost:11434"
            assert c.llm_api_key == "sk-key"

    def test_models_dir_is_canonical(self) -> None:
        """models_dir uses canonical system path, not per-project data_root."""
        import os

        from lilbee.platform import canonical_models_dir

        with mock.patch.dict(os.environ, {"LILBEE_DATA": "/tmp/test-lilbee"}):
            from lilbee.config import Config

            c = Config.from_env()
            assert c.models_dir == canonical_models_dir()


# ---------------------------------------------------------------------------
# RoutingProvider
# ---------------------------------------------------------------------------


class TestRoutingProvider:
    def _make_provider(self) -> RoutingProvider:
        from lilbee.providers.routing_provider import RoutingProvider

        return RoutingProvider()

    def test_routes_chat_to_litellm_when_model_in_litellm(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = ["qwen3:8b"]
        mock_litellm.chat.return_value = "hello"
        rp._litellm = mock_litellm

        cfg.chat_model = "qwen3:8b"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "hello"
        mock_litellm.chat.assert_called_once()

    def test_routes_chat_to_llama_cpp_when_not_in_litellm(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = []
        rp._litellm = mock_litellm

        mock_llama = mock.MagicMock()
        mock_llama.chat.return_value = "local"
        rp._llama_cpp = mock_llama

        cfg.chat_model = "local-model.gguf"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "local"
        mock_llama.chat.assert_called_once()

    def test_routes_embed_to_litellm_when_model_available(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = ["nomic-embed-text:latest"]
        mock_litellm.embed.return_value = [[0.1, 0.2]]
        rp._litellm = mock_litellm

        cfg.embedding_model = "nomic-embed-text:latest"
        result = rp.embed(["test"])
        assert result == [[0.1, 0.2]]
        mock_litellm.embed.assert_called_once()

    def test_routes_embed_to_llama_cpp_when_not_in_litellm(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = []
        rp._litellm = mock_litellm

        mock_llama = mock.MagicMock()
        mock_llama.embed.return_value = [[0.3, 0.4]]
        rp._llama_cpp = mock_llama

        cfg.embedding_model = "embed.gguf"
        result = rp.embed(["test"])
        assert result == [[0.3, 0.4]]

    def test_list_models_merges_both_sources(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = ["qwen3:8b", "mistral:7b"]
        rp._litellm = mock_litellm

        mock_llama = mock.MagicMock()
        mock_llama.list_models.return_value = ["local.gguf"]
        rp._llama_cpp = mock_llama

        result = rp.list_models()
        assert "qwen3:8b" in result
        assert "local.gguf" in result
        assert len(result) == 3

    def test_litellm_unreachable_falls_back_to_llama_cpp(self) -> None:
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.side_effect = ProviderError("unreachable")
        rp._litellm = mock_litellm

        mock_llama = mock.MagicMock()
        mock_llama.chat.return_value = "fallback"
        rp._llama_cpp = mock_llama

        cfg.chat_model = "local.gguf"
        result = rp.chat([{"role": "user", "content": "hi"}])
        assert result == "fallback"

    def test_show_model_delegates_to_litellm(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = ["qwen3:8b"]
        mock_litellm.show_model.return_value = {"parameters": "temp 0.7"}
        rp._litellm = mock_litellm

        result = rp.show_model("qwen3:8b")
        assert result == {"parameters": "temp 0.7"}
        mock_litellm.show_model.assert_called_once_with("qwen3:8b")

    def test_show_model_falls_back_to_llama_cpp(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = []
        rp._litellm = mock_litellm

        mock_llama = mock.MagicMock()
        mock_llama.show_model.return_value = None
        rp._llama_cpp = mock_llama

        result = rp.show_model("local.gguf")
        assert result is None

    def test_invalidate_cache_clears_litellm_list(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = ["qwen3:8b"]
        rp._litellm = mock_litellm

        # First call caches
        assert rp._is_in_litellm("qwen3:8b")
        assert rp._remote_models is not None

        rp.invalidate_cache()
        assert rp._remote_models is None

    def test_pull_model_delegates_to_litellm(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = ["qwen3:8b"]
        mock_litellm.pull_model.return_value = None
        rp._litellm = mock_litellm

        rp.pull_model("qwen3:8b")
        mock_litellm.pull_model.assert_called_once()
        # Cache should be invalidated after pull
        assert rp._remote_models is None

    def test_pull_model_raises_when_litellm_fails(self) -> None:
        from lilbee.providers.base import ProviderError

        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = []
        mock_litellm.pull_model.side_effect = ProviderError("fail")
        rp._litellm = mock_litellm

        with pytest.raises(ProviderError, match="no pull-capable backend"):
            rp.pull_model("bad-model")

    def test_chat_with_explicit_model_override(self) -> None:
        rp = self._make_provider()
        mock_litellm = mock.MagicMock()
        mock_litellm.list_models.return_value = ["vision:7b"]
        mock_litellm.chat.return_value = "saw it"
        rp._litellm = mock_litellm

        cfg.chat_model = "local.gguf"
        result = rp.chat(
            [{"role": "user", "content": "describe"}],
            model="vision:7b",
        )
        assert result == "saw it"
        mock_litellm.chat.assert_called_once()

    def test_litellm_not_installed_skips_remote(self) -> None:
        """When litellm is not installed, routing provider skips remote models."""
        rp = self._make_provider()

        with mock.patch(
            "lilbee.providers.litellm_provider.litellm_available", return_value=False
        ):
            models = rp._litellm_models()
        assert models == set()


# ---------------------------------------------------------------------------
# litellm_available guard
# ---------------------------------------------------------------------------


class TestLitellmAvailable:
    def test_returns_true_when_installed(self) -> None:
        from lilbee.providers.litellm_provider import litellm_available

        assert litellm_available() is True

    def test_returns_false_when_not_installed(self) -> None:
        from lilbee.providers.litellm_provider import litellm_available

        with mock.patch.dict("sys.modules", {"litellm": None}):
            assert litellm_available() is False

    def test_provider_static_method(self) -> None:
        from lilbee.providers.litellm_provider import LiteLLMProvider

        assert LiteLLMProvider.available() is True

    def test_factory_raises_when_litellm_unavailable(self) -> None:
        from lilbee.providers.base import ProviderError
        from lilbee.providers.factory import get_provider
        from lilbee.providers.litellm_provider import LiteLLMProvider

        cfg.llm_provider = "litellm"
        with (
            mock.patch.object(LiteLLMProvider, "available", return_value=False),
            pytest.raises(ProviderError, match="litellm is not installed"),
        ):
            get_provider()
