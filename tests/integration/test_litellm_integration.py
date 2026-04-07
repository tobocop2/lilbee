"""LiteLLM provider tests — require litellm installed (run in integration CI job).

These tests mock the HTTP calls (litellm.embedding, litellm.completion) but
require litellm itself to be importable for the provider class to work.
"""

from __future__ import annotations

import json
from dataclasses import fields
from unittest import mock

import pytest

litellm = pytest.importorskip("litellm")

from lilbee.config import cfg  # noqa: E402
from lilbee.providers.litellm_provider import LiteLLMProvider, _format_messages  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_cfg():
    snapshot = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    yield
    for name, val in snapshot.items():
        setattr(cfg, name, val)


class TestLiteLLMProvider:
    def test_embed(self) -> None:
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
        cfg.embedding_model = "ollama/my-model"

        mock_response = {"data": [{"embedding": [0.1]}]}
        with mock.patch("litellm.embedding", return_value=mock_response) as mock_embed:
            provider = LiteLLMProvider()
            provider.embed(["test"])

        assert mock_embed.call_args[1]["model"] == "ollama/my-model"

    def test_model_name_preserves_ollama_prefix(self) -> None:
        cfg.chat_model = "ollama/prefixed-model"
        provider = LiteLLMProvider()
        result = provider._model_name()
        assert result == "ollama/prefixed-model"

    def test_model_name_adds_ollama_prefix(self) -> None:
        cfg.chat_model = "plain-model"
        provider = LiteLLMProvider()
        result = provider._model_name()
        assert result == "ollama/plain-model"

    def test_embed_error_wrapped(self) -> None:
        from lilbee.providers.base import ProviderError

        cfg.embedding_model = "test"

        with mock.patch("litellm.embedding", side_effect=RuntimeError("connection lost")):
            provider = LiteLLMProvider()
            with pytest.raises(ProviderError, match="Embedding failed"):
                provider.embed(["test"])

    def test_chat_non_stream(self) -> None:
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
        cfg.chat_model = "test"

        chunk = mock.MagicMock()
        chunk.choices = []

        with mock.patch("litellm.completion", return_value=iter([chunk])):
            provider = LiteLLMProvider()
            result = provider.chat([{"role": "user", "content": "hi"}], stream=True)

        assert list(result) == []

    def test_chat_with_options(self) -> None:
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
        cfg.chat_model = "default-model"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="ok"))]

        with mock.patch("litellm.completion", return_value=mock_response) as mock_chat:
            provider = LiteLLMProvider()
            provider.chat([{"role": "user", "content": "hi"}], model="other-model")

        assert mock_chat.call_args[1]["model"] == "ollama/other-model"

    def test_chat_error_wrapped(self) -> None:
        from lilbee.providers.base import ProviderError

        cfg.chat_model = "test"

        with mock.patch("litellm.completion", side_effect=RuntimeError("timeout")):
            provider = LiteLLMProvider()
            with pytest.raises(ProviderError, match="Chat failed"):
                provider.chat([{"role": "user", "content": "hi"}])

    def test_chat_with_api_key(self) -> None:
        cfg.chat_model = "test"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="ok"))]

        with mock.patch("litellm.completion", return_value=mock_response) as mock_chat:
            provider = LiteLLMProvider(api_key="sk-test123")
            provider.chat([{"role": "user", "content": "hi"}])

        call_kwargs = mock_chat.call_args[1]
        assert call_kwargs["api_key"] == "sk-test123"

    def test_chat_without_api_key(self) -> None:
        cfg.chat_model = "test"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock(message=mock.MagicMock(content="ok"))]

        with mock.patch("litellm.completion", return_value=mock_response) as mock_chat:
            provider = LiteLLMProvider()
            provider.chat([{"role": "user", "content": "hi"}])

        call_kwargs = mock_chat.call_args[1]
        assert "api_key" not in call_kwargs

    def test_list_models(self) -> None:
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

        with mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            provider = LiteLLMProvider()
            with pytest.raises(ProviderError, match="Cannot list models"):
                provider.list_models()

    def test_pull_model(self) -> None:
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

        mock_client = mock.MagicMock()
        mock_client.stream.side_effect = httpx.ConnectError("refused")
        mock_client.__enter__ = mock.MagicMock(return_value=mock_client)
        mock_client.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("httpx.Client", return_value=mock_client):
            provider = LiteLLMProvider()
            with pytest.raises(ProviderError, match="Cannot pull model"):
                provider.pull_model("llama3")

    def test_pull_model_skips_empty_lines(self) -> None:
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
        mock_client.stream.assert_called_once()

    def test_show_model(self) -> None:
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": "4.7B"}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result == {"parameters": "4.7B"}

    def test_show_model_no_params(self) -> None:
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result is None

    def test_show_model_error_returns_none(self) -> None:
        import httpx

        with mock.patch("httpx.post", side_effect=httpx.HTTPError("fail")):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result is None

    def test_show_model_non_string_params(self) -> None:
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": {"key": "value"}}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result == {"parameters": "{'key': 'value'}"}

    def test_show_model_empty_params_returns_none(self) -> None:
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": ""}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result is None

    def test_show_model_falsy_non_string_params(self) -> None:
        mock_resp = mock.MagicMock()
        mock_resp.json.return_value = {"parameters": 0}
        mock_resp.raise_for_status = mock.MagicMock()

        with mock.patch("httpx.post", return_value=mock_resp):
            provider = LiteLLMProvider()
            result = provider.show_model("llama3")

        assert result is None

    def test_format_messages_with_images(self) -> None:
        messages = [{"role": "user", "content": "describe this", "images": [b"\x89PNG fake"]}]
        result = _format_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in result[0]["content"][1]["image_url"]["url"]

    def test_format_messages_without_images(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        result = _format_messages(messages)
        assert result == [{"role": "user", "content": "hello"}]

    def test_format_messages_empty_images(self) -> None:
        messages = [{"role": "user", "content": "test", "images": []}]
        result = _format_messages(messages)
        assert result[0]["content"][0] == {"type": "text", "text": "test"}

    def test_format_messages_empty_content_with_images(self) -> None:
        messages = [{"role": "user", "images": [b"img"]}]
        result = _format_messages(messages)
        assert result[0]["content"][0] == {"type": "text", "text": ""}


class TestLitellmAvailable:
    def test_returns_true_when_installed(self) -> None:
        from lilbee.providers.litellm_provider import litellm_available

        assert litellm_available() is True

    def test_provider_static_method(self) -> None:
        assert LiteLLMProvider.available() is True


class TestLitellmFactory:
    def test_ollama_alias_provider(self) -> None:
        from lilbee.providers.factory import create_provider

        cfg.llm_provider = "ollama"
        provider = create_provider(cfg)
        assert isinstance(provider, LiteLLMProvider)
        assert provider._base_url == "http://localhost:11434"

    def test_litellm_provider(self) -> None:
        from lilbee.providers.factory import create_provider

        cfg.llm_provider = "litellm"
        cfg.llm_api_key = "sk-test"
        provider = create_provider(cfg)
        assert isinstance(provider, LiteLLMProvider)
        assert provider._api_key == "sk-test"

    def test_custom_base_url(self) -> None:
        from lilbee.providers.factory import create_provider

        cfg.llm_provider = "litellm"
        cfg.litellm_base_url = "http://custom:11434"
        provider = create_provider(cfg)
        assert isinstance(provider, LiteLLMProvider)
        assert provider._base_url == "http://custom:11434"
