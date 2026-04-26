"""Parametrized contract tests for ``LlmSdkBackend`` implementations.

Every adapter must satisfy these invariants. The tests are the
executable contract behind the Protocol. Today the only adapter is
``LitellmSdkBackend``, which is exercised with ``sys.modules["litellm"]``
patched to a ``MagicMock`` so no real SDK call happens. When a second
adapter lands (e.g. ``LiterLlmSdkBackend``), add it to the
``BACKEND_FACTORIES`` parametrization and make sure it keeps passing.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any
from unittest import mock

import httpx
import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.sdk_backend import (
    CompletionRequest,
    EmbeddingRequest,
    LlmSdkBackend,
)


def _make_litellm_backend() -> LlmSdkBackend:
    """Construct the litellm-backed adapter."""
    from lilbee.providers.litellm_sdk import LitellmSdkBackend

    return LitellmSdkBackend()


BACKEND_FACTORIES: list[Callable[[], LlmSdkBackend]] = [_make_litellm_backend]


@pytest.fixture(params=BACKEND_FACTORIES, ids=["litellm"])
def backend(request: pytest.FixtureRequest) -> LlmSdkBackend:
    return request.param()


def _completion_request(model: str = "ollama/m") -> CompletionRequest:
    return CompletionRequest(
        ref=parse_model_ref(model),
        messages=[{"role": "user", "content": "hi"}],
        api_base="http://localhost:11434",
    )


def _embedding_request(model: str = "ollama/m") -> EmbeddingRequest:
    return EmbeddingRequest(
        ref=parse_model_ref(model),
        inputs=["x"],
        api_base="http://localhost:11434",
    )


def _fake_completion_response(content: str) -> Any:
    resp = mock.MagicMock()
    choice = mock.MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp.choices = [choice]
    resp.model = "fake-model"
    return resp


def _fake_stream_response(tokens: list[str]) -> list[Any]:
    chunks: list[Any] = []
    for tok in tokens:
        chunk = mock.MagicMock()
        chunk.choices = [mock.MagicMock()]
        chunk.choices[0].delta.content = tok
        chunk.choices[0].finish_reason = None
        chunks.append(chunk)
    return chunks


class TestProviderName:
    def test_provider_name_is_stable(self, backend: LlmSdkBackend) -> None:
        # A stable, non-empty identifier used in ProviderError.provider.
        assert backend.provider_name
        assert isinstance(backend.provider_name, str)


class TestActiveBackendName:
    def test_ollama_url_returns_ollama(self, backend: LlmSdkBackend) -> None:
        assert backend.active_backend_name("http://localhost:11434") == "Ollama"

    def test_openai_url_returns_openai(self, backend: LlmSdkBackend) -> None:
        assert backend.active_backend_name("https://api.openai.com/v1") == "OpenAI"

    def test_anthropic_url_returns_anthropic(self, backend: LlmSdkBackend) -> None:
        assert backend.active_backend_name("https://api.anthropic.com") == "Anthropic"

    def test_gemini_url_returns_gemini(self, backend: LlmSdkBackend) -> None:
        assert backend.active_backend_name("https://generativelanguage.googleapis.com") == "Gemini"

    def test_unknown_url_falls_back_to_remote(self, backend: LlmSdkBackend) -> None:
        assert backend.active_backend_name("http://192.168.1.50:9000") == "Remote"

    def test_case_insensitive(self, backend: LlmSdkBackend) -> None:
        assert backend.active_backend_name("http://LOCALHOST:11434") == "Ollama"


class TestAvailable:
    def test_returns_true_when_sdk_importable(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            assert backend.available() is True

    @pytest.mark.real_litellm_probe
    def test_returns_false_when_sdk_missing(self, backend: LlmSdkBackend) -> None:
        with mock.patch.dict(sys.modules, {"litellm": None}):
            assert backend.available() is False


class TestCompleteReturnsCompletionResult:
    def test_content_is_extracted(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.completion.return_value = _fake_completion_response("hello world")
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            result = backend.complete(_completion_request())
        assert result.content == "hello world"

    def test_wraps_sdk_error_in_provider_error(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.completion.side_effect = RuntimeError("sdk exploded")
        with (
            mock.patch.dict(sys.modules, {"litellm": fake}),
            pytest.raises(ProviderError, match="Chat failed"),
        ):
            backend.complete(_completion_request())

    def test_formats_image_messages_as_content_parts(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.completion.return_value = _fake_completion_response("ok")
        req = CompletionRequest(
            ref=parse_model_ref("ollama/m"),
            messages=[{"role": "user", "content": "what?", "images": [b"\x89PNG"]}],
            api_base="http://localhost:11434",
        )
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            backend.complete(req)
        call_kwargs = fake.completion.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        # Image payloads get translated to OpenAI content-parts schema.
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "what?"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_plain_messages_pass_through(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.completion.return_value = _fake_completion_response("ok")
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            backend.complete(_completion_request())
        call_kwargs = fake.completion.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_non_bytes_images_are_ignored(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.completion.return_value = _fake_completion_response("ok")
        req = CompletionRequest(
            ref=parse_model_ref("ollama/m"),
            messages=[{"role": "user", "content": "hi", "images": ["not-bytes"]}],
            api_base="http://localhost:11434",
        )
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            backend.complete(req)
        content = fake.completion.call_args[1]["messages"][0]["content"]
        assert len(content) == 1


class TestCompleteStream:
    def test_yields_stream_chunks_with_content(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.completion.return_value = iter(_fake_stream_response(["a", "b", "c"]))
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            chunks = list(backend.complete_stream(_completion_request()))
        assert [c.content for c in chunks] == ["a", "b", "c"]

    def test_wraps_sdk_error_when_opening_stream(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.completion.side_effect = RuntimeError("stream boom")
        with (
            mock.patch.dict(sys.modules, {"litellm": fake}),
            pytest.raises(ProviderError, match="Chat failed"),
        ):
            stream = backend.complete_stream(_completion_request())
            list(stream)

    def test_wraps_mid_iteration_error(self, backend: LlmSdkBackend) -> None:
        def _exploding_iter() -> Any:
            yield from _fake_stream_response(["ok"])
            raise RuntimeError("mid boom")

        fake = mock.MagicMock()
        fake.completion.return_value = _exploding_iter()
        with (
            mock.patch.dict(sys.modules, {"litellm": fake}),
            pytest.raises(ProviderError, match="Chat failed: mid boom"),
        ):
            list(backend.complete_stream(_completion_request()))

    def test_passes_through_provider_error_from_chunk_stream(self, backend: LlmSdkBackend) -> None:
        def _exploding_iter() -> Any:
            # Yield one dummy chunk then raise a pre-wrapped ProviderError
            # during iteration so the adapter's try/except sees the provider
            # error and re-raises it unchanged rather than wrapping it again.
            yield from _fake_stream_response(["ok"])
            raise ProviderError("pre-wrapped", provider="litellm")

        fake = mock.MagicMock()
        fake.completion.return_value = _exploding_iter()
        with (
            mock.patch.dict(sys.modules, {"litellm": fake}),
            pytest.raises(ProviderError, match="pre-wrapped"),
        ):
            list(backend.complete_stream(_completion_request()))

    def test_skips_chunks_with_empty_choices(self, backend: LlmSdkBackend) -> None:
        # Some SDKs emit heartbeat frames with no choices; those must be silently skipped.
        empty_chunk = mock.MagicMock()
        empty_chunk.choices = []
        content_chunk = mock.MagicMock()
        content_chunk.choices = [mock.MagicMock()]
        content_chunk.choices[0].delta.content = "ok"
        content_chunk.choices[0].finish_reason = None
        fake = mock.MagicMock()
        fake.completion.return_value = iter([empty_chunk, content_chunk])
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            chunks = list(backend.complete_stream(_completion_request()))
        assert [c.content for c in chunks] == ["ok"]


class TestCompletionKwargs:
    def test_api_key_and_options_forwarded(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.completion.return_value = _fake_completion_response("ok")
        req = CompletionRequest(
            ref=parse_model_ref("openai/gpt-4o"),
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0.2, "max_tokens": 50},
            api_key="sk-test",
        )
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            backend.complete(req)
        call_kwargs = fake.completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 50

    def test_api_key_forwarded_to_embed(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.embedding.return_value = {"data": [{"embedding": [0.1]}], "model": "x"}
        req = EmbeddingRequest(
            ref=parse_model_ref("openai/text-embedding-3-small"),
            inputs=["hi"],
            api_key="sk-embed",
        )
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            backend.embed(req)
        call_kwargs = fake.embedding.call_args[1]
        assert call_kwargs["api_key"] == "sk-embed"


class TestEmbedReturnsEmbeddingResult:
    def test_vectors_returned(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.embedding.return_value = {"data": [{"embedding": [0.1, 0.2]}], "model": "fake"}
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            result = backend.embed(_embedding_request())
        assert result.vectors == [[0.1, 0.2]]

    def test_embed_error_is_wrapped(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.embedding.side_effect = RuntimeError("nope")
        with (
            mock.patch.dict(sys.modules, {"litellm": fake}),
            pytest.raises(ProviderError, match="Embedding failed"),
        ):
            backend.embed(_embedding_request())

    def test_object_response_model_attribute(self, backend: LlmSdkBackend) -> None:
        # Some SDKs return a response object rather than a dict.
        resp = mock.MagicMock()
        resp.data = [{"embedding": [0.3]}]
        resp.model = "obj-model"
        fake = mock.MagicMock()
        fake.embedding.return_value = resp
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            result = backend.embed(_embedding_request())
        assert result.vectors == [[0.3]]
        assert result.model == "obj-model"


class TestConfigureLogging:
    def test_noop_when_not_suppressing(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.suppress_debug_info = False
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            backend.configure_logging(suppress_debug=False)
        # When suppress_debug is False the backend must not mutate global flags.
        assert fake.suppress_debug_info is False

    def test_sets_flag_when_suppressing(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.suppress_debug_info = False
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            backend.configure_logging(suppress_debug=True)
        assert fake.suppress_debug_info is True

    def test_swallows_import_error(self, backend: LlmSdkBackend) -> None:
        # Calling configure_logging without litellm installed must not raise.
        with mock.patch.dict(sys.modules, {"litellm": None}):
            backend.configure_logging(suppress_debug=True)


class TestListModels:
    def test_ollama_url_hits_api_tags(self, backend: LlmSdkBackend) -> None:
        resp = mock.MagicMock()
        resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
        resp.raise_for_status = mock.MagicMock()
        with mock.patch("httpx.get", return_value=resp) as mock_get:
            models = backend.list_models(base_url="http://localhost:11434", api_key="")
        assert models == ["llama3:8b"]
        assert "/api/tags" in mock_get.call_args[0][0]

    def test_ollama_http_error_wraps_in_provider_error(self, backend: LlmSdkBackend) -> None:
        with (
            mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")),
            pytest.raises(ProviderError, match="Cannot list models"),
        ):
            backend.list_models(base_url="http://localhost:11434", api_key="")

    def test_openai_v1_models_happy_path(self, backend: LlmSdkBackend) -> None:
        resp = mock.MagicMock()
        resp.json.return_value = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        resp.raise_for_status = mock.MagicMock()
        with mock.patch("httpx.get", return_value=resp) as mock_get:
            models = backend.list_models(base_url="https://api.openai.com", api_key="sk-x")
        assert models == ["gpt-4o", "gpt-4o-mini"]
        headers = mock_get.call_args[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-x"

    def test_non_ollama_returns_empty_on_error(self, backend: LlmSdkBackend) -> None:
        with mock.patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert backend.list_models(base_url="https://api.openai.com", api_key="") == []


class TestShowModel:
    def test_returns_none_on_http_error(self, backend: LlmSdkBackend) -> None:
        with mock.patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            assert backend.show_model("m", base_url="http://localhost:11434") is None

    def test_returns_dict_with_parameters(self, backend: LlmSdkBackend) -> None:
        resp = mock.MagicMock()
        resp.json.return_value = {
            "parameters": "temperature 0.5",
            "capabilities": ["completion"],
        }
        resp.raise_for_status = mock.MagicMock()
        with mock.patch("httpx.post", return_value=resp):
            info = backend.show_model("m", base_url="http://localhost:11434")
        assert info is not None
        assert info["parameters"] == "temperature 0.5"
        assert info["capabilities"] == ["completion"]

    def test_non_string_parameters_are_stringified(self, backend: LlmSdkBackend) -> None:
        resp = mock.MagicMock()
        resp.json.return_value = {"parameters": {"temperature": 0.5}}
        resp.raise_for_status = mock.MagicMock()
        with mock.patch("httpx.post", return_value=resp):
            info = backend.show_model("m", base_url="http://localhost:11434")
        assert info is not None
        assert isinstance(info["parameters"], str)


class TestListChatModels:
    def test_returns_chat_mode_models_only(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.models_by_provider = {"openai": {"gpt-4o", "text-embedding-3-small"}}
        fake.model_cost = {
            "gpt-4o": {"mode": "chat"},
            "text-embedding-3-small": {"mode": "embedding"},
        }
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            models = backend.list_chat_models("openai")
        assert models == ["gpt-4o"]

    def test_unknown_provider_returns_empty(self, backend: LlmSdkBackend) -> None:
        fake = mock.MagicMock()
        fake.models_by_provider = {}
        fake.model_cost = {}
        with mock.patch.dict(sys.modules, {"litellm": fake}):
            assert backend.list_chat_models("openai") == []

    def test_returns_empty_when_sdk_missing(self, backend: LlmSdkBackend) -> None:
        with mock.patch.dict(sys.modules, {"litellm": None}):
            assert backend.list_chat_models("openai") == []


class TestPullModel:
    def test_streams_progress_events(self, backend: LlmSdkBackend) -> None:
        events: list[dict[str, Any]] = []
        resp_ctx = mock.MagicMock()
        resp_ctx.__enter__ = lambda self: resp_ctx
        resp_ctx.__exit__ = lambda self, *a: None
        resp_ctx.iter_lines.return_value = [
            json.dumps({"status": "downloading"}),
            "",  # empty lines skipped
            json.dumps({"status": "success"}),
        ]
        resp_ctx.raise_for_status = mock.MagicMock()

        client_ctx = mock.MagicMock()
        client_ctx.__enter__ = lambda self: client_ctx
        client_ctx.__exit__ = lambda self, *a: None
        client_ctx.stream.return_value = resp_ctx

        with mock.patch("httpx.Client", return_value=client_ctx):
            backend.pull_model(
                "m",
                base_url="http://localhost:11434",
                on_progress=events.append,
            )
        assert events == [{"status": "downloading"}, {"status": "success"}]

    def test_http_error_wraps_in_provider_error(self, backend: LlmSdkBackend) -> None:
        client_ctx = mock.MagicMock()
        client_ctx.__enter__ = lambda self: client_ctx
        client_ctx.__exit__ = lambda self, *a: None
        client_ctx.stream.side_effect = httpx.ConnectError("refused")

        with (
            mock.patch("httpx.Client", return_value=client_ctx),
            pytest.raises(ProviderError, match="Cannot pull model"),
        ):
            backend.pull_model("m", base_url="http://localhost:11434")
