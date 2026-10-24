"""Contract tests for ``LlmSdkBackend`` implementations.

The tests are the executable contract behind the Protocol. The adapter is
``LiterLlmBackend``, run against a scripted stand-in for ``liter_llm`` so no
real SDK call happens and the extra need not be installed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

import httpx
import pytest

from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.model_ref import format_remote_ref, parse_model_ref
from lilbee.providers.sdk_backend import (
    CompletionRequest,
    EmbeddingRequest,
    LlmSdkBackend,
    RerankRequest,
)
from tests import _fake_sdk as fakes


@pytest.fixture
def backend() -> LlmSdkBackend:
    return LiterLlmBackend()


@pytest.fixture
def sdk() -> Iterator[fakes.FakeSdk]:
    with fakes.installed(fakes.FakeSdk()) as fake:
        yield fake


def _completion_request(model: str = "ollama/m", **overrides: Any) -> CompletionRequest:
    fields: dict[str, Any] = {
        "ref": parse_model_ref(model),
        "messages": [{"role": "user", "content": "hi"}],
        "api_base": "http://localhost:11434",
    }
    fields.update(overrides)
    return CompletionRequest(**fields)


def _embedding_request(
    model: str = "ollama/m", inputs: list[str] | None = None, **overrides: Any
) -> EmbeddingRequest:
    fields: dict[str, Any] = {
        "ref": parse_model_ref(model),
        "inputs": inputs if inputs is not None else ["x"],
        "api_base": "http://localhost:11434",
    }
    fields.update(overrides)
    return EmbeddingRequest(**fields)


class TestProviderName:
    def test_provider_name_is_stable(self, backend: LlmSdkBackend) -> None:
        assert backend.provider_name == "remote"


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
    def test_reads_the_installed_distribution(self, backend: LlmSdkBackend) -> None:
        assert backend.available() is True

    @pytest.mark.real_provider_sdk_probe
    def test_returns_false_when_sdk_missing(self, backend: LlmSdkBackend) -> None:
        from importlib.metadata import PackageNotFoundError

        def not_installed(dist: str) -> str:
            raise PackageNotFoundError(dist)

        with mock.patch("lilbee.providers.availability._dist_version", not_installed):
            from lilbee.providers.availability import provider_sdk_available

            provider_sdk_available.cache_clear()
            try:
                assert backend.available() is False
            finally:
                provider_sdk_available.cache_clear()


class TestMissingSdk:
    def test_complete_names_the_extra(self, backend: LlmSdkBackend) -> None:
        from tests._sys_modules import inject_modules

        with (
            inject_modules({fakes.SDK_MODULE: None}),
            pytest.raises(ProviderError, match=r"lilbee\[providers\]"),
        ):
            backend.complete(_completion_request())


class TestCompleteReturnsCompletionResult:
    def test_content_model_and_finish_reason_are_extracted(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("hello world", model="m-1", finish="length")
        result = backend.complete(_completion_request())
        assert (result.content, result.model, result.finish_reason) == (
            "hello world",
            "m-1",
            "length",
        )
        assert (type(result.content), type(result.finish_reason)) == (str, str)

    def test_tool_calls_are_extracted(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.chat_result = fakes.tool_call_response("c1", "lookup", '{"q": 1}')
        result = backend.complete(_completion_request())
        assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
            ("c1", "lookup", '{"q": 1}')
        ]
        assert result.content == ""

    def test_response_without_choices_is_empty(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("x")
        sdk.chat_result.choices = []
        result = backend.complete(_completion_request())
        assert (result.content, result.finish_reason, result.tool_calls) == ("", None, ())

    def test_wraps_sdk_error_in_provider_error(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_error = RuntimeError("sdk exploded")
        with pytest.raises(ProviderError, match="Chat failed: sdk exploded"):
            backend.complete(_completion_request())

    def test_formats_image_messages_as_content_parts(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(
            _completion_request(
                messages=[{"role": "user", "content": "what?", "images": [b"\x89PNG"]}]
            )
        )
        content = sdk.last("chat")["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "what?"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_plain_messages_pass_through(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request())
        assert sdk.last("chat")["messages"] == [{"role": "user", "content": "hi"}]

    def test_non_bytes_images_are_ignored(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(
            _completion_request(
                messages=[{"role": "user", "content": "hi", "images": ["not-bytes"]}]
            )
        )
        assert len(sdk.last("chat")["messages"][0]["content"]) == 1


class TestCompleteStream:
    def test_yields_stream_chunks_with_content(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.stream_chunks = [fakes.chunk("a"), fakes.chunk("b"), fakes.chunk("c", finish="stop")]
        chunks = list(backend.complete_stream(_completion_request()))
        assert [(c.content, c.finish_reason) for c in chunks] == [
            ("a", None),
            ("b", None),
            ("c", "stop"),
        ]
        assert {type(c.content) for c in chunks} == {str}
        assert type(chunks[-1].finish_reason) is str
        assert sdk.last("chat_stream")["stream"] is True

    def test_tool_call_deltas_are_normalised(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.stream_chunks = [
            fakes.chunk(
                tool_calls=[fakes.tool_call_delta(0, call_id="c1", name="f", arguments="")]
            ),
            fakes.chunk(tool_calls=[fakes.tool_call_delta(0, arguments='{"a": 1}')]),
        ]
        chunks = list(backend.complete_stream(_completion_request()))
        deltas = [d for c in chunks for d in c.tool_call_deltas]
        assert [(d.index, d.id, d.name, d.arguments_delta) for d in deltas] == [
            (0, "c1", "f", None),
            (0, None, None, '{"a": 1}'),
        ]

    def test_a_failure_opening_the_stream_raises_before_iteration(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.stream_open_error = sdk.error("NetworkError", "refused")
        with pytest.raises(ProviderError) as err:
            backend.complete_stream(_completion_request())
        assert err.value.kind is ProviderErrorKind.CONNECTION

    def test_wraps_mid_iteration_error(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.stream_chunks = [fakes.chunk("ok")]
        sdk.stream_error = RuntimeError("mid boom")
        stream = backend.complete_stream(_completion_request())
        assert next(stream).content == "ok"
        with pytest.raises(ProviderError, match="Chat failed: mid boom"):
            next(stream)

    def test_skips_chunks_with_empty_choices(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        # Some providers emit heartbeat frames with no choices; those are skipped.
        heartbeat = fakes.chunk("ignored")
        heartbeat.choices = []
        sdk.stream_chunks = [heartbeat, fakes.chunk("ok")]
        assert [c.content for c in backend.complete_stream(_completion_request())] == ["ok"]

    def test_skips_chunks_without_content_or_finish(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.stream_chunks = [fakes.chunk(None), fakes.chunk("ok")]
        assert [c.content for c in backend.complete_stream(_completion_request())] == ["ok"]

    def test_an_empty_stream_yields_nothing(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.stream_chunks = []
        assert list(backend.complete_stream(_completion_request())) == []


class TestRequestFields:
    """What the adapter puts on the request it hands to liter-llm."""

    def test_local_ref_is_prefixed_and_gets_the_v1_base_the_ollama_route_strips(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request("ollama/m"))
        assert sdk.last("chat")["model"] == "ollama/m:latest"
        assert sdk.clients[-1]["base_url"] == "http://localhost:11434/v1"

    def test_a_base_that_already_ends_in_v1_is_kept(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request("lm_studio/m", api_base="http://localhost:1234/v1/"))
        assert sdk.last("chat")["model"] == "lm_studio/m"
        assert sdk.clients[-1]["base_url"] == "http://localhost:1234/v1"

    def test_hosted_ref_sends_no_base_url(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request("openai/gpt-4o", api_base=None, api_key="sk-test"))
        assert sdk.last("chat")["model"] == "openai/gpt-4o"
        assert sdk.clients[-1]["base_url"] is None
        assert sdk.clients[-1]["api_key"] == "sk-test"

    @pytest.mark.parametrize(
        ("model", "api_base", "retries"),
        [
            ("ollama/m", "http://localhost:11434", 0),
            ("lm_studio/m", "http://localhost:1234/v1", 0),
            ("openai/gpt-4o", None, 2),
            ("anthropic/claude-3-5-sonnet", None, 2),
        ],
    )
    def test_retries_are_none_for_local_servers_and_two_for_hosted(
        self,
        backend: LlmSdkBackend,
        sdk: fakes.FakeSdk,
        model: str,
        api_base: str | None,
        retries: int,
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request(model, api_base=api_base, api_key="sk-test"))
        assert sdk.clients[-1]["max_retries"] == retries
        assert sdk.clients[-1]["timeout_secs"] == 600

    def test_missing_key_sends_an_empty_key(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request())
        assert sdk.clients[-1]["api_key"] == ""

    def test_one_client_serves_repeat_calls_to_the_same_endpoint(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request())
        backend.complete(_completion_request())
        assert len(sdk.clients) == 1

    def test_bare_local_ref_takes_the_prefix_of_its_server(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(
            _completion_request("org/repo/model.gguf", api_base="http://localhost:1234/v1")
        )
        assert sdk.last("chat")["model"] == "lm_studio/org/repo/model.gguf"

    def test_bare_local_ref_without_a_server_stays_bare(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(
            _completion_request("org/repo/model.gguf", api_base="http://192.168.1.50:9000")
        )
        assert sdk.last("chat")["model"] == "org/repo/model.gguf"

    def test_typed_options_are_forwarded(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.chat_result = fakes.text_response("ok")
        options = {
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 3,
            "max_tokens": 50,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop": ["a"],
            "response_format": {"type": "json_object"},
            "tool_choice": "auto",
        }
        backend.complete(_completion_request("openai/gpt-4o", options=options, api_base=None))
        sent = sdk.last("chat")
        assert {key: sent[key] for key in options} == options
        assert "extra_body" not in sent

    def test_num_predict_becomes_max_tokens(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request(options={"num_predict": 64}))
        sent = sdk.last("chat")
        assert sent["max_tokens"] == 64
        assert "num_predict" not in sent

    def test_tools_are_sent_as_typed_tools(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
        backend.complete(_completion_request(options={"tools": [tool]}))
        assert [t.parsed for t in sdk.last("chat")["tools"]] == [tool]

    def test_reasoning_effort_is_the_typed_field(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request(options={"reasoning_effort": "low"}))
        sent = sdk.last("chat")
        assert sent["reasoning_effort"] is fakes.ReasoningEffort.LOW
        assert "extra_body" not in sent

    def test_thinking_off_is_the_none_effort(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request(options={"reasoning_effort": "none"}))
        assert sdk.last("chat")["reasoning_effort"] is fakes.ReasoningEffort.NONE

    def test_unknown_options_travel_in_the_body(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request(options={"min_p": 0.05}))
        assert sdk.last("chat")["extra_body"] == {"min_p": 0.05}

    @pytest.mark.parametrize(
        ("option", "value"), [("top_k", 5), ("num_ctx", 4096), ("repeat_penalty", 1.1)]
    )
    @pytest.mark.parametrize(
        ("ref", "api_base"),
        [("ollama/m", None), ("lm_studio/m", "http://localhost:1234")],
    )
    def test_local_server_keeps_the_option_in_the_body(
        self,
        backend: LlmSdkBackend,
        sdk: fakes.FakeSdk,
        ref: str,
        api_base: str | None,
        option: str,
        value: float,
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(_completion_request(ref, api_base=api_base, options={option: value}))
        assert sdk.last("chat")["extra_body"] == {option: value}


class TestEmbedReturnsEmbeddingResult:
    def test_vectors_and_model_returned(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.embed_result = fakes.embedding_response([(0, [0.1, 0.2])], model="fake")
        result = backend.embed(_embedding_request())
        assert (result.vectors, result.model) == ([[0.1, 0.2]], "fake")

    def test_request_carries_model_inputs_and_key(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.embed_result = fakes.embedding_response([(0, [0.1])])
        backend.embed(
            _embedding_request(
                "openai/text-embedding-3-small", ["hi"], api_base=None, api_key="sk-embed"
            )
        )
        assert sdk.last("embed") == {"model": "openai/text-embedding-3-small", "input": ["hi"]}
        assert sdk.clients[-1]["api_key"] == "sk-embed"

    def test_embeddings_reordered_by_index(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        """A gateway returning the batch out of order is corrected by the response's
        index, since the consumer zips vectors to inputs positionally."""
        sdk.embed_result = fakes.embedding_response([(1, [1.0]), (0, [0.0])], as_given=True)
        result = backend.embed(_embedding_request(inputs=["x", "y"]))
        assert result.vectors == [[0.0], [1.0]]

    def test_short_embedding_batch_is_refused(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        """A dropped item still sorts cleanly but breaks the positional pairing."""
        sdk.embed_result = fakes.embedding_response([(0, [0.0])])
        with pytest.raises(ProviderError, match="incomplete or misindexed") as err:
            backend.embed(_embedding_request(inputs=["x", "y"]))
        assert err.value.kind is ProviderErrorKind.SERVER

    def test_duplicate_embedding_index_is_refused(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        """A repeated index yields the right count and the wrong pairing."""
        sdk.embed_result = fakes.embedding_response([(0, [0.0]), (0, [1.0])])
        with pytest.raises(ProviderError, match="incomplete or misindexed"):
            backend.embed(_embedding_request(inputs=["x", "y"]))

    def test_embed_error_is_wrapped(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.embed_error = RuntimeError("nope")
        with pytest.raises(ProviderError, match="Embedding failed: nope"):
            backend.embed(_embedding_request())


class TestRerank:
    def test_scores_follow_the_candidate_order(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        # The SDK returns best first; the caller expects one score per candidate.
        sdk.rerank_result = fakes.rerank_response([(1, 0.9), (0, 0.1)])
        result = backend.rerank(
            RerankRequest(
                ref=parse_model_ref("openai/rerank-x"),
                query="q",
                candidates=["a", "b"],
                api_key="k",
            )
        )
        assert result.scores == [0.1, 0.9]
        assert sdk.last("rerank") == {
            "model": "openai/rerank-x",
            "query": "q",
            "documents": ["a", "b"],
        }

    def test_no_candidates_makes_no_call(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        result = backend.rerank(
            RerankRequest(ref=parse_model_ref("openai/rerank-x"), query="q", candidates=[])
        )
        assert result.scores == []
        assert sdk.calls == []

    def test_rerank_error_is_wrapped(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.rerank_error = sdk.error("EndpointNotSupportedError")
        with pytest.raises(ProviderError) as err:
            backend.rerank(
                RerankRequest(ref=parse_model_ref("openai/rerank-x"), query="q", candidates=["a"])
            )
        assert err.value.kind is ProviderErrorKind.BAD_REQUEST


class TestConfigureLogging:
    def test_is_a_no_op_either_way(self, backend: LlmSdkBackend) -> None:
        backend.configure_logging(suppress_debug=True)
        backend.configure_logging(suppress_debug=False)


class TestSupportsTools:
    def test_every_ref_reports_tool_support(self, backend: LlmSdkBackend) -> None:
        assert backend.supports_tools("ollama/m") is True


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

    def test_returns_none_for_a_server_without_a_metadata_endpoint(
        self, backend: LlmSdkBackend
    ) -> None:
        assert backend.show_model("m", base_url="http://localhost:1234/v1") is None

    def test_returns_dict_with_parameters(self, backend: LlmSdkBackend) -> None:
        resp = mock.MagicMock()
        resp.json.return_value = {
            "parameters": "temperature 0.5",
            "capabilities": ["completion"],
        }
        resp.raise_for_status = mock.MagicMock()
        with mock.patch("httpx.post", return_value=resp):
            info = backend.show_model("ollama/m", base_url="http://localhost:11434")
        assert info == {"parameters": "temperature 0.5", "capabilities": ["completion"]}

    def test_non_string_parameters_are_stringified(self, backend: LlmSdkBackend) -> None:
        resp = mock.MagicMock()
        resp.json.return_value = {"parameters": {"temperature": 0.5}}
        resp.raise_for_status = mock.MagicMock()
        with mock.patch("httpx.post", return_value=resp):
            info = backend.show_model("m", base_url="http://localhost:11434")
        assert info == {"parameters": "{'temperature': 0.5}"}

    def test_empty_reply_is_none(self, backend: LlmSdkBackend) -> None:
        resp = mock.MagicMock()
        resp.json.return_value = {}
        resp.raise_for_status = mock.MagicMock()
        with mock.patch("httpx.post", return_value=resp):
            assert backend.show_model("m", base_url="http://localhost:11434") is None


class TestListChatModels:
    @pytest.fixture
    def catalog(self, sdk: fakes.FakeSdk) -> fakes.FakeSdk:
        sdk.catalog = [
            fakes.catalog_model("openai", "gpt-4o"),
            fakes.catalog_model("openai", "gpt-4o-mini"),
            fakes.catalog_model("openai", "text-embedding-3-small", "embedding"),
            fakes.catalog_model("openai", "gpt-image-1", "image_generation"),
            fakes.catalog_model("openai", "tts-1", "audio_speech"),
            fakes.catalog_model("openai", "unclassified", None),
            fakes.catalog_model("google", "gemini-2.5-pro"),
            fakes.catalog_model("openrouter", "anthropic/claude-sonnet"),
            fakes.catalog_model("anthropic", "claude-haiku"),
        ]
        return sdk

    def test_lists_chat_models_without_the_provider_prefix(
        self, backend: LlmSdkBackend, catalog: fakes.FakeSdk
    ) -> None:
        assert backend.list_chat_models("openai") == ["gpt-4o", "gpt-4o-mini"]

    def test_excludes_embedding_image_audio_and_unclassified_models(
        self, backend: LlmSdkBackend, catalog: fakes.FakeSdk
    ) -> None:
        names = backend.list_chat_models("openai")
        assert not {"text-embedding-3-small", "gpt-image-1", "tts-1", "unclassified"} & set(names)

    def test_gemini_reads_the_catalog_entries_of_google(
        self, backend: LlmSdkBackend, catalog: fakes.FakeSdk
    ) -> None:
        assert backend.list_chat_models("gemini") == ["gemini-2.5-pro"]
        assert catalog.last("list_catalog_models") == {"provider": "google"}

    def test_a_provider_the_catalog_lacks_lists_nothing(
        self, backend: LlmSdkBackend, catalog: fakes.FakeSdk
    ) -> None:
        assert backend.list_chat_models("no-such-provider") == []

    def test_an_empty_provider_lists_nothing(
        self, backend: LlmSdkBackend, catalog: fakes.FakeSdk
    ) -> None:
        assert backend.list_chat_models("") == []

    def test_every_listing_is_for_one_provider(
        self, backend: LlmSdkBackend, catalog: fakes.FakeSdk
    ) -> None:
        backend.list_chat_models("anthropic")
        assert catalog.last("list_catalog_models") == {"provider": "anthropic"}

    @pytest.mark.parametrize(
        ("provider", "name"),
        [
            ("openai", "gpt-4o"),
            ("gemini", "gemini-2.5-pro"),
            ("openrouter", "anthropic/claude-sonnet"),
        ],
    )
    def test_names_route_back_to_the_provider(
        self, backend: LlmSdkBackend, catalog: fakes.FakeSdk, provider: str, name: str
    ) -> None:
        assert name in backend.list_chat_models(provider)
        ref = parse_model_ref(format_remote_ref(name, provider))
        assert (ref.provider, ref.name) == (provider, name)
        assert ref.for_openai_prefix() == f"{provider}/{name}"


class TestPullModel:
    def test_refused_for_read_only_ollama(self, backend: LlmSdkBackend) -> None:
        """Ollama is read-only: pull is refused without any network call."""
        with (
            mock.patch("httpx.Client") as client,
            pytest.raises(ProviderError, match="Ollama"),
        ):
            backend.pull_model("m", base_url="http://localhost:11434")
        client.assert_not_called()

    def test_refused_for_read_only_lm_studio(self, backend: LlmSdkBackend) -> None:
        with pytest.raises(ProviderError, match="LM Studio"):
            backend.pull_model("m", base_url="http://localhost:1234/v1")

    def test_refused_for_an_unknown_server(self, backend: LlmSdkBackend) -> None:
        with pytest.raises(ProviderError, match="This server"):
            backend.pull_model("m", base_url="http://192.168.1.50:9000")


class TestReplyText:
    """A reply's content object is read through its accessors, never through ``str``."""

    def test_a_text_reply_gives_its_exact_text(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("Pong. \u00e9\n")
        assert backend.complete(_completion_request()).content == "Pong. \u00e9\n"

    def test_a_parts_reply_gives_its_text_parts_joined(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response(
            None, parts=[("text", "Po"), ("refusal", "NO"), ("text", "ng")]
        )
        assert backend.complete(_completion_request()).content == "Pong"

    def test_a_reply_with_neither_text_nor_parts_is_empty(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("x")
        sdk.chat_result.choices[0].message.content = fakes.AssistantContent(None, None)
        assert backend.complete(_completion_request()).content == ""

    def test_a_reply_with_no_content_is_empty(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response(None)
        assert backend.complete(_completion_request()).content == ""

    def test_an_empty_text_reply_is_empty(self, backend: LlmSdkBackend, sdk: fakes.FakeSdk) -> None:
        sdk.chat_result = fakes.text_response("")
        assert backend.complete(_completion_request()).content == ""


class TestFinishReasons:
    WIRE = ("stop", "length", "tool_calls", "content_filter", "function_call", "other")

    @pytest.mark.parametrize("wire", WIRE)
    def test_complete_gives_the_wire_string(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk, wire: str
    ) -> None:
        sdk.chat_result = fakes.text_response("x", finish=wire)
        assert backend.complete(_completion_request()).finish_reason == wire

    @pytest.mark.parametrize("wire", WIRE)
    def test_stream_gives_the_wire_string(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk, wire: str
    ) -> None:
        sdk.stream_chunks = [fakes.chunk("x", finish=wire)]
        (chunk,) = backend.complete_stream(_completion_request())
        assert chunk.finish_reason == wire

    def test_the_tests_cover_every_member_the_fake_lists(self) -> None:
        assert set(self.WIRE) == set(fakes.FINISH_WIRE.values())


class TestStreamText:
    def test_a_missing_delta_text_between_texts_adds_nothing(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.stream_chunks = [
            fakes.chunk("Po"),
            fakes.chunk(None, tool_calls=[fakes.tool_call_delta(0, call_id="c")]),
            fakes.chunk("ng"),
            fakes.chunk(None, finish="stop"),
        ]
        chunks = list(backend.complete_stream(_completion_request()))
        assert "".join(c.content for c in chunks) == "Pong"
        assert [c.finish_reason for c in chunks] == [None, None, None, "stop"]
        assert chunks[-1].content == ""


class TestToolArguments:
    ARGS = '{"city":  "Paris", "n": 1,\n "u": "\\u00e9"}'

    def test_a_complete_call_keeps_the_arguments_byte_for_byte(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.tool_call_response("c1", "f", self.ARGS)
        (call,) = backend.complete(_completion_request()).tool_calls
        assert call.arguments == self.ARGS
        assert json.loads(call.arguments)["city"] == "Paris"

    def test_joined_stream_deltas_keep_the_arguments_byte_for_byte(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        cut = 7
        sdk.stream_chunks = [
            fakes.chunk(tool_calls=[fakes.tool_call_delta(0, call_id="c1", name="f")]),
            fakes.chunk(tool_calls=[fakes.tool_call_delta(0, arguments=self.ARGS[:cut])]),
            fakes.chunk(tool_calls=[fakes.tool_call_delta(0, arguments=self.ARGS[cut:])]),
        ]
        deltas = [
            d for c in backend.complete_stream(_completion_request()) for d in c.tool_call_deltas
        ]
        assert "".join(d.arguments_delta or "" for d in deltas) == self.ARGS


class TestImageParts:
    def test_each_image_part_is_an_image_url_with_a_data_url(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("ok")
        backend.complete(
            _completion_request(
                messages=[{"role": "user", "content": "what?", "images": [b"\x89PNG", b"\x01"]}]
            )
        )
        assert sdk.last("chat")["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw=="}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQ=="}},
                ],
            }
        ]


class TestMultiVectorEmbeddings:
    def test_vectors_follow_the_input_order_when_the_reply_is_reversed(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        vectors = [(0, [0.0, 0.5]), (1, [1.0, 1.5]), (2, [2.0, 2.5])]
        sdk.embed_result = fakes.embedding_response(vectors)
        assert [item.index for item in sdk.embed_result.data] == [2, 1, 0]
        result = backend.embed(_embedding_request(inputs=["a", "b", "c"]))
        assert result.vectors == [vector for _, vector in vectors]

    @pytest.mark.parametrize(
        ("base", "wire"),
        [
            ("http://localhost:11434", "http://localhost:11434/v1"),
            ("http://localhost:11434/v1", "http://localhost:11434/v1"),
            ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
        ],
    )
    def test_an_ollama_embed_takes_the_v1_base_once(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk, base: str, wire: str
    ) -> None:
        sdk.embed_result = fakes.embedding_response([(0, [0.1])])
        backend.embed(_embedding_request("ollama/all-minilm", api_base=base))
        assert sdk.clients[-1]["base_url"] == wire
        assert sdk.last("embed")["model"] == "ollama/all-minilm:latest"


class TestRerankOrder:
    def test_scores_are_placed_by_index_not_by_rank(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.rerank_result = fakes.rerank_response([(0, 0.2), (1, 0.9), (2, 0.5)])
        assert [r.index for r in sdk.rerank_result.results] == [1, 2, 0]
        result = backend.rerank(
            RerankRequest(
                ref=parse_model_ref("openai/rerank-x"), query="q", candidates=["a", "b", "c"]
            )
        )
        assert result.scores == [0.2, 0.9, 0.5]


def _call_complete(backend: LlmSdkBackend) -> object:
    return backend.complete(_completion_request())


def _call_open_stream(backend: LlmSdkBackend) -> object:
    return backend.complete_stream(_completion_request())


def _call_mid_stream(backend: LlmSdkBackend) -> object:
    stream = backend.complete_stream(_completion_request())
    return list(stream)


def _call_embed(backend: LlmSdkBackend) -> object:
    return backend.embed(_embedding_request())


def _call_rerank(backend: LlmSdkBackend) -> object:
    request = RerankRequest(ref=parse_model_ref("openai/rerank-x"), query="q", candidates=["a"])
    return backend.rerank(request)


def _arm(attr: str) -> Callable[[fakes.FakeSdk, BaseException], None]:
    def arm(sdk: fakes.FakeSdk, error: BaseException) -> None:
        setattr(sdk, attr, error)
        if attr == "stream_error":
            sdk.stream_chunks = [fakes.chunk("ok")]

    return arm


class TestErrorKindsOnEveryCallKind:
    """Every wrapped failure carries the kind of its error class, on all five paths."""

    PATHS = (
        ("chat_error", _call_complete),
        ("stream_open_error", _call_open_stream),
        ("stream_error", _call_mid_stream),
        ("embed_error", _call_embed),
        ("rerank_error", _call_rerank),
    )
    KINDS = (
        ("AuthenticationError", ProviderErrorKind.AUTH),
        ("NotFoundError", ProviderErrorKind.NOT_FOUND),
        ("RateLimitedError", ProviderErrorKind.RATE_LIMIT),
        ("ContextWindowExceededError", ProviderErrorKind.CONTEXT_OVERFLOW),
        ("BadRequestError", ProviderErrorKind.BAD_REQUEST),
        ("NetworkError", ProviderErrorKind.CONNECTION),
        ("ServerError", ProviderErrorKind.SERVER),
    )

    @pytest.mark.parametrize(("attr", "call"), PATHS)
    @pytest.mark.parametrize(("name", "kind"), KINDS)
    def test_the_wrapper_has_the_kind_of_the_error_class(
        self,
        backend: LlmSdkBackend,
        sdk: fakes.FakeSdk,
        attr: str,
        call: Callable[[LlmSdkBackend], object],
        name: str,
        kind: ProviderErrorKind,
    ) -> None:
        _arm(attr)(sdk, sdk.error(name))
        with pytest.raises(ProviderError) as err:
            call(backend)
        assert err.value.kind is kind

    @pytest.mark.parametrize(("attr", "call"), PATHS)
    def test_a_foreign_error_is_unknown(
        self,
        backend: LlmSdkBackend,
        sdk: fakes.FakeSdk,
        attr: str,
        call: Callable[[LlmSdkBackend], object],
    ) -> None:
        _arm(attr)(sdk, RuntimeError("odd"))
        with pytest.raises(ProviderError, match="failed: odd") as err:
            call(backend)
        assert err.value.kind is ProviderErrorKind.UNKNOWN


class TestUsage:
    def test_complete_reports_prompt_then_completion_tokens(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("x", usage=fakes.Usage(31, 4))
        usage = backend.complete(_completion_request()).usage
        assert (usage.prompt_tokens, usage.completion_tokens) == (31, 4)

    def test_complete_without_usage_reports_zero(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.chat_result = fakes.text_response("x")
        usage = backend.complete(_completion_request()).usage
        assert (usage.prompt_tokens, usage.completion_tokens) == (0, 0)

    def test_the_closing_stream_chunk_reports_prompt_then_completion_tokens(
        self, backend: LlmSdkBackend, sdk: fakes.FakeSdk
    ) -> None:
        sdk.stream_chunks = [fakes.chunk("x"), fakes.usage_chunk(fakes.Usage(12, 30))]
        chunks = list(backend.complete_stream(_completion_request()))
        assert chunks[0].usage is None
        usage = chunks[-1].usage
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (12, 30)
