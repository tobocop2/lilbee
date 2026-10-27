"""The adapter against liter-llm's own response types, parsed from recorded wire bodies.

``_fake_sdk`` stands in for the library where the extra is absent. These tests run
wherever the extra is installed and feed the adapter the real typed objects, so a
value the library wraps (message text, finish reason) cannot reach lilbee unconverted.
The bodies are OpenAI-compatible wire bodies (the shape LM Studio answers). Ollama's
native route is covered by the integration suite.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.backends.liter_llm.client import _client
from lilbee.providers.factory import sdk_library
from lilbee.providers.model_ref import format_remote_ref, parse_model_ref
from lilbee.providers.sdk_backend import CompletionRequest, EmbeddingRequest, RerankRequest
from tests import _fake_sdk as fakes

liter_llm = pytest.importorskip(sdk_library().module)

_BASE = "http://localhost:11434"
_CHAT_BODY: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "qwen2.5:1.5b",
    "system_fingerprint": "fp_ollama",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Pong."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 31, "completion_tokens": 4, "total_tokens": 35},
}
_TOOL_BODY: dict[str, Any] = {
    **_CHAT_BODY,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_7bfud2cr",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}
_STREAM_BODIES: list[dict[str, Any]] = [
    {
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen2.5:1.5b",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "Po"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen2.5:1.5b",
        "choices": [{"index": 0, "delta": {"content": "ng"}, "finish_reason": None}],
    },
    {
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen2.5:1.5b",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
    },
    {
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen2.5:1.5b",
        "choices": [],
        "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42},
    },
]
_EMBED_BODY: dict[str, Any] = {
    "object": "list",
    "model": "all-minilm",
    "data": [
        {"object": "embedding", "index": 1, "embedding": [0.5, 0.25]},
        {"object": "embedding", "index": 0, "embedding": [0.125, 0.0625]},
    ],
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}
_RERANK_BODY: dict[str, Any] = {
    "id": "rr-1",
    "results": [{"index": 1, "relevance_score": 0.75}, {"index": 0, "relevance_score": 0.25}],
}


def _json(body: dict[str, Any]) -> str:
    return json.dumps(body)


@pytest.fixture(autouse=True)
def _fresh_clients() -> Iterator[None]:
    _client.cache_clear()
    yield
    _client.cache_clear()


def _answer(monkeypatch: pytest.MonkeyPatch, name: str, result: Any) -> None:
    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        return result

    monkeypatch.setattr(liter_llm, name, _call)


def _stream(monkeypatch: pytest.MonkeyPatch, bodies: list[dict[str, Any]]) -> None:
    async def _chunks() -> AsyncIterator[Any]:
        for body in bodies:
            yield liter_llm.ChatCompletionChunk.from_json(_json(body))

    monkeypatch.setattr(liter_llm, "chat_stream", lambda *_args, **_kwargs: _chunks())


def _completion(options: dict[str, Any] | None = None) -> CompletionRequest:
    return CompletionRequest(
        ref=parse_model_ref("ollama/qwen2.5:1.5b"),
        messages=[{"role": "user", "content": "Say pong"}],
        api_base=_BASE,
        options=options or {},
    )


def _record_chat(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Answer chat with the recorded reply and keep each real request object."""
    sent: list[Any] = []
    reply = liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY))

    async def _call(_client: Any, request: Any) -> Any:
        sent.append(request)
        return reply

    monkeypatch.setattr(liter_llm, "chat", _call)
    return sent


class TestCompleteOnRealTypes:
    def test_a_text_reply_gives_its_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY)))
        result = LiterLlmBackend().complete(_completion())
        assert result.content == "Pong."

    def test_a_parts_reply_gives_its_text_parts_joined(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parts = [{"type": "text", "text": "Po"}, {"type": "text", "text": "ng."}]
        choice = {"index": 0, "message": {"role": "assistant", "content": parts}}
        body = {**_CHAT_BODY, "choices": [{**choice, "finish_reason": "stop"}]}
        _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(body)))
        assert LiterLlmBackend().complete(_completion()).content == "Pong."

    def test_finish_reason_and_model_are_plain_strings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY)))
        result = LiterLlmBackend().complete(_completion())
        assert (type(result.content), type(result.finish_reason)) == (str, str)
        assert (result.finish_reason, result.model) == ("stop", "qwen2.5:1.5b")

    def test_usage_is_the_reply_token_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY)))
        usage = LiterLlmBackend().complete(_completion()).usage
        assert (usage.prompt_tokens, usage.completion_tokens) == (31, 4)
        assert {type(usage.prompt_tokens), type(usage.completion_tokens)} == {int}

    def test_every_finish_reason_member_gives_its_wire_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wires = ("stop", "length", "tool_calls", "content_filter", "function_call", "other")
        got = []
        for wire in wires:
            choice = {**_CHAT_BODY["choices"][0], "finish_reason": wire}
            body = {**_CHAT_BODY, "choices": [choice]}
            _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(body)))
            got.append(LiterLlmBackend().complete(_completion()).finish_reason)
        assert got == list(wires)

    def test_reasoning_effort_low_is_the_typed_request_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = _record_chat(monkeypatch)
        LiterLlmBackend().complete(_completion(options={"reasoning_effort": "low"}))
        assert sent[0].reasoning_effort == liter_llm.ReasoningEffort.LOW

    def test_reasoning_effort_none_turns_thinking_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = _record_chat(monkeypatch)
        LiterLlmBackend().complete(_completion(options={"reasoning_effort": "none"}))
        assert sent[0].reasoning_effort == liter_llm.ReasoningEffort.NONE

    def test_tool_call_fields_are_plain_strings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(_TOOL_BODY)))
        result = LiterLlmBackend().complete(_completion())
        assert result.content == ""
        assert [
            (type(c.id), type(c.name), type(c.arguments), c.name, c.arguments)
            for c in result.tool_calls
        ] == [(str, str, str, "get_weather", '{"city":"Paris"}')]
        assert result.finish_reason == "tool_calls"

    def test_a_reply_without_text_is_an_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = {**_CHAT_BODY, "choices": [{"index": 0, "message": {"role": "assistant"}}]}
        _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(body)))
        result = LiterLlmBackend().complete(_completion())
        assert (result.content, result.finish_reason) == ("", None)


class TestStreamOnRealTypes:
    def test_chunk_text_and_finish_reason_are_plain_strings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stream(monkeypatch, _STREAM_BODIES)
        chunks = list(LiterLlmBackend().complete_stream(_completion()))
        assert [(c.content, c.finish_reason) for c in chunks] == [
            ("Po", None),
            ("ng", None),
            ("", "length"),
            ("", None),
        ]
        assert {type(c.content) for c in chunks} == {str}
        assert type(chunks[2].finish_reason) is str
        usage = chunks[-1].usage
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (12, 30)

    def test_tool_call_deltas_are_plain_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opener = {
            **_STREAM_BODIES[0],
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"c'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        _stream(monkeypatch, [opener])
        (chunk,) = LiterLlmBackend().complete_stream(_completion())
        (delta,) = chunk.tool_call_deltas
        assert (delta.index, delta.id, delta.name, delta.arguments_delta) == (
            0,
            "call_1",
            "get_weather",
            '{"c',
        )
        assert {type(delta.id), type(delta.name), type(delta.arguments_delta)} == {str}


class TestEmbedAndRerankOnRealTypes:
    def test_vectors_come_back_in_input_order_as_floats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answer(monkeypatch, "embed", liter_llm.EmbeddingResponse.from_json(_json(_EMBED_BODY)))
        result = LiterLlmBackend().embed(
            EmbeddingRequest(ref=parse_model_ref("ollama/all-minilm"), inputs=["a", "b"])
        )
        assert result.vectors == [[0.125, 0.0625], [0.5, 0.25]]
        assert {type(v) for vector in result.vectors for v in vector} == {float}
        assert (type(result.model), result.model) == (str, "all-minilm")

    def test_scores_come_back_in_candidate_order_as_floats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answer(monkeypatch, "rerank", liter_llm.RerankResponse.from_json(_json(_RERANK_BODY)))
        result = LiterLlmBackend().rerank(
            RerankRequest(
                ref=parse_model_ref("openai/rerank-x"),
                query="q",
                candidates=["a", "b"],
                api_key="k",
            )
        )
        assert result.scores == [0.25, 0.75]
        assert {type(s) for s in result.scores} == {float}


def _members(owner: Any) -> set[str]:
    return {name for name in dir(owner) if name.isupper()}


def _accepts(effort: Any, value: str) -> bool:
    try:
        effort(value)
    except ValueError:
        return False
    return True


class TestFakeMatchesTheLibrary:
    """The stand-in declares what the library declares, so a drifted fake fails here."""

    @staticmethod
    def _real_content() -> Any:
        return (
            liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY)).choices[0].message.content
        )

    def test_reply_content_has_the_same_accessors_and_is_no_str(self) -> None:
        real = self._real_content()
        fake = fakes.text_response("Pong.").choices[0].message.content
        for content in (real, fake):
            assert {"text", "parts"} <= set(dir(content))
            assert not isinstance(content, str)
        assert type(fake).__str__ is object.__str__

    def test_finish_reason_members_are_the_same_and_no_str(self) -> None:
        real = liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY)).choices[0]
        fake = fakes.text_response("Pong.").choices[0]
        assert _members(liter_llm.FinishReason)
        assert _members(liter_llm.FinishReason) == _members(fakes.FinishReason)
        assert not isinstance(real.finish_reason, str)
        assert not isinstance(fake.finish_reason, str)

    def test_reasoning_effort_members_and_refusals_are_the_same(self) -> None:
        assert _members(liter_llm.ReasoningEffort)
        assert _members(liter_llm.ReasoningEffort) == _members(fakes.ReasoningEffort)
        for value in ("low", "none"):
            assert _accepts(liter_llm.ReasoningEffort, value) == _accepts(
                fakes.ReasoningEffort, value
            )

    def test_streamed_text_is_a_str_in_both(self) -> None:
        real = liter_llm.ChatCompletionChunk.from_json(_json(_STREAM_BODIES[0])).choices[0]
        fake = fakes.chunk("Po").choices[0]
        assert type(real.delta.content) is str
        assert type(fake.delta.content) is str

    def test_usage_fields_are_the_same_ints(self) -> None:
        real = liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY)).usage
        fake = fakes.Usage(31, 4)
        for usage in (real, fake):
            assert (usage.prompt_tokens, usage.completion_tokens) == (31, 4)


class TestCatalogOnRealTypes:
    """The listing against the library's own ``CatalogModel`` and its embedded catalog."""

    def test_fake_entry_has_the_fields_the_adapter_reads(self) -> None:
        real = liter_llm.CatalogModel(provider="openai", model="gpt-4o", mode="chat")
        fake = fakes.catalog_model("openai", "gpt-4o")
        for field in ("key", "provider", "model", "mode"):
            assert hasattr(real, field)
            assert hasattr(fake, field)
        assert type(real.mode) is type(fake.mode) is str

    def test_adapter_reads_a_real_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entries = [
            liter_llm.CatalogModel(provider="openai", model="b-chat", mode="chat"),
            liter_llm.CatalogModel(provider="openai", model="a-chat", mode="chat"),
            liter_llm.CatalogModel(provider="openai", model="embed", mode="embedding"),
            liter_llm.CatalogModel(provider="openai", model="unknown"),
        ]
        monkeypatch.setattr(liter_llm, "list_catalog_models", lambda _provider=None: entries)
        assert LiterLlmBackend().list_chat_models("openai") == ["a-chat", "b-chat"]

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "mistral", "deepseek"])
    def test_embedded_catalog_lists_chat_models_for_each_keyed_provider(
        self, provider: str
    ) -> None:
        names = LiterLlmBackend().list_chat_models(provider)
        assert names
        assert not any(name.startswith(f"{provider}/") for name in names)

    def test_embedded_catalog_openrouter_names_round_trip_through_routing(self) -> None:
        """OpenRouter ids are nested, and its own meta models start with ``openrouter/``."""
        names = LiterLlmBackend().list_chat_models("openrouter")
        assert "aion-labs/aion-2.0" in names
        assert "openrouter/free" in names
        for name in names:
            ref = parse_model_ref(format_remote_ref(name, "openrouter"))
            assert (ref.provider, ref.name) == ("openrouter", name)

    def test_embedded_catalog_leaves_embedding_models_out(self) -> None:
        assert not [n for n in LiterLlmBackend().list_chat_models("openai") if "embedding" in n]

    def test_an_unknown_provider_lists_nothing(self) -> None:
        assert LiterLlmBackend().list_chat_models("no-such-provider") == []
