"""The adapter against liter-llm's own response types, parsed from recorded wire bodies.

``_fake_sdk`` stands in for the library where the extra is absent. These tests run
wherever the extra is installed and feed the adapter the real typed objects, so a
value the library wraps (message text, finish reason) cannot reach lilbee unconverted.
The bodies are the shapes Ollama 0.40.6 answered on ``/v1/chat/completions``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from lilbee.providers.model_ref import parse_model_ref
from lilbee.providers.remote_sdk import SDK_MODULE, RemoteSdkBackend, _client
from lilbee.providers.sdk_backend import CompletionRequest, EmbeddingRequest, RerankRequest
from tests import _fake_sdk as fakes

liter_llm = pytest.importorskip(SDK_MODULE)

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


def _completion() -> CompletionRequest:
    return CompletionRequest(
        ref=parse_model_ref("ollama/qwen2.5:1.5b"),
        messages=[{"role": "user", "content": "Say pong"}],
        api_base=_BASE,
    )


class TestCompleteOnRealTypes:
    def test_message_text_and_finish_reason_are_plain_strings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY)))
        result = RemoteSdkBackend().complete(_completion())
        assert (type(result.content), result.content) == (str, "Pong.")
        assert (type(result.finish_reason), result.finish_reason) == (str, "stop")
        assert (type(result.model), result.model) == (str, "qwen2.5:1.5b")

    def test_tool_call_fields_are_plain_strings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(_TOOL_BODY)))
        result = RemoteSdkBackend().complete(_completion())
        assert result.content == ""
        assert [
            (type(c.id), type(c.name), type(c.arguments), c.name, c.arguments)
            for c in result.tool_calls
        ] == [(str, str, str, "get_weather", '{"city":"Paris"}')]
        assert result.finish_reason == "tool_calls"

    def test_a_reply_without_text_is_an_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = {**_CHAT_BODY, "choices": [{"index": 0, "message": {"role": "assistant"}}]}
        _answer(monkeypatch, "chat", liter_llm.ChatCompletionResponse.from_json(_json(body)))
        result = RemoteSdkBackend().complete(_completion())
        assert (result.content, result.finish_reason) == ("", None)


class TestStreamOnRealTypes:
    def test_chunk_text_and_finish_reason_are_plain_strings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stream(monkeypatch, _STREAM_BODIES)
        chunks = list(RemoteSdkBackend().complete_stream(_completion()))
        assert [(c.content, c.finish_reason) for c in chunks] == [
            ("Po", None),
            ("ng", None),
            ("", "length"),
        ]
        assert {type(c.content) for c in chunks} == {str}
        assert type(chunks[-1].finish_reason) is str

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
        (chunk,) = RemoteSdkBackend().complete_stream(_completion())
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
        result = RemoteSdkBackend().embed(
            EmbeddingRequest(ref=parse_model_ref("ollama/all-minilm"), inputs=["a", "b"])
        )
        assert result.vectors == [[0.125, 0.0625], [0.5, 0.25]]
        assert {type(v) for vector in result.vectors for v in vector} == {float}
        assert (type(result.model), result.model) == (str, "all-minilm")

    def test_scores_come_back_in_candidate_order_as_floats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answer(monkeypatch, "rerank", liter_llm.RerankResponse.from_json(_json(_RERANK_BODY)))
        result = RemoteSdkBackend().rerank(
            RerankRequest(
                ref=parse_model_ref("openai/rerank-x"),
                query="q",
                candidates=["a", "b"],
                api_key="k",
            )
        )
        assert result.scores == [0.25, 0.75]
        assert {type(s) for s in result.scores} == {float}


class TestFakeMatchesTheLibrary:
    """The stand-in's typed values behave like the library's."""

    def test_message_text_is_not_a_str_in_either(self) -> None:
        real = liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY)).choices[0].message
        fake = fakes.text_response("Pong.").choices[0].message
        assert not isinstance(real.content, str)
        assert not isinstance(fake.content, str)
        assert str(real.content) == str(fake.content) == "Pong."

    def test_finish_reason_is_not_a_str_in_either(self) -> None:
        real = liter_llm.ChatCompletionResponse.from_json(_json(_CHAT_BODY)).choices[0]
        fake = fakes.text_response("Pong.").choices[0]
        assert real.finish_reason != "stop"
        assert fake.finish_reason != "stop"
        assert str(real.finish_reason) == str(fake.finish_reason) == "stop"

    def test_streamed_text_is_a_str_in_both(self) -> None:
        real = liter_llm.ChatCompletionChunk.from_json(_json(_STREAM_BODIES[0])).choices[0]
        fake = fakes.chunk("Po").choices[0]
        assert type(real.delta.content) is str
        assert type(fake.delta.content) is str
        assert not isinstance(real.finish_reason, str)
        assert not isinstance(fakes.chunk("x", finish="stop").choices[0].finish_reason, str)
