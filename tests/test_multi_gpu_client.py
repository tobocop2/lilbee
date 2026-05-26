"""Tests for the multi-GPU httpx llama-server client."""

from __future__ import annotations

import json

import httpx
import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.multi_gpu.client import LlamaServerClient, _parse_sse_delta

_STREAM_BODY = (
    'data: {"choices":[{"delta":{"content":"He"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
    "data: [DONE]\n\n"
)


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/health":
        return httpx.Response(200)
    if path == "/v1/chat/completions":
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, text=_STREAM_BODY)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})
    if path == "/v1/embeddings":
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3]}]})
    return httpx.Response(404)


def _client(handler=_handler) -> LlamaServerClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    return LlamaServerClient("http://gpu0", "test-model", http=http)


def test_rerank_scores_pairs_via_rank_pooling() -> None:
    seen: dict[str, list] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["input"] = body["input"]
        # rank pooling returns a 1-element embedding (the score) per pair, in order
        n = len(body["input"])
        return httpx.Response(200, json={"data": [{"embedding": [float(i)]} for i in range(n)]})

    scores = _client(handler).rerank("q", ["a", "b", "c"])
    assert scores == [0.0, 1.0, 2.0]
    # mirrors the in-process pairing exactly
    assert seen["input"] == ["q</s></s>a", "q</s></s>b", "q</s></s>c"]


def test_rerank_empty_candidates_returns_empty() -> None:
    assert _client().rerank("q", []) == []


def test_rerank_raises_on_count_mismatch() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [0.5]}]})  # 1 for 2 pairs

    with pytest.raises(ProviderError, match="entries for"):
        _client(handler).rerank("q", ["a", "b"])


def test_rerank_raises_on_bad_score_shape() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": "nope"}]})

    with pytest.raises(ProviderError, match="unexpected score shape"):
        _client(handler).rerank("q", ["a"])


def test_in_flight_counter_is_atomic_under_threads() -> None:
    import threading

    c = _client()

    def _work() -> None:
        for _ in range(2000):
            with c._track():
                pass

    threads = [threading.Thread(target=_work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Every enter is balanced by an exit; a lost read-modify-write under the
    # racing threads would leave a non-zero residual.
    assert c.in_flight == 0


def test_health_true_on_200() -> None:
    assert _client().health() is True


def test_health_false_on_non_200() -> None:
    assert _client(lambda _r: httpx.Response(503)).health() is False


def test_health_false_on_transport_error() -> None:
    def _raise(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert _client(_raise).health() is False


def test_chat_returns_content() -> None:
    assert _client().chat([{"role": "user", "content": "hi"}]) == "Hello"


def test_chat_stream_yields_deltas() -> None:
    chunks = list(_client().chat([{"role": "user", "content": "hi"}], stream=True))
    assert chunks == ["He", "llo"]


def test_embed_returns_vectors() -> None:
    assert _client().embed(["a", "b"]) == [[0.1, 0.2], [0.3]]


def test_embed_empty_returns_empty_without_request() -> None:
    # The server rejects an empty input; in-process returns [] for no texts.
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("embed([]) must not hit the server")

    assert _client(handler).embed([]) == []


def _capped_client(handler, cap: int) -> LlamaServerClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    return LlamaServerClient("http://gpu0", "m", http=http, token_cap=cap)


def test_embed_truncates_oversize_input_via_server_tokenizer() -> None:
    # Mirrors the in-process backstop: an input longer than the context is
    # token-truncated before embedding so the server does not error.
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            # 5 tokens for the long text, more than the cap of 3
            return httpx.Response(200, json={"tokens": [10, 11, 12, 13, 14]})
        if request.url.path == "/detokenize":
            seen["detok_tokens"] = body["tokens"]
            return httpx.Response(200, json={"content": "trunc"})
        seen["embedded"] = body["input"]
        return httpx.Response(200, json={"data": [{"embedding": [0.5]}]})

    out = _capped_client(handler, cap=3).embed(["a very long input"])
    assert out == [[0.5]]
    assert seen["detok_tokens"] == [10, 11, 12]  # first cap tokens
    assert seen["embedded"] == ["trunc"]


def test_embed_keeps_input_within_cap_unchanged() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2]})  # 2 <= cap 3
        if request.url.path == "/detokenize":
            raise AssertionError("must not detokenize an input within the cap")
        seen["embedded"] = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [{"embedding": [0.5]}]})

    assert _capped_client(handler, cap=3).embed(["short"]) == [[0.5]]
    assert seen["embedded"] == ["short"]  # original text passed through


def test_embed_tokenize_request_matches_in_process_flags() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tokenize":
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

    _capped_client(handler, cap=10).embed(["x"])
    assert seen["add_special"] is True
    assert seen["parse_special"] is False


def test_rerank_truncates_oversize_pairs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(9))})  # 9 > cap 4
        if request.url.path == "/detokenize":
            return httpx.Response(200, json={"content": "t"})
        # one score per (truncated) pair
        return httpx.Response(200, json={"data": [{"embedding": [0.9]} for _ in body["input"]]})

    assert _capped_client(handler, cap=4).rerank("q", ["cand"]) == [0.9]


def test_embed_requests_no_normalization() -> None:
    # Match in-process create_embedding (normalize=False): the server defaults to
    # L2, so we must send embd_normalize=-1 or the stored vectors diverge.
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

    _client(handler).embed(["a"])
    assert seen["embd_normalize"] == -1


def test_rerank_requests_no_normalization() -> None:
    # L2-normalizing a 1-element rank score would collapse it to +-1; -1 keeps the
    # raw score, matching the in-process reranker.
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [0.7]}]})

    _client(handler).rerank("q", ["a"])
    assert seen["embd_normalize"] == -1


def test_in_flight_resets_after_chat() -> None:
    c = _client()
    assert c.chat([{"role": "user", "content": "hi"}]) == "Hello"
    assert c.in_flight == 0


def test_close_closes_owned_client() -> None:
    c = LlamaServerClient("http://gpu0", "m")  # owns its httpx.Client
    c.close()
    assert c._http.is_closed


def test_close_leaves_injected_client_open() -> None:
    http = httpx.Client(transport=httpx.MockTransport(_handler))
    c = LlamaServerClient("http://gpu0", "m", http=http)
    c.close()
    assert not http.is_closed


class TestParseSseDelta:
    def test_non_data_line_is_empty(self) -> None:
        assert _parse_sse_delta("event: message") == ""

    def test_done_sentinel_is_empty(self) -> None:
        assert _parse_sse_delta("data: [DONE]") == ""

    def test_invalid_json_is_empty(self) -> None:
        assert _parse_sse_delta("data: {not json") == ""

    def test_no_choices_is_empty(self) -> None:
        assert _parse_sse_delta('data: {"choices": []}') == ""

    def test_extracts_content(self) -> None:
        assert _parse_sse_delta('data: {"choices":[{"delta":{"content":"hi"}}]}') == "hi"


def _tools() -> list[dict]:
    return [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


def test_chat_tools_parses_native_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"]  # tools are forwarded to the server
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"SF"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    result = _client(handler).chat_tools([{"role": "user", "content": "weather?"}], tools=_tools())
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "c1"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == '{"city":"SF"}'


def test_chat_tools_forwards_tool_choice() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    _client(handler).chat_tools(
        [{"role": "user", "content": "x"}], tools=_tools(), tool_choice="required"
    )
    assert seen["tool_choice"] == "required"


def test_chat_tools_text_only_response_has_no_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "just text"}}]})

    result = _client(handler).chat_tools([{"role": "user", "content": "x"}], tools=_tools())
    assert result.content == "just text"
    assert result.tool_calls == []


def test_chat_tools_recovers_bare_json_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"name":"get_weather","arguments":{"city":"SF"}}'}}
                ]
            },
        )

    result = _client(handler).chat_tools([{"role": "user", "content": "x"}], tools=_tools())
    assert result.content == ""
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == '{"city": "SF"}'


def test_parse_native_tool_calls_skips_malformed() -> None:
    from lilbee.providers.multi_gpu.client import _parse_native_tool_calls

    raw = [
        "notadict",
        {"function": "notamap"},
        {"function": {"name": ""}},  # empty name -> skipped
        {"function": {"name": "f", "arguments": {"a": 1}}},  # dict args -> json string
        {"id": "x", "function": {"name": "g", "arguments": "{}"}},
    ]
    calls = _parse_native_tool_calls(raw)
    assert [c.name for c in calls] == ["f", "g"]
    assert calls[0].id == "call_3"  # synthetic id keeps the source index
    assert calls[0].arguments == '{"a": 1}'
    assert calls[1].id == "x"


def test_parse_native_tool_calls_non_list_returns_empty() -> None:
    from lilbee.providers.multi_gpu.client import _parse_native_tool_calls

    assert _parse_native_tool_calls(None) == []
    assert _parse_native_tool_calls("nope") == []


def test_arguments_to_str_variants() -> None:
    from lilbee.providers.multi_gpu.client import _arguments_to_str

    assert _arguments_to_str('{"a":1}') == '{"a":1}'
    assert _arguments_to_str(None) == "{}"
    assert _arguments_to_str({"a": 1}) == '{"a": 1}'


def test_recover_bare_json_list_of_calls() -> None:
    from lilbee.providers.multi_gpu.client import _recover_bare_json_tool_calls

    result = _recover_bare_json_tool_calls('[{"name":"a"},{"parameters":{"x":1},"name":"b"}]')
    assert [c.name for c in result.tool_calls] == ["a", "b"]
    assert result.tool_calls[0].arguments == "{}"  # no args/params -> empty object
    assert result.tool_calls[1].arguments == '{"x": 1}'


@pytest.mark.parametrize("content", ["hello world", '{"name": ', '{"foo": 1}', "   "])
def test_recover_bare_json_leaves_non_calls_as_text(content: str) -> None:
    from lilbee.providers.multi_gpu.client import _recover_bare_json_tool_calls

    result = _recover_bare_json_tool_calls(content)
    assert result.content == content
    assert result.tool_calls == []
