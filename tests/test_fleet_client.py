"""Tests for the multi-GPU httpx llama-server client."""

from __future__ import annotations

import json

import httpx
import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.client import (
    LlamaServerClient,
    _first_token_top_logprobs,
    _llm_rerank_score,
    _parse_sse_delta,
)
from lilbee.providers.roles import RerankMode

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


def test_llm_rerank_score_softmax_yes_over_no() -> None:
    top = [{"token": "yes", "logprob": 0.0}, {"token": "no", "logprob": -2.0}]
    assert 0.8 < _llm_rerank_score(top) < 1.0


def test_llm_rerank_score_softmax_no_over_yes() -> None:
    top = [{"token": "No", "logprob": 0.0}, {"token": " yes", "logprob": -3.0}]
    assert _llm_rerank_score(top) < 0.2


def test_llm_rerank_score_only_yes_present() -> None:
    assert _llm_rerank_score([{"token": "yes", "logprob": -0.5}]) > 0.0


def test_llm_rerank_score_neither_present_is_zero() -> None:
    assert _llm_rerank_score([{"token": "maybe", "logprob": -0.1}]) == 0.0


def test_first_token_top_logprobs_empty_on_missing_fields() -> None:
    assert _first_token_top_logprobs({}) == []
    assert _first_token_top_logprobs({"choices": [{}]}) == []
    assert _first_token_top_logprobs({"choices": [{"logprobs": {"content": []}}]}) == []


def _llm_rerank_client(handler) -> LlamaServerClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    return LlamaServerClient("http://gpu0", "rerank-0", http=http, rerank_mode=RerankMode.LLM)


def _chat_logprobs_response(yes_lp: float, no_lp: float) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": "yes", "logprob": yes_lp},
                                    {"token": "no", "logprob": no_lp},
                                ]
                            }
                        ]
                    }
                }
            ]
        },
    )


def test_llm_rerank_routes_to_chat_and_ranks_relevant_higher() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        body = json.loads(request.content)
        assert body["max_tokens"] == 1
        assert body["logprobs"] is True
        content = body["messages"][0]["content"]
        if "relevant doc" in content:
            return _chat_logprobs_response(0.0, -4.0)
        return _chat_logprobs_response(-4.0, 0.0)

    scores = _llm_rerank_client(handler).rerank("q", ["relevant doc", "off-topic"])
    assert seen_paths == ["/v1/chat/completions", "/v1/chat/completions"]
    assert scores[0] > scores[1]


def test_llm_rerank_uses_prompt_override(monkeypatch) -> None:
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "reranker_prompt", "Q:{query} D:{document}")
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body["messages"][0]["content"])
        return _chat_logprobs_response(0.0, -1.0)

    _llm_rerank_client(handler).rerank("hello", ["world"])
    assert captured == ["Q:hello D:world"]


def test_rerank_empty_candidates_returns_empty() -> None:
    assert _client().rerank("q", []) == []


def test_rerank_raises_on_count_mismatch() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [0.5]}]})  # 1 for 2 pairs

    with pytest.raises(ProviderError, match="vectors for"):
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


def test_chat_result_reads_usage_from_response() -> None:
    """chat_result threads llama-server's usage block into ChatResult. (F4)"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                },
            )
        return httpx.Response(404)

    result = _client(handler).chat_result([{"role": "user", "content": "hi"}])
    assert result.text == "Hi"
    assert result.usage.prompt_tokens == 7
    assert result.usage.completion_tokens == 3


def test_chat_result_usage_defaults_to_zero_when_absent() -> None:
    result = _client().chat_result([{"role": "user", "content": "hi"}])
    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0


def test_chat_stream_items_yields_usage_terminator_frame() -> None:
    """The include_usage terminator chunk surfaces as a final TokenUsage frame. (F4)"""
    from lilbee.providers.base import TokenUsage

    body = (
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, text=body)
        return httpx.Response(404)

    frames = list(_client(handler).chat_stream_items([{"role": "user", "content": "hi"}]))
    assert frames[0] == "Hi"
    assert frames[-1] == TokenUsage(prompt_tokens=4, completion_tokens=1)


def test_chat_stream_items_requests_include_usage() -> None:
    """The stream request opts into include_usage so the server emits usage."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            seen.update(json.loads(request.content))
            return httpx.Response(200, text="data: [DONE]\n\n")
        return httpx.Response(404)

    list(_client(handler).chat_stream_items([{"role": "user", "content": "hi"}]))
    assert seen["stream_options"] == {"include_usage": True}


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
        if request.url.path.endswith("/tokenize"):
            # 5 tokens for the long text, more than the cap of 3
            return httpx.Response(200, json={"tokens": [10, 11, 12, 13, 14]})
        if request.url.path.endswith("/detokenize"):
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
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"tokens": [1, 2]})  # 2 <= cap 3
        if request.url.path.endswith("/detokenize"):
            raise AssertionError("must not detokenize an input within the cap")
        seen["embedded"] = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [{"embedding": [0.5]}]})

    assert _capped_client(handler, cap=3).embed(["short"]) == [[0.5]]
    assert seen["embedded"] == ["short"]  # original text passed through


def test_embed_tokenize_request_matches_in_process_flags() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/tokenize"):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"tokens": [1]})
        return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})

    # A long input whose char estimate exceeds the cap forces the /tokenize probe.
    _capped_client(handler, cap=2).embed(["x" * 30])
    assert seen["add_special"] is True
    assert seen["parse_special"] is False


def test_embed_estimates_tokens_without_per_input_tokenize() -> None:
    # Bulk ingest must not pay a /tokenize round-trip per chunk: a normal-sized
    # input's token count is estimated from its char length, so /tokenize is
    # never hit and every input embeds in one pass.
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(("/tokenize", "/detokenize")):
            raise AssertionError("normal-sized chunks must not hit the server tokenizer")
        inputs = json.loads(request.content)["input"]
        sent.append(list(inputs))
        return httpx.Response(200, json={"data": [{"embedding": [0.5]} for _ in inputs]})

    chunks = ["a normal chunk of text"] * 5
    out = _capped_client(handler, cap=8192).embed(chunks)
    assert len(out) == 5
    assert [c for batch in sent for c in batch] == chunks  # order preserved


def test_rerank_truncates_oversize_pairs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"tokens": list(range(9))})  # 9 > cap 4
        if request.url.path.endswith("/detokenize"):
            return httpx.Response(200, json={"content": "t"})
        # one score per (truncated) pair
        return httpx.Response(200, json={"data": [{"embedding": [0.9]} for _ in body["input"]]})

    assert _capped_client(handler, cap=4).rerank("q", ["cand"]) == [0.9]


def test_embed_surfaces_server_error_body() -> None:
    # raise_for_status drops the body, so a 500 used to surface as a bare
    # "Internal Server Error" with no cause. We must include the server's message.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="input is too large to process. increase the n_batch")

    with pytest.raises(ProviderError, match="too large to process"):
        _client(handler).embed(["a"])


def test_raise_for_status_tags_context_overflow_400() -> None:
    # llama-server reports an oversize prompt as a 400 carrying the
    # exceed_context_size_error type. It must surface as CONTEXT_OVERFLOW with a
    # user-facing message, not a generic internal error.
    from lilbee.providers.base import ProviderErrorKind
    from lilbee.providers.fleet.client import _raise_for_status

    resp = httpx.Response(400, text='{"error":{"type":"exceed_context_size_error"}}')
    with pytest.raises(ProviderError) as excinfo:
        _raise_for_status(resp)
    assert excinfo.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert "context window" in str(excinfo.value).lower()


def test_chat_stream_surfaces_server_error_body() -> None:
    # On a streaming request the response body isn't read yet, so the error path
    # must read it before extracting the message (the ResponseNotRead branch).
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model is still loading")

    with pytest.raises(ProviderError, match="model is still loading"):
        list(_client(handler).chat([{"role": "user", "content": "hi"}], stream=True))


def test_embed_subbatches_when_token_budget_exceeded() -> None:
    # Sub-batching still fits the server's n_batch, now driven by the char-length
    # token estimate rather than /tokenize. With the cap at 4 tokens and the
    # estimate at ~1 token per 3 chars, each 7-char input estimates to 3 tokens,
    # so two cannot share a request (3+3 > 4) and each lands on its own.
    sent: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(("/tokenize", "/detokenize")):
            raise AssertionError("estimation must not hit the server tokenizer here")
        inputs = json.loads(request.content)["input"]
        sent.append(list(inputs))
        return httpx.Response(200, json={"data": [{"embedding": [0.0]} for _ in inputs]})

    out = _capped_client(handler, cap=4).embed(["x" * 7, "y" * 7, "z" * 7])
    assert sent == [["x" * 7], ["y" * 7], ["z" * 7]]  # 3+3 > 4, so never two per request
    assert len(out) == 3  # one vector per input, order preserved


def test_embed_subbatches_when_sequence_count_exceeded() -> None:
    # Many tiny chunks (1 token each) stay under the token budget but must still
    # split at _EMBED_N_SEQ_MAX sequences per request.
    from lilbee.providers.fleet.client import _EMBED_N_SEQ_MAX

    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"tokens": [1]})  # 1 token each
        sizes.append(len(body["input"]))
        return httpx.Response(200, json={"data": [{"embedding": [0.0]} for _ in body["input"]]})

    n = _EMBED_N_SEQ_MAX + 5
    out = _capped_client(handler, cap=100_000).embed([f"t{i}" for i in range(n)])
    assert sizes == [_EMBED_N_SEQ_MAX, 5]  # capped at the sequence limit per request
    assert len(out) == n


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
    from lilbee.providers.fleet.client import _parse_native_tool_calls

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
    from lilbee.providers.fleet.client import _parse_native_tool_calls

    assert _parse_native_tool_calls(None) == []
    assert _parse_native_tool_calls("nope") == []


def test_arguments_to_str_variants() -> None:
    from lilbee.providers.fleet.client import _arguments_to_str

    assert _arguments_to_str('{"a":1}') == '{"a":1}'
    assert _arguments_to_str(None) == "{}"
    assert _arguments_to_str({"a": 1}) == '{"a": 1}'


def test_recover_bare_json_list_of_calls() -> None:
    from lilbee.providers.fleet.client import _recover_bare_json_tool_calls

    result = _recover_bare_json_tool_calls('[{"name":"a"},{"parameters":{"x":1},"name":"b"}]')
    assert [c.name for c in result.tool_calls] == ["a", "b"]
    assert result.tool_calls[0].arguments == "{}"  # no args/params -> empty object
    assert result.tool_calls[1].arguments == '{"x": 1}'


@pytest.mark.parametrize("content", ["hello world", '{"name": ', '{"foo": 1}', "   "])
def test_recover_bare_json_leaves_non_calls_as_text(content: str) -> None:
    from lilbee.providers.fleet.client import _recover_bare_json_tool_calls

    result = _recover_bare_json_tool_calls(content)
    assert result.content == content
    assert result.tool_calls == []


def test_chat_result_text_only_carries_finish_reason() -> None:
    from lilbee.providers.base import FinishReason

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}]},
        )

    result = _client(handler).chat_result([{"role": "user", "content": "x"}])
    assert result.text == "hi there"
    assert result.tool_calls == ()
    assert result.finish_reason is FinishReason.STOP


def test_chat_result_surfaces_native_tool_calls_and_finish_reason() -> None:
    from lilbee.providers.base import FinishReason

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"]  # tools forwarded
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
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    result = _client(handler).chat_result([{"role": "user", "content": "weather?"}], tools=_tools())
    assert result.text == ""
    assert result.finish_reason is FinishReason.TOOL_CALLS
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == '{"city":"SF"}'


def test_chat_result_recovers_bare_json_call_as_tool_calls_finish() -> None:
    from lilbee.providers.base import FinishReason

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"name":"get_weather","arguments":{"city":"SF"}}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    result = _client(handler).chat_result([{"role": "user", "content": "x"}], tools=_tools())
    # A bare-JSON native miss is recovered as a tool call and re-flagged TOOL_CALLS.
    assert result.text == ""
    assert result.finish_reason is FinishReason.TOOL_CALLS
    assert result.tool_calls[0].name == "get_weather"


def test_chat_result_unknown_finish_reason_falls_back_to_stop() -> None:
    from lilbee.providers.base import FinishReason

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "x"}, "finish_reason": "weird"}]},
        )

    result = _client(handler).chat_result([{"role": "user", "content": "x"}])
    assert result.finish_reason is FinishReason.STOP


def test_chat_stream_items_yields_text_and_tool_call_deltas() -> None:
    from lilbee.providers.base import ToolCallDelta

    stream_body = (
        'data: {"choices":[{"delta":{"content":"He"}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":['
        '{"index":0,"id":"c1","function":{"name":"get_weather","arguments":""}}]}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":['
        '{"index":0,"function":{"arguments":"{\\"city\\""}}]}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":['
        '{"index":0,"function":{"arguments":":\\"SF\\"}"}}]}}]}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["tools"]
        return httpx.Response(200, text=stream_body)

    items = list(
        _client(handler).chat_stream_items(
            [{"role": "user", "content": "weather?"}], tools=_tools()
        )
    )
    assert items[0] == "He"
    deltas = [i for i in items if isinstance(i, ToolCallDelta)]
    # Opener carries id+name; subsequent frames accumulate arguments by index.
    assert deltas[0].index == 0
    assert deltas[0].id == "c1"
    assert deltas[0].name == "get_weather"
    assert deltas[1].name is None  # continuation frames have no name
    assert "".join(d.arguments_delta or "" for d in deltas) == '{"city":"SF"}'


def test_chat_stream_items_text_only() -> None:
    items = list(_client().chat_stream_items([{"role": "user", "content": "hi"}]))
    assert items == ["He", "llo"]


def test_chat_result_forwards_tool_choice() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    _client(handler).chat_result(
        [{"role": "user", "content": "x"}], tools=_tools(), tool_choice="required"
    )
    assert seen["tool_choice"] == "required"


def test_chat_stream_items_forwards_tool_choice() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, text="data: [DONE]\n\n")

    list(
        _client(handler).chat_stream_items(
            [{"role": "user", "content": "x"}], tools=_tools(), tool_choice={"type": "function"}
        )
    )
    assert seen["tool_choice"] == {"type": "function"}


def test_coerce_finish_reason_non_string_is_stop() -> None:
    from lilbee.providers.base import FinishReason
    from lilbee.providers.fleet.client import _coerce_finish_reason

    # A missing finish_reason (None) or any non-string falls back to STOP.
    assert _coerce_finish_reason(None) is FinishReason.STOP
    assert _coerce_finish_reason(42) is FinishReason.STOP


def test_parse_sse_stream_items_skips_malformed_json() -> None:
    from lilbee.providers.fleet.client import _parse_sse_stream_items

    assert list(_parse_sse_stream_items("data: {not json")) == []


def test_parse_sse_stream_items_skips_empty_choices() -> None:
    from lilbee.providers.fleet.client import _parse_sse_stream_items

    assert list(_parse_sse_stream_items('data: {"choices": []}')) == []


def test_tokenize_and_detokenize_use_upstream_route() -> None:
    """llama-swap proxies the native /tokenize + /detokenize routes only under
    /upstream/<model>/...; the bare paths 404 (bb-4pw)."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/tokenize"):
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        if request.url.path.endswith("/detokenize"):
            return httpx.Response(200, json={"content": "hi"})
        return httpx.Response(404)

    client = _client(handler)
    client._tokenize("hello")
    client._detokenize([1, 2, 3])
    assert "/upstream/test-model/tokenize" in seen
    assert "/upstream/test-model/detokenize" in seen


def test_embed_retries_with_exact_tokenize_on_context_overflow() -> None:
    """A token-dense input the char estimate trusts can overflow the context; embed
    retries that batch with exact server-side tokenization so it truncates (bb-54r)."""
    calls = {"embed": 0, "tokenize": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/tokenize"):
            calls["tokenize"] += 1
            return httpx.Response(200, json={"tokens": list(range(20))})
        if path.endswith("/detokenize"):
            return httpx.Response(200, json={"content": "t"})
        if path == "/v1/embeddings":
            calls["embed"] += 1
            if calls["embed"] == 1:
                return httpx.Response(400, text='{"error":{"message":"exceed_context_size_error"}}')
            return httpx.Response(200, json={"data": [{"embedding": [0.5]}]})
        return httpx.Response(404)

    # cap=10 so "dense" (est 2) is trusted and sent untruncated on the first try.
    out = _capped_client(handler, 10).embed(["dense"])
    assert out == [[0.5]]
    assert calls["embed"] == 2  # first overflowed, retry succeeded
    assert calls["tokenize"] >= 1  # retry used exact tokenization


def test_embed_does_not_retry_on_non_overflow_error() -> None:
    """A non-overflow embed error propagates without a tokenize-retry."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embeddings":
            return httpx.Response(500, text="boom")
        return httpx.Response(404)

    with pytest.raises(ProviderError, match="500"):
        _capped_client(handler, 10).embed(["x"])
