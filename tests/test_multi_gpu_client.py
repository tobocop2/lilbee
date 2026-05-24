"""Tests for the multi-GPU httpx llama-server client."""

from __future__ import annotations

import json

import httpx

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
