"""Tests for the multi-GPU httpx llama-server client."""

from __future__ import annotations

import json
from itertools import pairwise

import httpx
import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.client import (
    ChatDeadlineError,
    LlamaServerClient,
    _first_token_top_logprobs,
    _llm_rerank_score,
    _parse_sse_deltas,
    _ThinkInliner,
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


def _unprobed_client(handler=_handler) -> LlamaServerClient:
    """A client whose template has not yet been probed for strict alternation."""
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    return LlamaServerClient("http://gpu0", "test-model", http=http)


def _client(handler=_handler) -> LlamaServerClient:
    # Most chat tests are orthogonal to alternation detection; treat the template
    # as already probed-lenient so the one-time probe adds no request. The probe
    # itself is exercised by the dedicated alternation tests below.
    client = _unprobed_client(handler)
    client._needs_alternation = False
    return client


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


def test_raise_for_status_maps_429_to_rate_limit() -> None:
    from lilbee.providers.base import ProviderErrorKind
    from lilbee.providers.fleet.client import _raise_for_status

    with pytest.raises(ProviderError) as excinfo:
        _raise_for_status(httpx.Response(429, json={"error": "busy"}))
    assert excinfo.value.kind is ProviderErrorKind.RATE_LIMIT


@pytest.mark.parametrize("status", [502, 503, 504])
def test_raise_for_status_maps_gateway_errors_to_server(status: int) -> None:
    # llama-swap answers for a momentarily-unreachable upstream (restarting,
    # OOM-killed, mid-swap) with a gateway error; the kind must be retryable,
    # not terminal.
    from lilbee.providers.base import ProviderErrorKind
    from lilbee.providers.fleet.client import _raise_for_status

    with pytest.raises(ProviderError) as excinfo:
        _raise_for_status(httpx.Response(status, text="Bad Gateway"))
    assert excinfo.value.kind is ProviderErrorKind.SERVER


def test_raise_for_status_gateway_premature_exit_stays_connection(monkeypatch) -> None:
    # A 502 whose body carries the died marker keeps its CONNECTION kind (and
    # with it the replica failover path); the gateway status must not shadow it.
    import lilbee.providers.fleet.client as client_mod
    from lilbee.providers.base import ProviderErrorKind
    from lilbee.providers.fleet.client import _raise_for_status

    monkeypatch.setattr(client_mod, "_upstream_failure_tail", lambda _resp: "")
    with pytest.raises(ProviderError) as excinfo:
        _raise_for_status(
            httpx.Response(502, text='{"error": "upstream command exited prematurely"}')
        )
    assert excinfo.value.kind is ProviderErrorKind.CONNECTION


def test_embed_retries_transient_gateway_error_then_succeeds(monkeypatch) -> None:
    # A 502 while the upstream restarts is retried like a busy 429 instead of
    # dropping the input.
    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

    assert _client(handler).embed(["hello"]) == [[0.1, 0.2]]
    assert calls["n"] == 3


def test_retry_on_busy_deadline_retries_gateway_error(monkeypatch) -> None:
    # The deadline-bound OCR retry rides out a transient 502 the same way it
    # rides out a busy 429: the page is re-attempted, not dropped.
    from lilbee.providers.base import ProviderError, ProviderErrorKind
    from lilbee.providers.fleet.client import retry_on_busy

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    monkeypatch.setattr("lilbee.providers.fleet.client.time.monotonic", lambda: 0.0)
    gateway = ProviderError("HTTP 502", provider="llama-server", kind=ProviderErrorKind.SERVER)
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise gateway
        return "ok"

    assert retry_on_busy(_call, deadline=100.0) == "ok"
    assert calls["n"] == 3


def test_embed_retries_on_busy_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "Too many requests"})
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

    assert _client(handler).embed(["hello"]) == [[0.1, 0.2]]
    assert calls["n"] == 3  # two 429s retried, third succeeds


def test_embed_gives_up_after_retries_when_persistently_busy(monkeypatch) -> None:
    from lilbee.providers.base import ProviderErrorKind
    from lilbee.providers.fleet.client import _EMBED_BUSY_RETRIES

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": "busy"})

    with pytest.raises(ProviderError) as excinfo:
        _client(handler).embed(["x"])
    assert excinfo.value.kind is ProviderErrorKind.RATE_LIMIT
    assert calls["n"] == _EMBED_BUSY_RETRIES


def test_embed_rides_out_cold_start_past_interactive_budget(monkeypatch) -> None:
    # A cold embedder can 429 more times than the short interactive budget allows;
    # bulk ingest must wait it out (via _EMBED_BUSY_RETRIES) instead of dropping the
    # file. With the old shared budget this warmup would have raised.
    from lilbee.providers.fleet.client import _BUSY_RETRIES, _EMBED_BUSY_RETRIES

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    warmup = _BUSY_RETRIES + 2  # more 429s than the interactive budget tolerates
    assert warmup < _EMBED_BUSY_RETRIES
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= warmup:
            return httpx.Response(429, json={"error": "warming"})
        return httpx.Response(200, json={"data": [{"embedding": [0.3]}]})

    assert _client(handler).embed(["x"]) == [[0.3]]
    assert calls["n"] == warmup + 1


def test_retry_on_busy_deadline_retries_past_attempt_cap(monkeypatch) -> None:
    # Under deep-queue contention the queue drains far slower than the fixed
    # attempt budget, so a deadline-bound retry must keep waiting past
    # _BUSY_RETRIES until the caller's own deadline, not drop the page.
    from lilbee.providers.base import ProviderError, ProviderErrorKind
    from lilbee.providers.fleet.client import _BUSY_RETRIES, retry_on_busy

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    monkeypatch.setattr("lilbee.providers.fleet.client.time.monotonic", lambda: 0.0)
    busy = ProviderError("busy", provider="llama-server", kind=ProviderErrorKind.RATE_LIMIT)
    attempts = _BUSY_RETRIES + 5  # more 429s than the attempt cap would tolerate
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        if calls["n"] <= attempts:
            raise busy
        return "ok"

    assert retry_on_busy(_call, deadline=100.0) == "ok"
    assert calls["n"] == attempts + 1


def test_retry_on_busy_deadline_gives_up_when_deadline_passes(monkeypatch) -> None:
    from lilbee.providers.base import ProviderError, ProviderErrorKind
    from lilbee.providers.fleet.client import retry_on_busy

    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    clock = {"t": 0.0}
    monkeypatch.setattr("lilbee.providers.fleet.client.time.monotonic", lambda: clock["t"])
    busy = ProviderError("busy", provider="llama-server", kind=ProviderErrorKind.RATE_LIMIT)
    calls = {"n": 0}

    def _call() -> str:
        calls["n"] += 1
        clock["t"] += 1.0  # each attempt advances the clock toward the deadline
        raise busy

    with pytest.raises(ProviderError) as excinfo:
        retry_on_busy(_call, deadline=3.0)
    assert excinfo.value.kind is ProviderErrorKind.RATE_LIMIT
    assert calls["n"] < 10  # bounded by the deadline, not looping forever


def test_busy_backoff_is_capped(monkeypatch) -> None:
    # Backoff must not balloon past the cap even across the long ingest budget.
    from lilbee.providers.fleet.client import _BUSY_BACKOFF_MAX_S, _EMBED_BUSY_RETRIES

    delays: list[float] = []
    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", delays.append)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "busy"})

    with pytest.raises(ProviderError):
        _client(handler).embed(["x"])
    assert len(delays) == _EMBED_BUSY_RETRIES - 1
    assert max(delays) <= _BUSY_BACKOFF_MAX_S


def test_embed_does_not_retry_non_busy_errors(monkeypatch) -> None:
    # A non-RATE_LIMIT failure (e.g. a malformed 500) must surface immediately,
    # not burn the retry budget.
    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, text="boom")

    with pytest.raises(ProviderError):
        _client(handler).embed(["x"])
    assert calls["n"] == 1


def test_llm_rerank_retries_on_busy(monkeypatch) -> None:
    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, json={"error": "busy"})
        return _chat_logprobs_response(0.0, -3.0)

    scores = _llm_rerank_client(handler).rerank("q", ["doc"])
    assert calls["n"] == 2
    assert scores[0] > 0.5


def test_llm_rerank_score_softmax_yes_over_no() -> None:
    top = [{"token": "yes", "logprob": 0.0}, {"token": "no", "logprob": -2.0}]
    assert 0.8 < _llm_rerank_score(top) < 1.0


def test_llm_rerank_score_softmax_no_over_yes() -> None:
    top = [{"token": "No", "logprob": 0.0}, {"token": " yes", "logprob": -3.0}]
    assert _llm_rerank_score(top) < 0.2


def test_llm_rerank_score_only_yes_present() -> None:
    assert _llm_rerank_score([{"token": "yes", "logprob": -0.5}]) > 0.0


def test_llm_rerank_score_only_no_present_is_zero() -> None:
    assert _llm_rerank_score([{"token": "no", "logprob": -0.5}]) == 0.0


def test_llm_rerank_score_neither_present_is_none() -> None:
    assert _llm_rerank_score([{"token": "maybe", "logprob": -0.1}]) is None
    assert _llm_rerank_score([]) is None


def test_first_token_top_logprobs_empty_on_missing_fields() -> None:
    assert _first_token_top_logprobs({}) == []
    assert _first_token_top_logprobs({"choices": [{}]}) == []
    assert _first_token_top_logprobs({"choices": [{"logprobs": {"content": []}}]}) == []


def _llm_rerank_client(handler) -> LlamaServerClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    return LlamaServerClient("http://gpu0", "rerank-0", http=http, rerank_mode=RerankMode.LLM)


def _think_token_response() -> httpx.Response:
    """What a thinking-capable chat template yields: the verdict is not token 0."""
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "logprobs": {
                        "content": [{"top_logprobs": [{"token": "<think>", "logprob": -0.01}]}]
                    }
                }
            ]
        },
    )


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


def test_llm_rerank_request_disables_template_thinking() -> None:
    captured: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content)["chat_template_kwargs"])
        return _chat_logprobs_response(0.0, -1.0)

    _llm_rerank_client(handler).rerank("q", ["doc"])
    assert captured == [{"enable_thinking": False}]


def test_llm_rerank_raises_when_no_candidate_yields_a_verdict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _think_token_response()

    with pytest.raises(ProviderError, match=r"yes.*no"):
        _llm_rerank_client(handler).rerank("q", ["one", "two"])


def test_llm_rerank_scores_a_verdictless_candidate_zero_without_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "answered" in body["messages"][0]["content"]:
            return _chat_logprobs_response(0.0, -4.0)
        return _think_token_response()

    scores = _llm_rerank_client(handler).rerank("q", ["answered", "silent"])
    assert scores[0] > 0.5
    assert scores[1] == 0.0


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


class TestHealthHalfOpen:
    def _clocked_client(self, monkeypatch: pytest.MonkeyPatch) -> tuple[LlamaServerClient, dict]:
        from lilbee.providers.fleet import client as client_mod

        clock = {"now": 0.0}
        monkeypatch.setattr(client_mod.time, "monotonic", lambda: clock["now"])
        return _client(), clock

    def test_unhealthy_within_cooldown_is_not_routable(self, monkeypatch) -> None:
        from lilbee.providers.fleet.client import _UNHEALTHY_RETRY_S

        c, clock = self._clocked_client(monkeypatch)
        c.mark_unhealthy()
        clock["now"] = _UNHEALTHY_RETRY_S - 0.1
        assert c.healthy is False

    def test_unhealthy_becomes_routable_after_cooldown(self, monkeypatch) -> None:
        from lilbee.providers.fleet.client import _UNHEALTHY_RETRY_S

        c, clock = self._clocked_client(monkeypatch)
        c.mark_unhealthy()
        clock["now"] = _UNHEALTHY_RETRY_S
        assert c.healthy is True  # half-open: the next routed request is the probe

    def test_refailure_restamps_the_cooldown(self, monkeypatch) -> None:
        from lilbee.providers.fleet.client import _UNHEALTHY_RETRY_S

        c, clock = self._clocked_client(monkeypatch)
        c.mark_unhealthy()
        clock["now"] = _UNHEALTHY_RETRY_S
        c.mark_unhealthy()  # the probe-by-traffic failed again
        clock["now"] = 2 * _UNHEALTHY_RETRY_S - 0.1
        assert c.healthy is False
        clock["now"] = 2 * _UNHEALTHY_RETRY_S
        assert c.healthy is True

    def test_mark_healthy_restores_immediately(self, monkeypatch) -> None:
        c, _clock = self._clocked_client(monkeypatch)
        c.mark_unhealthy()
        c.mark_healthy()
        assert c.healthy is True


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


def test_chat_coerces_null_content_to_empty_string() -> None:
    """content is null for a refusal / content-filter stop / empty completion;
    chat() must return "" (like chat_result/chat_tools), never the string "None"."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    assert _client(handler).chat([{"role": "user", "content": "hi"}]) == ""


def test_chat_stream_yields_deltas() -> None:
    chunks = list(_client().chat([{"role": "user", "content": "hi"}], stream=True))
    assert chunks == ["He", "llo"]


def test_chat_stream_forwards_caller_timeout() -> None:
    """A caller deadline must reach the streaming request, not be silently dropped."""
    client = _client()
    seen: dict[str, object] = {}
    real_stream = client._http.stream

    def _spy_stream(method, url, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return real_stream(method, url, **kwargs)

    client._http.stream = _spy_stream  # type: ignore[method-assign]
    list(client.chat([{"role": "user", "content": "hi"}], stream=True, timeout=7.5))
    assert seen["timeout"] == 7.5


def test_chat_bounded_accumulates_streamed_content() -> None:
    client = _client()
    assert client.chat_bounded([{"role": "user", "content": "hi"}], deadline_s=5.0) == "Hello"
    assert client.in_flight == 0  # slot released on the normal exit


def test_chat_bounded_raises_and_releases_slot_on_deadline() -> None:
    """A blown deadline surfaces ChatDeadlineError and frees the in-flight slot.

    A per-phase httpx timeout leaks the socket read (and its in-flight increment)
    past the deadline; the streamed total-deadline path must release it promptly.
    """
    client = _client()
    with pytest.raises(ChatDeadlineError):
        # deadline_s=0 is already spent by the first streamed frame, so the loop's
        # monotonic check trips and the with-block closes the stream at once.
        client.chat_bounded([{"role": "user", "content": "hi"}], deadline_s=0.0)
    assert client.in_flight == 0


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


def _premature_exit_response() -> httpx.Response:
    request = httpx.Request(
        "POST", "http://127.0.0.1:9100/v1/embeddings", json={"model": "embed-0"}
    )
    return httpx.Response(
        500,
        text='{"src":"llama-swap", "error": "unspecific error: upstream command'
        ' exited prematurely"}',
        request=request,
    )


def test_raise_for_status_surfaces_upstream_tail_on_premature_exit(monkeypatch, caplog) -> None:
    # llama-swap masks the dead server's stderr behind "exited prematurely"; the
    # client fetches the upstream's captured output, logs it, AND surfaces it in
    # the raised error so the real exit reason reaches the caller.
    import lilbee.providers.fleet.client as client_mod
    from lilbee.providers.fleet.client import _raise_for_status

    seen: dict[str, str] = {}

    class _FakeStream:
        def __enter__(self) -> _FakeStream:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def iter_text(self):
            # Padded past the tail cap so the read stops at the size limit
            # instead of waiting out the stream timeout.
            yield "x" * 2500 + "E srv start: couldn't bind HTTP server socket, port: 5801\n"
            raise AssertionError("must stop reading once the tail cap is reached")

    def _fake_stream(method: str, url: str, timeout: float) -> _FakeStream:
        seen["url"] = url
        return _FakeStream()

    monkeypatch.setattr(client_mod.httpx, "stream", _fake_stream)
    with caplog.at_level("WARNING"), pytest.raises(ProviderError) as excinfo:
        _raise_for_status(_premature_exit_response())
    assert seen["url"] == "http://127.0.0.1:9100/logs/stream/embed-0"
    # The captured server output is both logged and surfaced in the error message.
    assert "couldn't bind HTTP server socket" in caplog.text
    assert "couldn't bind HTTP server socket" in str(excinfo.value)
    assert "exited prematurely" in str(excinfo.value)


def test_raise_for_status_premature_exit_survives_log_fetch_failure(monkeypatch) -> None:
    # A dead log stream must not mask the original ProviderError.
    import lilbee.providers.fleet.client as client_mod
    from lilbee.providers.fleet.client import _raise_for_status

    def _refuse(method: str, url: str, timeout: float) -> object:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(client_mod.httpx, "stream", _refuse)
    with pytest.raises(ProviderError) as excinfo:
        _raise_for_status(_premature_exit_response())
    # An unreadable tail appends nothing; the original error still propagates.
    assert "exited prematurely" in str(excinfo.value)
    assert "upstream server output:" not in str(excinfo.value)


def test_chat_stream_surfaces_server_error_body() -> None:
    # On a streaming request the response body isn't read yet, so the error path
    # must read it before extracting the message (the ResponseNotRead branch).
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model is still loading")

    with pytest.raises(ProviderError, match="model is still loading"):
        list(_client(handler).chat([{"role": "user", "content": "hi"}], stream=True))


_ALTERNATION_BODY = (
    "Jinja Exception: After the optional system message, conversation roles "
    "must alternate user/assistant/user/assistant/..."
)
_TOOL_LOOP_MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "find foo"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}],
    },
    {"role": "tool", "content": "a.py:1", "tool_call_id": "c1"},
    {"role": "assistant", "content": "It is in a.py."},
]


def _strict_alternation_handler(request: httpx.Request) -> httpx.Response:
    """Model a strict-alternation template: reject the tool role or two same-role
    turns in a row after the system block, render anything else."""
    if request.url.path == "/health":
        return httpx.Response(200)
    body = json.loads(request.content)
    convo = [m["role"] for m in body["messages"] if m["role"] != "system"]
    if "tool" in convo or any(earlier == later for earlier, later in pairwise(convo)):
        return httpx.Response(500, text=_ALTERNATION_BODY)
    if body.get("stream"):
        return httpx.Response(
            200, text='data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
        )
    return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})


def _is_probe(request: httpx.Request) -> bool:
    """Whether *request* is the alternation probe (its tool is named ``probe``)."""
    tools = json.loads(request.content).get("tools") or []
    return bool(tools) and tools[0]["function"]["name"] == "probe"


def test_chat_result_normalizes_when_template_requires_alternation() -> None:
    # The probe finds the template rejects the raw tool exchange but renders the
    # normalized one, so the real (non-probe) request is reshaped before sending.
    sent: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions" and not _is_probe(request):
            sent.append(json.loads(request.content)["messages"])
        return _strict_alternation_handler(request)

    client = _unprobed_client(handler)
    result = client.chat_result(_TOOL_LOOP_MESSAGES)

    assert result.text == "ok"
    assert client._needs_alternation is True
    convo_roles = [m["role"] for m in sent[-1] if m["role"] != "system"]
    assert "tool" not in convo_roles
    for earlier, later in pairwise(convo_roles):
        assert earlier != later


def test_chat_result_keeps_raw_messages_when_template_accepts_tool_exchange() -> None:
    # A lenient template renders the raw exchange, so the probe leaves messages
    # untouched and the tool role survives into the real request.
    sent: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions" and not _is_probe(request):
            sent.append(json.loads(request.content)["messages"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = _unprobed_client(handler)
    result = client.chat_result(_TOOL_LOOP_MESSAGES)

    assert result.text == "ok"
    assert client._needs_alternation is False
    assert any(m["role"] == "tool" for m in sent[-1])


def test_chat_stream_items_normalizes_when_template_requires_alternation() -> None:
    # The stream path reshapes up front too, so the open never hits the rejection.
    sent: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions" and not _is_probe(request):
            sent.append(json.loads(request.content)["messages"])
        return _strict_alternation_handler(request)

    frames = list(_unprobed_client(handler).chat_stream_items(_TOOL_LOOP_MESSAGES))

    assert frames == ["hi"]
    convo_roles = [m["role"] for m in sent[-1] if m["role"] != "system"]
    assert "tool" not in convo_roles


def test_alternation_probe_runs_once_and_is_cached() -> None:
    # The probe (raw + normalized) fires once; a second chat reuses the verdict.
    probes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions" and _is_probe(request):
            probes.append(1)
        return _strict_alternation_handler(request)

    client = _unprobed_client(handler)
    client.chat_result(_TOOL_LOOP_MESSAGES)
    client.chat_result(_TOOL_LOOP_MESSAGES)

    assert client._needs_alternation is True
    assert len(probes) == 2  # one raw + one normalized, from the single probe round


def test_alternation_probe_skips_when_another_thread_resolved_it() -> None:
    # Double-checked locking: if another caller resolves the verdict between the
    # outer check and the lock, the inner re-check returns without re-probing.
    client = _unprobed_client()

    class _ResolvingLock:
        def __enter__(self) -> object:
            client._needs_alternation = False
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    client._alternation_lock = _ResolvingLock()  # type: ignore[assignment]
    client._ensure_alternation_probed()
    assert client._needs_alternation is False


def test_alternation_probe_inconclusive_when_server_unreachable() -> None:
    # A connection failure during the probe must not cache a verdict; once the
    # server recovers, the next probe re-runs and resolves it.
    state = {"up": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["up"]:
            raise httpx.ConnectError("replica down")
        return _strict_alternation_handler(request)

    client = _unprobed_client(handler)
    client._ensure_alternation_probed()
    assert client._needs_alternation is None

    state["up"] = True
    client._ensure_alternation_probed()
    assert client._needs_alternation is True


def test_alternation_probe_inconclusive_when_server_busy() -> None:
    # A 429 (cold replica, slots warming) during the probe is transient, not a
    # template verdict: it must not be cached, and the next probe resolves it.
    state = {"busy": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if state["busy"]:
            return httpx.Response(429)
        return _strict_alternation_handler(request)

    client = _unprobed_client(handler)
    client._ensure_alternation_probed()
    assert client._needs_alternation is None

    state["busy"] = False
    client._ensure_alternation_probed()
    assert client._needs_alternation is True


def test_alternation_probe_inconclusive_when_reshape_probe_is_busy() -> None:
    # The raw exchange is rejected but the reshaped probe hits a transient 429;
    # the verdict stays undetermined rather than caching a wrong False.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        convo = [m["role"] for m in json.loads(request.content)["messages"]]
        if "tool" in convo:
            return httpx.Response(500, text=_ALTERNATION_BODY)
        return httpx.Response(429)

    client = _unprobed_client(handler)
    client._ensure_alternation_probed()
    assert client._needs_alternation is None


def test_alternation_probe_skips_normalization_when_reshape_also_rejected() -> None:
    # If even the normalized exchange is rejected (an unrelated template fault),
    # the probe leaves the model un-normalized rather than guessing.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(500, text="decode failure")

    client = _unprobed_client(handler)
    client._ensure_alternation_probed()
    assert client._needs_alternation is False


def test_chat_result_propagates_server_error_after_probe() -> None:
    # With no normalization flagged, the real request's error surfaces unchanged.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(500, text="decode failure")

    with pytest.raises(ProviderError, match="decode failure"):
        _unprobed_client(handler).chat_result(_TOOL_LOOP_MESSAGES)


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


def test_loopback_ssl_context_is_shared_and_unverified() -> None:
    """Loopback clients reuse one minimal SSL context instead of rebuilding the
    default (CA-loading) one on every reload."""
    import ssl

    from lilbee.providers.fleet.client import _LOOPBACK_SSL_CONTEXT

    assert _LOOPBACK_SSL_CONTEXT.verify_mode == ssl.CERT_NONE
    assert _LOOPBACK_SSL_CONTEXT.check_hostname is False


def test_owned_client_uses_shared_context_not_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Creating a client must reuse the shared context and never call the expensive
    create_default_context on the reload path."""
    import lilbee.providers.fleet.client as client_mod

    captured: dict[str, object] = {}
    real_client = httpx.Client

    def _spy_client(**kwargs: object) -> httpx.Client:
        captured.update(kwargs)
        return real_client(transport=httpx.MockTransport(_handler))

    monkeypatch.setattr(client_mod.httpx, "Client", _spy_client)
    monkeypatch.setattr(client_mod.ssl, "create_default_context", _boom_create_default_context)
    LlamaServerClient("http://gpu0", "m")
    assert captured["verify"] is client_mod._LOOPBACK_SSL_CONTEXT


def _boom_create_default_context(*_a: object, **_k: object) -> object:
    raise AssertionError("create_default_context must not be called on the reload path")


def test_close_leaves_injected_client_open() -> None:
    http = httpx.Client(transport=httpx.MockTransport(_handler))
    c = LlamaServerClient("http://gpu0", "m", http=http)
    c.close()
    assert not http.is_closed


class TestParseSseDeltas:
    def test_non_data_line_is_empty(self) -> None:
        assert _parse_sse_deltas("event: message") == ("", "")

    def test_done_sentinel_is_empty(self) -> None:
        assert _parse_sse_deltas("data: [DONE]") == ("", "")

    def test_invalid_json_is_empty(self) -> None:
        assert _parse_sse_deltas("data: {not json") == ("", "")

    def test_no_choices_is_empty(self) -> None:
        assert _parse_sse_deltas('data: {"choices": []}') == ("", "")

    def test_extracts_content(self) -> None:
        line = 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        assert _parse_sse_deltas(line) == ("", "hi")

    def test_extracts_reasoning(self) -> None:
        line = 'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}'
        assert _parse_sse_deltas(line) == ("hmm", "")


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


def _content_sse(content: str) -> str:
    """One OpenAI SSE chunk carrying *content* as a text delta."""
    return f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"


def _stream_handler(stream_body: str):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, text=stream_body)
        return httpx.Response(404)

    return handler


def test_chat_stream_items_recovers_bare_json_tool_call() -> None:
    """A model that streams a bare-JSON call as text yields recovered deltas, no
    raw JSON leaking to the client."""
    from lilbee.providers.base import ToolCallDelta

    # Split the bare-JSON call across chunks so the buffer must reassemble the
    # full call before parsing.
    # A leading whitespace-only token (common from small models) must not resolve
    # the buffer as plain text: it stays buffered until the '{' arrives.
    head = '{"name": "lilbee_search", '
    tail = '"arguments": {"query": "cats"}}'
    body = _content_sse("  ") + _content_sse(head) + _content_sse(tail) + "data: [DONE]\n\n"
    items = list(
        _client(_stream_handler(body)).chat_stream_items(
            [{"role": "user", "content": "find cats"}], tools=_tools()
        )
    )
    assert not any(isinstance(i, str) for i in items), "raw JSON text must not leak"
    deltas = [i for i in items if isinstance(i, ToolCallDelta)]
    assert len(deltas) == 1
    assert deltas[0].index == 0
    assert deltas[0].name == "lilbee_search"
    assert json.loads(deltas[0].arguments_delta or "") == {"query": "cats"}


def test_recover_bare_json_stream_forwards_close_to_source() -> None:
    # Closing the wrapper must close the source generator, or the
    # underlying HTTP stream and its in_flight slot leak until GC.
    from lilbee.providers.fleet.client import _recover_bare_json_stream

    source_closed = {"value": False}

    def source():
        try:
            yield "hello "
            yield "world"
        finally:
            source_closed["value"] = True

    wrapped = _recover_bare_json_stream(source())
    assert next(wrapped) == "hello "  # plain text streams straight through
    wrapped.close()
    assert source_closed["value"]


def test_chat_stream_items_streams_normal_text_incrementally() -> None:
    """Plain text is not buffered to the end; tokens arrive one frame at a time."""
    body = _content_sse("He") + _content_sse("llo") + _content_sse(" there") + "data: [DONE]\n\n"
    items = list(
        _client(_stream_handler(body)).chat_stream_items([{"role": "user", "content": "hi"}])
    )
    # Each content token is its own frame (incremental), not one coalesced blob.
    assert items == ["He", "llo", " there"]


def test_chat_stream_items_midstream_brace_does_not_rebuffer() -> None:
    """A '{' or '[' token that appears AFTER plain text already streamed must keep
    streaming token-by-token. Bare-call recovery only applies to the leading text;
    once committed to plain text the wrapper must never buffer again, or an answer
    containing a JSON/code example freezes mid-stream and dumps at the end."""
    body = (
        _content_sse("Here is ")
        + _content_sse("a config: ")
        + _content_sse("{\n")
        + _content_sse('  "k": 1\n}')
        + _content_sse(" -- done")
        + "data: [DONE]\n\n"
    )
    items = list(
        _client(_stream_handler(body)).chat_stream_items(
            [{"role": "user", "content": "show config"}], tools=_tools()
        )
    )
    # Every token is its own frame; nothing coalesces after the brace token.
    assert items == ["Here is ", "a config: ", "{\n", '  "k": 1\n}', " -- done"]


def test_chat_stream_items_leading_whitespace_then_text_keeps_order() -> None:
    """A whitespace-only first token is buffered (its first non-ws char is unknown),
    then flushed in order once plain text resolves it -- not reordered to the end."""
    body = _content_sse("  ") + _content_sse("Hello") + _content_sse(" world") + "data: [DONE]\n\n"
    items = list(
        _client(_stream_handler(body)).chat_stream_items([{"role": "user", "content": "hi"}])
    )
    # The leading whitespace stays first; subsequent tokens stream through in order.
    assert items == ["  ", "Hello", " world"]
    assert "".join(i for i in items if isinstance(i, str)) == "  Hello world"


def test_chat_stream_items_text_starting_with_brace_is_not_recovered() -> None:
    """Text that opens with '{' but is not a tool call streams through as text."""
    body = _content_sse("{not really") + _content_sse(" a call}") + "data: [DONE]\n\n"
    items = list(
        _client(_stream_handler(body)).chat_stream_items(
            [{"role": "user", "content": "x"}], tools=_tools()
        )
    )
    from lilbee.providers.base import ToolCallDelta

    assert not any(isinstance(i, ToolCallDelta) for i in items)
    assert "".join(i for i in items if isinstance(i, str)) == "{not really a call}"


def test_chat_stream_items_native_tool_calls_pass_through_untouched() -> None:
    """Native tool_calls deltas are forwarded as is with no double-recovery. The
    leading text opens with '{' so it is buffered as a candidate, then flushed
    verbatim when the native delta proves it was just text."""
    from lilbee.providers.base import ToolCallDelta

    native_delta = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "function": {"name": "get_weather", "arguments": '{"city":"SF"}'},
                        }
                    ]
                }
            }
        ]
    }
    body = (
        _content_sse("{thinking} ") + f"data: {json.dumps(native_delta)}\n\n" + "data: [DONE]\n\n"
    )
    items = list(
        _client(_stream_handler(body)).chat_stream_items(
            [{"role": "user", "content": "weather?"}], tools=_tools()
        )
    )
    # The buffered '{...}' lead is flushed as text; exactly one native delta, not recovered.
    assert items[0] == "{thinking} "
    deltas = [i for i in items if isinstance(i, ToolCallDelta)]
    assert len(deltas) == 1
    assert deltas[0].id == "c1"
    assert deltas[0].name == "get_weather"


def test_chat_stream_items_recovery_still_emits_usage_terminator_last() -> None:
    """The usage terminator is emitted after a recovered bare-JSON call."""
    from lilbee.providers.base import TokenUsage, ToolCallDelta

    call_text = '{"name": "lilbee_search", "arguments": {"query": "x"}}'
    usage_chunk = '{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}'
    body = _content_sse(call_text) + f"data: {usage_chunk}\n\n" + "data: [DONE]\n\n"
    items = list(
        _client(_stream_handler(body)).chat_stream_items(
            [{"role": "user", "content": "x"}], tools=_tools()
        )
    )
    assert isinstance(items[0], ToolCallDelta)
    assert items[0].name == "lilbee_search"
    assert items[-1] == TokenUsage(prompt_tokens=5, completion_tokens=2)


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

    # A missing finish_reason (None), a non-string, or an unknown string all
    # fall back to STOP; a known value maps to its member.
    assert _coerce_finish_reason(None) is FinishReason.STOP
    assert _coerce_finish_reason(42) is FinishReason.STOP
    assert _coerce_finish_reason("not_a_reason") is FinishReason.STOP
    assert _coerce_finish_reason("length") is FinishReason.LENGTH


def test_parse_sse_stream_items_skips_malformed_json() -> None:
    from lilbee.providers.fleet.client import _parse_sse_stream_items

    assert list(_parse_sse_stream_items("data: {not json", _ThinkInliner(enabled=True))) == []


def test_parse_sse_stream_items_skips_empty_choices() -> None:
    from lilbee.providers.fleet.client import _parse_sse_stream_items

    assert list(_parse_sse_stream_items('data: {"choices": []}', _ThinkInliner(enabled=True))) == []


def test_parse_sse_stream_items_emits_finish_frame_on_length() -> None:
    from lilbee.providers.base import FinishReason, StreamFinish
    from lilbee.providers.fleet.client import _parse_sse_stream_items

    line = 'data: {"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]}'
    items = list(_parse_sse_stream_items(line, _ThinkInliner(enabled=True)))
    assert items == ["x", StreamFinish(reason=FinishReason.LENGTH)]


def test_parse_sse_stream_items_no_finish_frame_without_reason() -> None:
    from lilbee.providers.base import StreamFinish
    from lilbee.providers.fleet.client import _parse_sse_stream_items

    line = 'data: {"choices": [{"delta": {"content": "x"}, "finish_reason": null}]}'
    items = list(_parse_sse_stream_items(line, _ThinkInliner(enabled=True)))
    assert not any(isinstance(i, StreamFinish) for i in items)


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


def test_embed_retries_exact_on_batch_overflow_500() -> None:
    """An input past the server's n_batch is a 500 saying "too large to process"
    (not the 400 context shape); it must take the same exact-tokenize retry."""
    calls = {"embed": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/tokenize"):
            return httpx.Response(200, json={"tokens": list(range(20))})
        if path.endswith("/detokenize"):
            return httpx.Response(200, json={"content": "t"})
        if path == "/v1/embeddings":
            calls["embed"] += 1
            if calls["embed"] == 1:
                return httpx.Response(
                    500,
                    text='{"error":{"code":500,"message":"input (136 tokens) is too large'
                    ' to process. increase the physical batch size","type":"server_error"}}',
                )
            return httpx.Response(200, json={"data": [{"embedding": [0.5]}]})
        return httpx.Response(404)

    assert _capped_client(handler, 10).embed(["dense"]) == [[0.5]]
    assert calls["embed"] == 2


def test_embed_does_not_retry_on_non_overflow_error() -> None:
    """A non-overflow embed error propagates without a tokenize-retry."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/embeddings":
            return httpx.Response(500, text="boom")
        return httpx.Response(404)

    with pytest.raises(ProviderError, match="500"):
        _capped_client(handler, 10).embed(["x"])


def test_raise_for_status_tags_premature_exit_as_connection(monkeypatch) -> None:
    import lilbee.providers.fleet.client as client_mod
    from lilbee.providers.base import ProviderErrorKind
    from lilbee.providers.fleet.client import _raise_for_status

    monkeypatch.setattr(client_mod, "_upstream_failure_tail", lambda _resp: "")
    with pytest.raises(ProviderError) as excinfo:
        _raise_for_status(_premature_exit_response())
    assert excinfo.value.kind is ProviderErrorKind.CONNECTION


class TestIsConnectionFailure:
    def test_true_for_httpx_transport_errors(self) -> None:
        from lilbee.providers.fleet.client import is_connection_failure

        assert is_connection_failure(httpx.ConnectError("refused")) is True

    def test_true_for_connection_kind_provider_error(self) -> None:
        from lilbee.providers.base import ProviderErrorKind
        from lilbee.providers.fleet.client import is_connection_failure

        exc = ProviderError("dead", provider="llama-server", kind=ProviderErrorKind.CONNECTION)
        assert is_connection_failure(exc) is True

    def test_false_for_model_level_errors(self) -> None:
        from lilbee.providers.fleet.client import is_connection_failure

        assert is_connection_failure(ProviderError("bad", provider="llama-server")) is False
        assert is_connection_failure(ValueError("nope")) is False


class TestClientHealthFlag:
    def test_starts_healthy_and_flips_with_marks(self) -> None:
        client = _client()
        assert client.healthy is True
        client.mark_unhealthy()
        assert client.healthy is False
        client.mark_healthy()
        assert client.healthy is True


def test_chat_forwards_per_request_timeout() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _client(handler).chat([{"role": "user", "content": "hi"}], timeout=7.5)
    assert seen["timeout"] == {"connect": 7.5, "read": 7.5, "write": 7.5, "pool": 7.5}


class TestThinkInliner:
    def test_wraps_reasoning_then_content(self) -> None:
        inliner = _ThinkInliner(enabled=True)
        out = (
            inliner.feed("step one", "")
            + inliner.feed(" step two", "")
            + inliner.feed("", "answer")
        )
        out += inliner.finish()
        assert out == "<think>step one step two</think>answer"

    def test_mixed_delta_carries_both(self) -> None:
        inliner = _ThinkInliner(enabled=True)
        assert inliner.feed("thought", "answer") == "<think>thought</think>answer"

    def test_unterminated_reasoning_closed_at_finish(self) -> None:
        inliner = _ThinkInliner(enabled=True)
        out = inliner.feed("endless thought", "") + inliner.finish()
        assert out == "<think>endless thought</think>"

    def test_disabled_drops_reasoning_and_passes_content(self) -> None:
        inliner = _ThinkInliner(enabled=False)
        assert inliner.feed("secret reasoning", "ocr text") == "ocr text"
        assert inliner.finish() == ""

    def test_no_reasoning_passthrough(self) -> None:
        inliner = _ThinkInliner(enabled=True)
        assert inliner.feed("", "plain") == "plain"
        assert inliner.finish() == ""


def _reasoning_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "The answer.",
                        "reasoning_content": "Let me think.",
                    },
                    "finish_reason": "stop",
                }
            ]
        },
    )


def test_chat_result_inlines_reasoning_for_chat_clients() -> None:
    http = httpx.Client(transport=httpx.MockTransport(_reasoning_handler), base_url="http://gpu0")
    c = LlamaServerClient("http://gpu0", "m", http=http, inline_reasoning=True)
    result = c.chat_result([{"role": "user", "content": "q"}])
    assert result.text == "<think>Let me think.</think>The answer."


def test_chat_result_drops_reasoning_when_disabled() -> None:
    http = httpx.Client(transport=httpx.MockTransport(_reasoning_handler), base_url="http://gpu0")
    c = LlamaServerClient("http://gpu0", "m", http=http)
    assert c.chat_result([{"role": "user", "content": "q"}]).text == "The answer."


def _reasoning_sse_handler(request: httpx.Request) -> httpx.Response:
    lines = (
        'data: {"choices":[{"delta":{"reasoning_content":"hmm "}}]}\n\n'
        'data: {"choices":[{"delta":{"reasoning_content":"ok"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"4"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(200, content=lines.encode())


def test_chat_stream_inlines_reasoning_deltas() -> None:
    http = httpx.Client(
        transport=httpx.MockTransport(_reasoning_sse_handler), base_url="http://gpu0"
    )
    c = LlamaServerClient("http://gpu0", "m", http=http, inline_reasoning=True)
    text = "".join(c.chat([{"role": "user", "content": "2+2?"}], stream=True))
    assert text == "<think>hmm ok</think>4"


def _unterminated_reasoning_sse_handler(request: httpx.Request) -> httpx.Response:
    # Reasoning streams but the answer never starts (e.g. token budget exhausted).
    lines = 'data: {"choices":[{"delta":{"reasoning_content":"endless"}}]}\n\ndata: [DONE]\n\n'
    return httpx.Response(200, content=lines.encode())


def test_chat_stream_closes_unterminated_reasoning() -> None:
    http = httpx.Client(
        transport=httpx.MockTransport(_unterminated_reasoning_sse_handler), base_url="http://gpu0"
    )
    c = LlamaServerClient("http://gpu0", "m", http=http, inline_reasoning=True)
    text = "".join(c.chat([{"role": "user", "content": "q"}], stream=True))
    assert text == "<think>endless</think>"


def test_chat_stream_items_closes_unterminated_reasoning() -> None:
    http = httpx.Client(
        transport=httpx.MockTransport(_unterminated_reasoning_sse_handler), base_url="http://gpu0"
    )
    c = LlamaServerClient("http://gpu0", "m", http=http, inline_reasoning=True)
    items = list(c.chat_stream_items([{"role": "user", "content": "q"}]))
    text = "".join(i for i in items if isinstance(i, str))
    assert text == "<think>endless</think>"
