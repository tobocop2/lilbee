"""An engine that cannot bound a chat prompt at the window has the client refuse an oversize one.

OVMS rejects a prompt only above the model's own maximum length, so a prompt between the served
window and that length would grow the cache past the memory the planner charged.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.client import LlamaServerClient, _message_text
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.ovms import OVMS
from lilbee.providers.roles import WorkerRole

WINDOW = 100
TOKENIZE_PATH = "/upstream/test-model/v3/tokenize"
SMALL = "hi"
OVER_ESTIMATE = "x" * (WINDOW * 3 + 30)
"""More characters than the window holds at the client's three characters a token."""


class _Server:
    """A mock server that counts chat and tokenize requests and answers the token count."""

    def __init__(self, tokens: int, fail: Callable[[], httpx.Response] | None = None) -> None:
        self.tokens = tokens
        self.fail = fail
        self.chats: list[dict[str, Any]] = []
        self.tokenized: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == TOKENIZE_PATH:
            self.tokenized.append(body)
            if self.fail is not None:
                return self.fail()
            return httpx.Response(200, json={"tokens": [0] * self.tokens})
        if request.url.path == "/v1/chat/completions":
            self.chats.append(body)
            if body.get("stream"):
                return httpx.Response(
                    200,
                    text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
                )
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
            )
        return httpx.Response(404)


def _client(server: _Server, *, engine: Any = OVMS, ctx: int | None = WINDOW) -> LlamaServerClient:
    http = httpx.Client(transport=httpx.MockTransport(server), base_url="http://gpu0")
    client = LlamaServerClient(
        "http://gpu0",
        "test-model",
        http=http,
        async_transport=httpx.MockTransport(server),
        profile=engine.request_profile(WorkerRole.CHAT),
        chat_ctx=ctx,
    )
    client._needs_alternation = False
    return client


def _messages(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


_CALLS: dict[str, Callable[[LlamaServerClient, str], object]] = {
    "chat": lambda c, t: c.chat(_messages(t)),
    "chat-stream": lambda c, t: list(c.chat(_messages(t), stream=True)),  # type: ignore[arg-type]
    "chat-abortable": lambda c, t: c.chat_abortable(_messages(t)),
    "chat-result": lambda c, t: c.chat_result(_messages(t)),
    "chat-tools": lambda c, t: c.chat_tools(_messages(t), tools=[]),
    "chat-stream-items": lambda c, t: list(c.chat_stream_items(_messages(t))),
}


def test_the_ovms_profile_names_the_route_that_counts_a_prompt() -> None:
    profile = OVMS.request_profile(WorkerRole.CHAT)
    assert (profile.window_tokenize_path, profile.window_tokenize_model) == (
        "/v3/tokenize",
        "default_model",
    )


@pytest.mark.parametrize("call", _CALLS)
def test_a_prompt_the_server_counts_above_the_window_is_refused_before_it_is_sent(
    call: str,
) -> None:
    server = _Server(tokens=WINDOW + 1)
    with pytest.raises(ProviderError) as refused:
        _CALLS[call](_client(server), OVER_ESTIMATE)
    assert refused.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert "context window" in str(refused.value)
    assert server.chats == []
    assert server.tokenized[0]["model"] == "default_model"
    assert OVER_ESTIMATE in server.tokenized[0]["text"]


@pytest.mark.parametrize("call", _CALLS)
def test_a_prompt_the_server_counts_within_the_window_is_sent(call: str) -> None:
    server = _Server(tokens=WINDOW)
    _CALLS[call](_client(server), OVER_ESTIMATE)
    assert len(server.chats) == 1


def test_a_short_prompt_is_sent_with_no_count_request() -> None:
    server = _Server(tokens=WINDOW + 1)
    _client(server).chat(_messages(SMALL))
    assert (len(server.chats), server.tokenized) == (1, [])


def test_the_tool_definitions_count_toward_the_window() -> None:
    server = _Server(tokens=WINDOW + 1)
    tools = [{"type": "function", "function": {"name": "t", "description": "d" * WINDOW * 3}}]
    with pytest.raises(ProviderError):
        _client(server).chat_tools(_messages(SMALL), tools=tools)
    assert "d" * WINDOW in server.tokenized[0]["text"]


def test_a_client_with_no_window_sends_any_prompt() -> None:
    server = _Server(tokens=WINDOW + 1)
    _client(server, ctx=None).chat(_messages(OVER_ESTIMATE))
    assert (len(server.chats), server.tokenized) == (1, [])


def test_an_engine_that_bounds_the_prompt_itself_is_not_counted() -> None:
    server = _Server(tokens=WINDOW + 1)
    _client(server, engine=LLAMA).chat(_messages(OVER_ESTIMATE))
    assert (len(server.chats), server.tokenized) == (1, [])


@pytest.mark.parametrize(
    ("message", "text"),
    [
        ({"role": "user", "content": "plain"}, "plain"),
        (
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:x"}},
                    {"type": "text", "text": "here"},
                ],
            },
            "look\nhere",
        ),
        ({"role": "assistant", "content": None}, ""),
        (
            {"role": "assistant", "content": "a", "tool_calls": [{"id": "1"}]},
            'a\n[{"id": "1"}]',
        ),
    ],
    ids=["string", "text-parts", "no-content", "tool-calls"],
)
def test_the_text_counted_for_a_message_holds_its_text_parts_and_tool_calls(
    message: dict[str, Any], text: str
) -> None:
    assert _message_text(message) == text


@pytest.mark.parametrize(
    "prompt",
    ["7" * 250, "\u6f22" * 80, "ab12" * 60, "\u00a7" * 120],
    ids=["digits", "cjk", "hex", "symbols"],
)
def test_a_prompt_with_more_bytes_than_the_window_is_counted_even_when_its_estimate_fits(
    prompt: str,
) -> None:
    assert len(prompt) // 3 <= WINDOW < len(prompt.encode())
    server = _Server(tokens=WINDOW + 1)
    with pytest.raises(ProviderError) as refused:
        _client(server).chat(_messages(prompt))
    assert refused.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert (len(server.chats), len(server.tokenized)) == (0, 1)


def test_a_prompt_with_no_more_bytes_than_the_window_is_not_counted() -> None:
    server = _Server(tokens=WINDOW + 1)
    _client(server).chat(_messages("y" * WINDOW))
    assert (len(server.chats), server.tokenized) == (1, [])


def _raise(exc: Exception) -> Callable[[], httpx.Response]:
    def _fail() -> httpx.Response:
        raise exc

    return _fail


@pytest.mark.parametrize(
    ("fail", "kind"),
    [
        (lambda: httpx.Response(404, text="not found"), ProviderErrorKind.UNKNOWN),
        (lambda: httpx.Response(503, text="down"), ProviderErrorKind.SERVER),
        (_raise(httpx.ReadTimeout("slow")), ProviderErrorKind.CONNECTION),
        (_raise(httpx.ConnectError("refused")), ProviderErrorKind.CONNECTION),
        (lambda: httpx.Response(200, text="not json"), ProviderErrorKind.SERVER),
    ],
    ids=["404", "5xx", "timeout", "connect", "bad-body"],
)
def test_a_count_that_fails_refuses_the_prompt_with_an_error_that_names_it(
    fail: Callable[[], httpx.Response], kind: ProviderErrorKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    server = _Server(tokens=1, fail=fail)
    with pytest.raises(ProviderError) as refused:
        _client(server).chat(_messages(OVER_ESTIMATE))
    assert "could not count the tokens of the prompt" in str(refused.value)
    assert refused.value.kind is kind
    assert server.chats == []


def test_the_count_request_is_tracked_as_in_flight() -> None:
    seen: list[int] = []
    client = _client(_Server(tokens=1))

    def _watch() -> httpx.Response:
        seen.append(client.in_flight)
        return httpx.Response(200, json={"tokens": [0]})

    client._http = httpx.Client(
        transport=httpx.MockTransport(lambda request: _watch()), base_url="http://gpu0"
    )
    client._refuse_past_window(_messages(OVER_ESTIMATE), None)
    assert seen == [1]
    assert client.in_flight == 0
