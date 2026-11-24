"""Pins for the chat-stream idle bound that a mutation run found loose.

Each test fixes one value that the stream sites pass on: the request method, the reasoning
field, the error provider, the prefill callback and the finish flag.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.client import (
    ChatDeadlineError,
    LlamaServerClient,
    StreamInterruptedError,
)
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import WorkerRole

_MESSAGES = [{"role": "user", "content": "hi"}]
_PROFILE = LLAMA.request_profile(WorkerRole.CHAT)


def _build(handler, *, inline_reasoning: bool = False, on_prefill=None) -> LlamaServerClient:
    client = LlamaServerClient(
        "http://gpu0",
        "m",
        http=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0"),
        async_transport=httpx.MockTransport(handler),
        inline_reasoning=inline_reasoning,
        on_prefill=on_prefill,
        profile=_PROFILE,
    )
    client._needs_alternation = False
    return client


def _sse(*frames: str, done: bool = True) -> str:
    return "".join(f"data: {frame}\n\n" for frame in frames) + ("data: [DONE]\n\n" if done else "")


def _text(content: str, finish: str | None = None) -> str:
    choice: dict[str, object] = {"delta": {"content": content}}
    if finish:
        choice["finish_reason"] = finish
    return json.dumps({"choices": [choice]})


_REASONING_BODY = _sse(
    json.dumps({"choices": [{"delta": {_PROFILE.reasoning_field: "hmm "}}]}),
    json.dumps({"choices": [{"delta": {"content": "4"}}]}),
)


def test_the_abortable_chat_inlines_the_reasoning_of_a_chat_client() -> None:
    """The abortable site reads the profile's reasoning field and the client's inline flag."""
    client = _build(lambda request: httpx.Response(200, text=_REASONING_BODY), inline_reasoning=True)
    assert client.chat_abortable(_MESSAGES) == "<think>hmm </think>4"


def test_the_text_stream_asks_with_a_post() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, text=_sse(_text("ok")))

    assert "".join(_build(handler).chat(_MESSAGES, stream=True)) == "ok"
    assert methods == ["POST"]


def test_a_stream_without_a_prefill_reading_never_calls_the_prefill_callback() -> None:
    """With no progress frame, the callback has nothing to clear."""
    events: list[object] = []
    client = _build(
        lambda request: httpx.Response(200, text=_sse(_text("a"), _text("b"))),
        on_prefill=events.append,
    )
    assert list(client.chat_stream_items(_MESSAGES)) == ["a", "b"]
    assert events == []


def test_a_frame_after_the_finish_frame_keeps_the_stream_ended() -> None:
    """The usage frame follows the finish frame; a body with no [DONE] is not a cut answer."""
    usage = json.dumps(
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
    )
    body = _sse(_text("a", finish="stop"), usage, done=False)
    client = _build(lambda request: httpx.Response(200, text=body))
    items = list(client.chat_stream_items(_MESSAGES))
    assert items[0] == "a"
    assert len(items) == 3


class _CutBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A body that sends one frame and then breaks, read by the sync or the async client."""

    def __iter__(self):
        yield _sse(_text("a"), done=False).encode()
        raise httpx.RemoteProtocolError("closed")

    async def __aiter__(self):
        yield _sse(_text("a"), done=False).encode()
        raise httpx.RemoteProtocolError("closed")


def _cut_after_a_frame(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, stream=_CutBody())


_CUT_SITES = {
    "text": lambda client: list(client.chat(_MESSAGES, stream=True)),
    "abortable": lambda client: client.chat_abortable(_MESSAGES),
    "items": lambda client: list(client.chat_stream_items(_MESSAGES)),
}


@pytest.mark.parametrize("site", sorted(_CUT_SITES))
def test_a_cut_stream_names_its_engine(site: str) -> None:
    with pytest.raises(StreamInterruptedError) as caught:
        _CUT_SITES[site](_build(_cut_after_a_frame))
    assert caught.value.provider == _PROFILE.label


def _fake_client() -> MagicMock:
    return MagicMock(spec=LlamaServerClient)


@pytest.mark.parametrize(("timeout", "deadline_s"), [(0.5, 0.5), (1.0, 1.0)])
def test_a_vision_timeout_of_one_second_or_less_is_the_deadline(timeout: float, deadline_s: float) -> None:
    client = _fake_client()
    client.chat_abortable.return_value = "text"
    prov_mod._vision_call(client, _MESSAGES, timeout)
    assert client.chat_abortable.call_args.kwargs["deadline_s"] == deadline_s


def test_a_vision_deadline_error_names_the_engine() -> None:
    client = _fake_client()
    client.chat_abortable.side_effect = ChatDeadlineError("late", provider="x")
    with pytest.raises(ProviderError, match="timed out after 2s") as caught:
        prov_mod._vision_call(client, _MESSAGES, 2.0)
    assert caught.value.provider == prov_mod._PROVIDER_NAME


def test_a_vision_page_cut_with_no_transport_cause_keeps_its_own_error() -> None:
    client = _fake_client()
    cut = StreamInterruptedError("llama.cpp")
    client.chat_abortable.side_effect = cut
    with pytest.raises(StreamInterruptedError) as caught:
        prov_mod._vision_call(client, _MESSAGES, None)
    assert caught.value is cut


def test_a_vision_page_cut_by_a_non_transport_error_keeps_its_own_error() -> None:
    client = _fake_client()
    cut = StreamInterruptedError("llama.cpp")
    cut.__cause__ = ValueError("not a transport error")
    client.chat_abortable.side_effect = cut
    with pytest.raises(StreamInterruptedError) as caught:
        prov_mod._vision_call(client, _MESSAGES, None)
    assert caught.value is cut


def test_no_healthy_replica_error_names_the_provider() -> None:
    with pytest.raises(ProviderError) as caught:
        prov_mod._raise_no_healthy_replica(httpx.ConnectError("refused"))
    assert caught.value.provider == prov_mod._PROVIDER_NAME
    assert caught.value.kind is ProviderErrorKind.CONNECTION
