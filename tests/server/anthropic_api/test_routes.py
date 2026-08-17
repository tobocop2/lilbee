"""Route tests for the Anthropic-shaped ``/v1/messages`` surface."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from litestar import Litestar
from litestar.testing import AsyncTestClient

from lilbee.app.services import set_services
from lilbee.providers.base import ChatResult, FinishReason, ToolCall
from lilbee.server import auth as _auth_mod
from lilbee.server.anthropic_api.routes import anthropic_router
from lilbee.server.chat_dispatch.concurrency import chat_gate

INSTALLED_REF = "vendor/Model-GGUF/model-Q4.gguf"


def _h() -> dict[str, str]:
    """Auth header carrying the active session token (Claude Code's shape)."""
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


def _build_app() -> Litestar:
    return Litestar(route_handlers=[anthropic_router])


def _body(**overrides) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": INSTALLED_REF,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(overrides)
    return body


def _installed_chat_model(ref: str = INSTALLED_REF) -> MagicMock:
    manifest = MagicMock()
    manifest.ref = ref
    manifest.task = "chat"
    manifest.downloaded_at = "2026-05-15T00:00:00+00:00"
    return manifest


def _services_with(provider: MagicMock, installed: list[MagicMock]) -> Any:
    from tests.conftest import make_mock_services

    provider.max_concurrent_chats.return_value = 1
    provider.served_chat_ctx.return_value = None
    services = make_mock_services(provider=provider)
    services.registry.list_installed = MagicMock(return_value=installed)
    refs = {m.ref for m in installed}
    services.known_models.refs = MagicMock(return_value=refs)
    services.known_models.resolve = MagicMock(
        side_effect=lambda model: model if model in refs else None
    )
    return services


@pytest.fixture
def services_with_chat_model(monkeypatch):
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "chat_model", INSTALLED_REF)
    provider = MagicMock()
    provider.chat.return_value = ChatResult(
        text="hello", tool_calls=(), finish_reason=FinishReason.STOP
    )
    provider.supports_tools.return_value = False
    services = _services_with(provider, [_installed_chat_model()])
    set_services(services)
    yield services
    set_services(None)


@pytest.fixture
def _auth_token():
    previous = _auth_mod.session_manager.token
    previous_init = _auth_mod.session_manager._initialized
    _auth_mod.session_manager.token = "test-token-" + "x" * 40
    _auth_mod.session_manager._initialized = True
    yield
    _auth_mod.session_manager.token = previous
    _auth_mod.session_manager._initialized = previous_init


@pytest.fixture(autouse=True)
def _clear_chat_lock():
    chat_gate.cache_clear()
    yield
    chat_gate.cache_clear()


class FakeProviderStream:
    """``ClosableIterator`` mimicking the provider streaming protocol."""

    def __init__(self, frames: list[Any]) -> None:
        self._frames = list(frames)
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if not self._frames:
            raise StopIteration
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


def _sse_events(body: bytes) -> list[tuple[str, Any]]:
    """Parse an Anthropic SSE body into (event_type, payload) pairs."""
    out: list[tuple[str, Any]] = []
    for frame in body.decode().split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        lines = frame.split("\n")
        assert lines[0].startswith("event: "), frame
        assert lines[1].startswith("data: "), frame
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


class TestAuth:
    async def test_missing_token_is_401_in_anthropic_envelope(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body())
        assert resp.status_code == 401
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "authentication_error"

    async def test_auth_wins_over_body_validation(self, services_with_chat_model, _auth_token):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json={"model": 5})
        assert resp.status_code == 401


class TestValidation:
    async def test_missing_max_tokens_is_400_envelope(self, services_with_chat_model, _auth_token):
        body = _body()
        del body["max_tokens"]
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=body, headers=_h())
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"
        assert "max_tokens" in resp.json()["error"]["message"]

    async def test_image_content_is_400(self, services_with_chat_model, _auth_token):
        body = _body(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "base64", "data": "x"}}],
                }
            ]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=body, headers=_h())
        assert resp.status_code == 400
        assert "Image content" in resp.json()["error"]["message"]

    async def test_unknown_model_is_404_not_found_error(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(model="nope/missing"), headers=_h())
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "not_found_error"

    async def test_thinking_param_is_ignored_not_rejected(
        self, services_with_chat_model, _auth_token
    ):
        body = _body(thinking={"type": "adaptive"}, output_config={"effort": "high"})
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=body, headers=_h())
        assert resp.status_code == 200


class TestNonStreaming:
    async def test_happy_path_returns_message(self, services_with_chat_model, _auth_token):
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["id"].startswith("msg_")
        assert body["model"] == INSTALLED_REF
        assert body["content"] == [{"type": "text", "text": "hello"}]
        assert body["stop_reason"] == "end_turn"
        assert body["stop_sequence"] is None
        assert set(body["usage"]) == {"input_tokens", "output_tokens"}

    async def test_tool_call_response(self, services_with_chat_model, _auth_token):
        services_with_chat_model.provider.supports_tools.return_value = True
        services_with_chat_model.provider.chat.return_value = ChatResult(
            text="",
            tool_calls=(ToolCall(id="c1", name="search", arguments='{"q": "foo"}'),),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        body = _body(
            tools=[{"name": "search", "description": "s", "input_schema": {"type": "object"}}]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=body, headers=_h())
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["stop_reason"] == "tool_use"
        tool_blocks = [b for b in payload["content"] if b["type"] == "tool_use"]
        assert tool_blocks == [
            {"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "foo"}}
        ]


class TestStreaming:
    async def test_stream_event_sequence(self, services_with_chat_model, _auth_token):
        services_with_chat_model.provider.chat.return_value = FakeProviderStream(["he", "llo"])
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(stream=True), headers=_h())
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(resp.content)
        types = [t for t, _ in events]
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        assert "content_block_start" in types
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        assert "message_delta" in types
        text = "".join(
            p["delta"]["text"]
            for t, p in events
            if t == "content_block_delta" and p["delta"]["type"] == "text_delta"
        )
        assert text == "hello"
        start = next(p for t, p in events if t == "message_start")
        assert start["message"]["role"] == "assistant"
        assert start["message"]["model"] == INSTALLED_REF

    async def test_mid_stream_error_emits_error_event(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        def _raising_stream(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "lilbee.server.anthropic_api.routes.dispatch_chat_stream", _raising_stream
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(stream=True), headers=_h())
        assert resp.status_code == 200
        events = _sse_events(resp.content)
        error_events = [p for t, p in events if t == "error"]
        assert len(error_events) == 1
        assert error_events[0]["error"]["type"] == "api_error"


class TestErrorPaths:
    async def test_busy_backend_is_429_with_retry_after(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        from lilbee.server.chat_dispatch.concurrency import ChatBusyError

        async def _busy(_limit):
            raise ChatBusyError

        monkeypatch.setattr("lilbee.server.anthropic_api.routes.acquire_chat_slot_or_busy", _busy)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        assert resp.status_code == 429
        assert resp.headers["retry-after"] == "1"
        assert resp.json()["error"]["type"] == "rate_limit_error"

    async def test_unclassified_preflight_error_is_500_api_error(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        def _boom(_req):
            raise RuntimeError("boom")

        monkeypatch.setattr("lilbee.server.anthropic_api.routes.preflight_chat_request", _boom)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        assert resp.status_code == 500
        assert resp.json()["error"]["type"] == "api_error"

    async def test_classified_dispatch_error_maps_status_and_type(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        def _overflow(_req, canonical_model):
            raise ProviderError("context overflow", kind=ProviderErrorKind.CONTEXT_OVERFLOW)

        monkeypatch.setattr("lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat", _overflow)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    async def test_unclassified_dispatch_error_is_500(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        def _boom(_req, canonical_model):
            raise RuntimeError("boom")

        monkeypatch.setattr("lilbee.server.chat_dispatch.reasoning_cap.dispatch_chat", _boom)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        assert resp.status_code == 500
        assert resp.json()["error"]["type"] == "api_error"

    async def test_classified_mid_stream_error_maps_error_type(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        from lilbee.providers.base import ProviderError, ProviderErrorKind

        def _rate_limited(*_args, **_kwargs):
            raise ProviderError("slow down", kind=ProviderErrorKind.RATE_LIMIT)

        monkeypatch.setattr(
            "lilbee.server.anthropic_api.routes.dispatch_chat_stream", _rate_limited
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(stream=True), headers=_h())
        assert resp.status_code == 200
        events = _sse_events(resp.content)
        error_events = [p for t, p in events if t == "error"]
        assert len(error_events) == 1
        assert error_events[0]["error"]["type"] == "rate_limit_error"

    async def test_uninitialized_auth_fails_closed_as_401(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        from litestar.exceptions import NotAuthorizedException

        def _raise(_header):
            raise NotAuthorizedException

        monkeypatch.setattr(_auth_mod.session_manager, "validate", _raise)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"


class TestThinkingControlOnRoute:
    """The reasoning mode: the ``thinking`` parameter first, then the setting."""

    _THINKING_TEXT = "<think>weighing</think>Answer."

    def _thinking_provider(self, services) -> None:
        services.provider.chat.return_value = ChatResult(
            text=self._THINKING_TEXT, tool_calls=(), finish_reason=FinishReason.STOP
        )

    def _set_mode(self, monkeypatch, mode) -> None:
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "messages_reasoning", mode)

    async def test_default_reports_thinking_in_its_own_block(
        self, services_with_chat_model, _auth_token
    ):
        self._thinking_provider(services_with_chat_model)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        assert resp.json()["content"] == [
            {"type": "thinking", "thinking": "weighing"},
            {"type": "text", "text": "Answer."},
        ]

    async def test_thinking_disabled_drops_the_block(self, services_with_chat_model, _auth_token):
        self._thinking_provider(services_with_chat_model)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/messages", json=_body(thinking={"type": "disabled"}), headers=_h()
            )
        assert resp.json()["content"] == [{"type": "text", "text": "Answer."}]

    async def test_thinking_disabled_sends_think_false_to_the_provider(
        self, services_with_chat_model, _auth_token
    ):
        async with AsyncTestClient(_build_app()) as client:
            await client.post(
                "/v1/messages", json=_body(thinking={"type": "disabled"}), headers=_h()
            )
        opts = services_with_chat_model.provider.chat.call_args.kwargs["options"]
        assert opts["think"] is False

    async def test_thinking_enabled_with_budget_is_accepted(
        self, services_with_chat_model, _auth_token
    ):
        self._thinking_provider(services_with_chat_model)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/messages",
                json=_body(thinking={"type": "enabled", "budget_tokens": 2048}),
                headers=_h(),
            )
        assert resp.status_code == 200
        assert resp.json()["content"][0] == {"type": "thinking", "thinking": "weighing"}

    async def test_setting_off_applies_without_the_request_parameter(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        from lilbee.core.config.enums import ReasoningMode

        self._set_mode(monkeypatch, ReasoningMode.OFF)
        self._thinking_provider(services_with_chat_model)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        assert resp.json()["content"] == [{"type": "text", "text": "Answer."}]

    async def test_request_enabled_overrides_the_off_setting(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        from lilbee.core.config.enums import ReasoningMode

        self._set_mode(monkeypatch, ReasoningMode.OFF)
        self._thinking_provider(services_with_chat_model)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/messages", json=_body(thinking={"type": "enabled"}), headers=_h()
            )
        assert resp.json()["content"][0] == {"type": "thinking", "thinking": "weighing"}

    async def test_request_disabled_overrides_the_separate_setting(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        from lilbee.core.config.enums import ReasoningMode

        self._set_mode(monkeypatch, ReasoningMode.SEPARATE)
        self._thinking_provider(services_with_chat_model)
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/messages", json=_body(thinking={"type": "disabled"}), headers=_h()
            )
        assert resp.json()["content"] == [{"type": "text", "text": "Answer."}]

    async def test_disabled_streaming_emits_no_thinking_delta(
        self, services_with_chat_model, _auth_token
    ):
        services_with_chat_model.provider.chat.return_value = FakeProviderStream(
            ["<think>", "why", "</think>", "hi"]
        )
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/messages",
                json=_body(stream=True, thinking={"type": "disabled"}),
                headers=_h(),
            )
        events = _sse_events(resp.content)
        deltas = [p["delta"] for t, p in events if t == "content_block_delta"]
        assert deltas == [{"type": "text_delta", "text": "hi"}]
        starts = [p["content_block"]["type"] for t, p in events if t == "content_block_start"]
        assert starts == ["text"]

    async def test_unknown_thinking_shape_does_not_fail_the_request(
        self, services_with_chat_model, _auth_token
    ):
        """Claude Code must keep working when it sends a shape this surface
        does not know."""
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/messages", json=_body(thinking={"type": "adaptive"}), headers=_h()
            )
        assert resp.status_code == 200


class TestReasoningCapOnRoute:
    """The reasoning cap bounds thinking on /v1/messages, both paths."""

    @pytest.fixture(autouse=True)
    def _no_model_defaults(self):
        """A per-model override would beat the cap these tests set."""
        from lilbee.core.config import cfg

        previous = cfg.model_defaults
        cfg.clear_model_defaults()
        yield
        cfg.apply_model_defaults(previous)

    def _cap(self, monkeypatch, chars: int) -> None:
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "max_reasoning_chars", chars)

    async def test_streaming_cap_stops_thinking_and_forces_an_answer(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        self._cap(monkeypatch, 10)
        services_with_chat_model.provider.chat.side_effect = [
            FakeProviderStream(["<think>", "x" * 40, "still thinking", "</think>", "ignored"]),
            FakeProviderStream(["forced answer"]),
        ]
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(stream=True), headers=_h())
        events = _sse_events(resp.content)
        thinking = "".join(
            p["delta"]["thinking"]
            for t, p in events
            if t == "content_block_delta" and p["delta"]["type"] == "thinking_delta"
        )
        text = "".join(
            p["delta"]["text"]
            for t, p in events
            if t == "content_block_delta" and p["delta"]["type"] == "text_delta"
        )
        # The cap notice rides the thinking channel, and the answer is the
        # continuation's -- the capped stream's own answer never arrives.
        assert "reasoning capped at 10 chars" in thinking
        assert text == "forced answer"
        assert "ignored" not in text

    async def test_non_streaming_reasoning_only_turn_is_re_issued(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        self._cap(monkeypatch, 10)
        services_with_chat_model.provider.chat.side_effect = [
            ChatResult(
                text="<think>" + "x" * 50 + "</think>",
                tool_calls=(),
                finish_reason=FinishReason.STOP,
            ),
            ChatResult(text="forced answer", tool_calls=(), finish_reason=FinishReason.STOP),
        ]
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(), headers=_h())
        blocks = resp.json()["content"]
        assert {b["type"] for b in blocks} == {"thinking", "text"}
        assert next(b["text"] for b in blocks if b["type"] == "text") == "forced answer"
        assert services_with_chat_model.provider.chat.call_count == 2

    async def test_budget_tokens_tightens_the_cap(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        """A 2-token budget caps thinking at 8 chars, well under the setting."""
        self._cap(monkeypatch, 64_000)
        services_with_chat_model.provider.chat.side_effect = [
            FakeProviderStream(["<think>", "x" * 40, "</think>", "ignored"]),
            FakeProviderStream(["forced answer"]),
        ]
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/messages",
                json=_body(stream=True, thinking={"type": "enabled", "budget_tokens": 2}),
                headers=_h(),
            )
        events = _sse_events(resp.content)
        text = "".join(
            p["delta"]["text"]
            for t, p in events
            if t == "content_block_delta" and p["delta"]["type"] == "text_delta"
        )
        assert text == "forced answer"

    async def test_budget_tokens_cannot_loosen_the_configured_cap(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        """A huge budget must not buy more thinking than the operator allows."""
        self._cap(monkeypatch, 10)
        services_with_chat_model.provider.chat.side_effect = [
            FakeProviderStream(["<think>", "x" * 40, "</think>", "ignored"]),
            FakeProviderStream(["forced answer"]),
        ]
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post(
                "/v1/messages",
                json=_body(stream=True, thinking={"type": "enabled", "budget_tokens": 1_000_000}),
                headers=_h(),
            )
        events = _sse_events(resp.content)
        text = "".join(
            p["delta"]["text"]
            for t, p in events
            if t == "content_block_delta" and p["delta"]["type"] == "text_delta"
        )
        assert text == "forced answer"
        assert services_with_chat_model.provider.chat.call_count == 2

    async def test_per_model_override_flows_through_the_cap(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        """ModelDefaults beats the global setting, as effective_reasoning_cap says."""
        from lilbee.core.config import cfg
        from lilbee.providers.model_defaults import ModelDefaults

        monkeypatch.setattr(cfg, "max_reasoning_chars", 64_000)
        cfg.apply_model_defaults(ModelDefaults(max_reasoning_chars=10))
        services_with_chat_model.provider.chat.side_effect = [
            FakeProviderStream(["<think>", "x" * 40, "</think>", "ignored"]),
            FakeProviderStream(["forced answer"]),
        ]
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(stream=True), headers=_h())
        events = _sse_events(resp.content)
        text = "".join(
            p["delta"]["text"]
            for t, p in events
            if t == "content_block_delta" and p["delta"]["type"] == "text_delta"
        )
        assert text == "forced answer"

    async def test_uncapped_setting_leaves_long_reasoning_alone(
        self, services_with_chat_model, _auth_token, monkeypatch
    ):
        self._cap(monkeypatch, 0)
        services_with_chat_model.provider.chat.side_effect = [
            FakeProviderStream(["<think>", "x" * 400, "</think>", "answer"]),
        ]
        async with AsyncTestClient(_build_app()) as client:
            resp = await client.post("/v1/messages", json=_body(stream=True), headers=_h())
        events = _sse_events(resp.content)
        text = "".join(
            p["delta"]["text"]
            for t, p in events
            if t == "content_block_delta" and p["delta"]["type"] == "text_delta"
        )
        assert text == "answer"
        assert services_with_chat_model.provider.chat.call_count == 1
