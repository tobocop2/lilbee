"""Confirm ``/api/chat`` and ``/api/chat/stream`` route through ``chat_dispatch``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from litestar.testing import AsyncTestClient

from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.providers.worker.transport import ChatResult, FinishReason
from lilbee.server import auth as _auth_mod
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    CanonicalResponse,
    CanonicalUsage,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageStart,
    MessageStop,
    StopReason,
    TextBlock,
    TextDelta,
)
from tests.server.conftest import parse_sse_events


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


def _installed_manifest(ref: str) -> MagicMock:
    m = MagicMock()
    m.ref = ref
    m.task = "chat"
    return m


@pytest.fixture
def services_with_chat_dispatch():
    from tests.conftest import make_mock_services

    provider = MagicMock()
    provider.chat.return_value = ChatResult(
        text="hello", tool_calls=(), finish_reason=FinishReason.STOP
    )
    provider.supports_tools.return_value = False
    services = make_mock_services(provider=provider)
    services.registry.list_installed = MagicMock(return_value=[_installed_manifest(cfg.chat_model)])
    services.searcher.build_rag_context = MagicMock(
        return_value=(
            [],
            [
                {"role": "system", "content": "ctx"},
                {"role": "user", "content": "q"},
            ],
        )
    )
    set_services(services)
    yield services
    set_services(None)


class TestChatRouteUsesDispatch:
    async def test_chat_dispatches_canonical_request(
        self, services_with_chat_dispatch, monkeypatch
    ) -> None:
        from lilbee.server import app as app_module
        from lilbee.server.handlers import rag

        captured: list[CanonicalChatRequest] = []

        def _fake_dispatch(req: CanonicalChatRequest) -> CanonicalResponse:
            captured.append(req)
            return CanonicalResponse(
                id="msg_test",
                model=req.model,
                content=[TextBlock(text="hello")],
                stop_reason=StopReason.END_TURN,
                usage=CanonicalUsage(input_tokens=0, output_tokens=0),
            )

        monkeypatch.setattr(rag, "dispatch_chat", _fake_dispatch)

        async with AsyncTestClient(app_module.create_app()) as client:
            resp = await client.post(
                "/api/chat",
                json={"question": "q", "history": []},
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        assert resp.json()["answer"] == "hello"
        assert len(captured) == 1
        assert isinstance(captured[0], CanonicalChatRequest)
        assert captured[0].model == cfg.chat_model
        assert captured[0].tools is None


class _FakeCanonicalStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class TestChatStreamRouteUsesDispatch:
    async def test_chat_stream_dispatches_canonical_stream(
        self, services_with_chat_dispatch, monkeypatch
    ) -> None:
        from lilbee.server import app as app_module
        from lilbee.server.handlers import rag

        captured: list[CanonicalChatRequest] = []

        def _fake_dispatch_stream(req: CanonicalChatRequest) -> _FakeCanonicalStream:
            captured.append(req)
            return _FakeCanonicalStream(
                [
                    MessageStart(id="msg_test", model=req.model),
                    ContentBlockStart(index=0, block=TextBlock(text="")),
                    ContentBlockDelta(index=0, delta=TextDelta(text="hel")),
                    ContentBlockDelta(index=0, delta=TextDelta(text="lo")),
                    ContentBlockStop(index=0),
                    MessageStop(),
                ]
            )

        monkeypatch.setattr(rag, "dispatch_chat_stream", _fake_dispatch_stream)

        async with AsyncTestClient(app_module.create_app()) as client:
            resp = await client.post(
                "/api/chat/stream",
                json={"question": "q", "history": []},
                headers=_auth_headers(),
            )

        assert resp.status_code == 201
        assert "text/event-stream" in resp.headers["content-type"]
        events = parse_sse_events(resp.content)
        tokens = [data["token"] for kind, data in events if kind == "token"]
        assert "".join(tokens) == "hello"
        assert any(kind == "sources" for kind, _ in events)
        assert any(kind == "done" for kind, _ in events)
        assert len(captured) == 1
        assert isinstance(captured[0], CanonicalChatRequest)
        assert captured[0].model == cfg.chat_model

    async def test_chat_stream_emits_error_when_no_documents(
        self, services_with_chat_dispatch
    ) -> None:
        from lilbee.server import app as app_module

        services_with_chat_dispatch.searcher.build_rag_context = MagicMock(return_value=None)

        async with AsyncTestClient(app_module.create_app()) as client:
            resp = await client.post(
                "/api/chat/stream",
                json={"question": "q", "history": []},
                headers=_auth_headers(),
            )

        events = parse_sse_events(resp.content)
        assert any(kind == "error" for kind, _ in events)

    async def test_chat_stream_propagates_dispatch_error(
        self, services_with_chat_dispatch, monkeypatch
    ) -> None:
        from lilbee.server import app as app_module
        from lilbee.server.handlers import rag

        def _boom(req: CanonicalChatRequest):
            async def _gen():
                raise RuntimeError("failed to load model: free RAM too low")
                yield  # pragma: no cover

            return _gen()

        monkeypatch.setattr(rag, "dispatch_chat_stream", _boom)

        async with AsyncTestClient(app_module.create_app()) as client:
            resp = await client.post(
                "/api/chat/stream",
                json={"question": "q", "history": []},
                headers=_auth_headers(),
            )

        events = parse_sse_events(resp.content)
        kinds = [k for k, _ in events]
        assert "error" in kinds


class TestRagHandlerHelpers:
    """Exercise the helper module functions to lock down behavior."""

    def test_text_from_event_returns_text_for_text_delta(self) -> None:
        from lilbee.server.handlers.rag import _text_from_event

        event = ContentBlockDelta(index=0, delta=TextDelta(text="hi"))
        assert _text_from_event(event) == "hi"

    def test_text_from_event_returns_empty_for_other_events(self) -> None:
        from lilbee.server.handlers.rag import _text_from_event

        assert _text_from_event(MessageStart(id="m", model="x")) == ""

    def test_split_system_extracts_leading_system(self) -> None:
        from lilbee.server.handlers.rag import _split_system

        system, rest = _split_system(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
            ]
        )
        assert system == "sys"
        assert rest == [{"role": "user", "content": "u"}]

    def test_split_system_none_when_no_system(self) -> None:
        from lilbee.server.handlers.rag import _split_system

        system, rest = _split_system([{"role": "user", "content": "u"}])
        assert system is None
        assert rest == [{"role": "user", "content": "u"}]

    def test_join_text_blocks_skips_non_text(self) -> None:
        from lilbee.server.chat_dispatch.canonical import ToolUseBlock
        from lilbee.server.handlers.rag import _join_text_blocks

        out = _join_text_blocks(
            [
                TextBlock(text="a"),
                ToolUseBlock(id="1", name="t", input={}),
                TextBlock(text="b"),
            ]
        )
        assert out == "ab"

    def test_direct_messages_appends_history_and_question(
        self, services_with_chat_dispatch
    ) -> None:
        from lilbee.server.handlers.rag import _direct_messages

        msgs = _direct_messages("q", [{"role": "user", "content": "hi"}])
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": "hi"}
        assert msgs[-1] == {"role": "user", "content": "q"}

    def test_retrieval_skipped_for_chat_mode(
        self, services_with_chat_dispatch, monkeypatch
    ) -> None:
        from lilbee.core.config.enums import ChatMode
        from lilbee.server.handlers.rag import _retrieval_skipped

        monkeypatch.setattr(cfg, "chat_mode", ChatMode.CHAT.value)
        assert _retrieval_skipped() is True

    def test_retrieval_skipped_when_no_embedding(self, services_with_chat_dispatch) -> None:
        from lilbee.server.handlers.rag import _retrieval_skipped

        services_with_chat_dispatch.embedder.embedding_available = MagicMock(return_value=False)
        assert _retrieval_skipped() is True

    def test_canonical_role_accepts_known_roles(self) -> None:
        from lilbee.server.handlers.rag import _canonical_role

        assert _canonical_role("user") == "user"
        assert _canonical_role("assistant") == "assistant"
        assert _canonical_role("tool") == "tool"

    def test_canonical_role_rejects_unknown_role(self) -> None:
        from lilbee.server.handlers.rag import _canonical_role

        with pytest.raises(ValueError, match="Unsupported message role"):
            _canonical_role("system")

    def test_build_chat_messages_falls_back_when_retrieval_skipped(
        self, services_with_chat_dispatch, monkeypatch
    ) -> None:
        from lilbee.core.config.enums import ChatMode
        from lilbee.server.handlers.rag import _build_chat_messages

        monkeypatch.setattr(cfg, "chat_mode", ChatMode.CHAT.value)
        sources, msgs = _build_chat_messages("q", [], 0, None)
        assert sources == []
        assert msgs[-1] == {"role": "user", "content": "q"}

    def test_build_chat_messages_falls_back_when_rag_returns_none(
        self, services_with_chat_dispatch
    ) -> None:
        from lilbee.server.handlers.rag import _build_chat_messages

        services_with_chat_dispatch.searcher.build_rag_context = MagicMock(return_value=None)
        sources, msgs = _build_chat_messages("q", [], 0, None)
        assert sources == []
        assert msgs[-1] == {"role": "user", "content": "q"}
