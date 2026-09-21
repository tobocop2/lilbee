"""A post-``done`` auto-extraction failure must not cost a client its answer."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from litestar.testing import AsyncTestClient

from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.retrieval.query.searcher import RagContext
from lilbee.server import auth as _auth_mod
from lilbee.server.chat_dispatch.canonical import (
    CanonicalChatRequest,
    ContentBlockDelta,
    ContentBlockStart,
    ContentBlockStop,
    MessageStart,
    MessageStop,
    TextBlock,
    TextDelta,
)
from lilbee.server.handlers import rag
from tests.server.conftest import parse_sse_events


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_auth_mod.session_manager.token}"}


def _installed_manifest(ref: str) -> MagicMock:
    m = MagicMock()
    m.ref = ref
    m.task = "chat"
    return m


@pytest.fixture
def grounded_services():
    from tests.conftest import make_mock_services

    provider = MagicMock()
    provider.max_concurrent_chats.return_value = 1
    provider.supports_tools.return_value = False
    services = make_mock_services(provider=provider)
    services.registry.list_installed = MagicMock(return_value=[_installed_manifest(cfg.chat_model)])
    services.searcher.build_rag_context = MagicMock(
        return_value=RagContext(
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


class _FakeCanonicalStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = list(events)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _answer_stream(req: CanonicalChatRequest) -> _FakeCanonicalStream:
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


class TestPostDoneExtractionFailure:
    """``/api/chat/stream`` ends at ``done`` even when auto-extraction raises."""

    async def test_drained_stream_keeps_the_answer(
        self, grounded_services, monkeypatch, caplog
    ) -> None:
        """A client that drains past ``done`` sees no error frame, and the failure logs."""
        from lilbee.server import app as app_module

        def _boom(_question: str, _answer: str) -> list:
            raise RuntimeError("extraction model is not installed")

        monkeypatch.setattr(rag, "dispatch_chat_stream", _answer_stream)
        monkeypatch.setattr(rag, "auto_extract_enabled", lambda: True)
        monkeypatch.setattr(rag, "auto_extract", _boom)

        async with AsyncTestClient(app_module.create_app()) as client:
            # App startup reconfigures logging and drops caplog's root handler,
            # so attach it to the emitting logger once the app is up.
            emitter = logging.getLogger(rag.__name__)
            emitter.addHandler(caplog.handler)
            try:
                resp = await client.post(
                    "/api/chat/stream",
                    json={"question": "q", "history": []},
                    headers=_auth_headers(),
                )
            finally:
                emitter.removeHandler(caplog.handler)

        assert resp.status_code == 201
        events = parse_sse_events(resp.content)
        kinds = [kind for kind, _ in events]
        tokens = [data["token"] for kind, data in events if kind == "token"]
        assert "".join(tokens) == "hello"
        assert "error" not in kinds, kinds
        assert kinds[-1] == "done", kinds
        assert "extraction model is not installed" in caplog.text
