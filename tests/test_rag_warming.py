"""Tests for the SSE cold-start "warming" notice on the RAG streaming handlers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from lilbee.providers.roles import WorkerRole
from lilbee.runtime.progress import SseEvent
from lilbee.server.handlers import rag


def _services_with_role_ready(ready: bool) -> MagicMock:
    services = MagicMock()
    services.provider.role_ready.return_value = ready
    return services


def test_warming_event_emitted_when_chat_cold(monkeypatch) -> None:
    monkeypatch.setattr(rag, "get_services", lambda: _services_with_role_ready(False))
    events = rag._chat_warming_events()
    assert len(events) == 1
    head, _, body = events[0].partition("\n")
    assert head == f"event: {SseEvent.WARMING}"
    payload = json.loads(body.removeprefix("data: ").strip())
    assert payload == {"role": WorkerRole.CHAT.value}


def test_no_warming_event_when_chat_ready(monkeypatch) -> None:
    services = _services_with_role_ready(True)
    monkeypatch.setattr(rag, "get_services", lambda: services)
    assert rag._chat_warming_events() == []
    services.provider.role_ready.assert_called_once_with(WorkerRole.CHAT)
