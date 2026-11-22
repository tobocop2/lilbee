"""A request to a role the memory budget leaves out answers 400 with the planner's sentence."""

from __future__ import annotations

from collections.abc import Iterator
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from litestar.testing import TestClient

from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole

_REASON = "{role} model {name} does not fit available memory and will not be served; "
_FIX = "free up memory or use a smaller model."


@pytest.fixture()
def client() -> Iterator[TestClient]:
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    yield TestClient(create_app())
    auth_mod.session_manager.cleanup()


def _refusal(monkeypatch: pytest.MonkeyPatch, role: WorkerRole) -> tuple[FleetProvider, str]:
    """A real provider whose plan left *role* out, and the sentence the plan gave for it."""
    reason = _REASON.format(role=role.value, name=f"the-{role.value}-model") + _FIX
    provider = FleetProvider()
    monkeypatch.setattr(provider, "_ensure_fleet", lambda: False)
    provider._refused_roles = {role: reason}
    return provider, reason


def _request_for(provider: FleetProvider, role: WorkerRole):
    def raise_the_provider_error(*_args: object, **_kwargs: object) -> None:
        provider._require_clients(role)

    return raise_the_provider_error


def test_a_chat_request_to_a_role_the_budget_left_out_gets_the_planner_sentence(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, reason = _refusal(monkeypatch, WorkerRole.CHAT)
    with mock.patch(
        "lilbee.server.handlers.chat",
        new_callable=AsyncMock,
        side_effect=_request_for(provider, WorkerRole.CHAT),
    ):
        response = client.post("/api/chat", json={"question": "q", "history": []})
    assert response.status_code == 400
    assert reason in response.text


def test_an_embed_request_to_a_role_the_budget_left_out_gets_the_planner_sentence(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, reason = _refusal(monkeypatch, WorkerRole.EMBED)
    with mock.patch(
        "lilbee.server.handlers.search",
        new_callable=AsyncMock,
        side_effect=_request_for(provider, WorkerRole.EMBED),
    ):
        response = client.get("/api/search", params={"q": "anything"})
    assert response.status_code == 400
    assert reason in response.text
