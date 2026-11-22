"""The chat route answers with the installed chat model when no chat model is set, as ask does."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from litestar.testing import AsyncTestClient

from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.providers.base import ChatResult, FinishReason
from lilbee.retrieval.query.searcher import RagContext
from lilbee.server import auth as _auth_mod
from tests.conftest import install_fake_model, make_mock_services


@pytest.fixture
def installed_chat_model(monkeypatch: pytest.MonkeyPatch):
    """An empty chat role, one installed chat model, and services that serve only it."""
    monkeypatch.setattr(cfg, "chat_model", "")
    ref = install_fake_model("org/Installed-Chat-GGUF", "installed-chat-Q4_K_M.gguf", task="chat")
    provider = MagicMock()
    provider.chat.return_value = ChatResult(
        text="hello", tool_calls=(), finish_reason=FinishReason.STOP
    )
    provider.supports_tools.return_value = False
    provider.max_concurrent_chats.return_value = 1
    services = make_mock_services(provider=provider)
    services.known_models.resolve.side_effect = lambda model: model if model == ref else None
    services.searcher.build_rag_context = MagicMock(
        return_value=RagContext([], [{"role": "user", "content": "q"}])
    )
    set_services(services)
    yield ref, provider
    set_services(None)


async def test_an_empty_chat_role_answers_with_the_installed_chat_model(
    installed_chat_model, caplog
) -> None:
    from lilbee.server import app as app_module

    ref, provider = installed_chat_model
    with caplog.at_level(logging.INFO, logger="lilbee.server.app"):
        async with AsyncTestClient(app_module.create_app()) as client:
            resp = await client.post(
                "/api/chat",
                json={"question": "q", "history": []},
                headers={"Authorization": f"Bearer {_auth_mod.session_manager.token}"},
            )

    assert resp.status_code == 201, resp.text
    assert resp.json()["answer"] == "hello"
    assert provider.chat.call_args.kwargs["model"] == ref
    assert f"No chat model configured; using installed {ref!r}." in caplog.messages
