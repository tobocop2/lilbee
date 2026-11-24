"""The chat route answers with the installed chat model when no chat model is set, as ask does."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest import mock
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


STALE_EMBEDDER = "ollama/nomic-embed-text:latest"
UNREACHABLE_CHAT = "ollama/llama3:latest"


@pytest.fixture
def stale_embedder(monkeypatch: pytest.MonkeyPatch):
    """An embedding ref that is not installed, beside an installed embedding model."""
    install_fake_model("org/Installed-Embed-GGUF", "installed-embed-Q8_0.gguf", task="embedding")
    monkeypatch.setattr(cfg, "embedding_model", STALE_EMBEDDER)
    return cfg.embedding_dim


async def _chat(app_module):
    async with AsyncTestClient(app_module.create_app()) as client:
        return await client.post(
            "/api/chat",
            json={"question": "q", "history": []},
            headers={"Authorization": f"Bearer {_auth_mod.session_manager.token}"},
        )


async def test_a_set_chat_model_whose_server_is_down_at_start_is_kept(
    installed_chat_model, stale_embedder, monkeypatch
) -> None:
    from lilbee.server import app as app_module

    _ref, provider = installed_chat_model
    monkeypatch.setattr(cfg, "chat_model", UNREACHABLE_CHAT)
    with (
        mock.patch("lilbee.providers.availability.provider_sdk_available", return_value=True),
        mock.patch(
            "lilbee.modelhub.model_manager.validation._local_server_reachable",
            return_value=False,
        ),
    ):
        resp = await _chat(app_module)

    assert resp.status_code == 404
    assert UNREACHABLE_CHAT in resp.text
    assert cfg.chat_model == UNREACHABLE_CHAT
    provider.chat.assert_not_called()


async def test_the_embedding_model_is_unchanged_after_startup_with_a_set_chat_model(
    installed_chat_model, stale_embedder, monkeypatch
) -> None:
    from lilbee.server import app as app_module

    monkeypatch.setattr(cfg, "chat_model", UNREACHABLE_CHAT)
    with (
        mock.patch("lilbee.providers.availability.provider_sdk_available", return_value=True),
        mock.patch(
            "lilbee.modelhub.model_manager.validation._local_server_reachable",
            return_value=False,
        ),
    ):
        await _chat(app_module)

    assert cfg.embedding_model == STALE_EMBEDDER
    assert cfg.embedding_dim == stale_embedder


async def test_the_embedding_model_is_unchanged_after_startup_with_an_empty_chat_model(
    installed_chat_model, stale_embedder
) -> None:
    from lilbee.server import app as app_module

    ref, provider = installed_chat_model
    await _chat(app_module)

    assert provider.chat.call_args.kwargs["model"] == ref
    assert cfg.embedding_model == STALE_EMBEDDER
    assert cfg.embedding_dim == stale_embedder


def test_an_empty_chat_role_with_an_api_key_logs_a_cloud_model(
    installed_chat_model, caplog
) -> None:
    from lilbee.server.app import _start_services

    cloud = SimpleNamespace(name="gpt-4o", provider="openai")
    with (
        mock.patch(
            "lilbee.modelhub.model_manager.validation.discover_api_models",
            return_value={"openai": [cloud]},
        ),
        mock.patch("lilbee.server.app.get_services"),
        mock.patch("lilbee.server.app._log_embedding_model_state"),
        caplog.at_level(logging.INFO, logger="lilbee.server.app"),
    ):
        _start_services()

    assert cfg.chat_model == "openai/gpt-4o"
    assert "No chat model configured; using cloud model 'openai/gpt-4o'." in caplog.messages
    assert "using installed" not in caplog.text


async def test_a_raise_in_the_swap_is_reported_as_a_failed_start(caplog) -> None:
    from lilbee.server.app import _lifespan

    with (
        mock.patch(
            "lilbee.server.app.fill_empty_chat_model",
            side_effect=OSError("registry unreadable"),
        ),
        mock.patch("lilbee.server.app.get_services") as get_services,
        caplog.at_level(logging.ERROR, logger="lilbee.app.services"),
        pytest.raises(OSError, match="registry unreadable"),
    ):
        async with _lifespan(mock.MagicMock()):
            pytest.fail("the server served after a failed swap")
    assert "The services build failed (the chat-model fill or the build)" in caplog.text
    get_services.assert_not_called()


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
