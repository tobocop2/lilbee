"""The tool check through the chat-completions route.

It runs over the real fleet provider and the real engine adapters. Only the
model registry, the set of installed engines and the chat call itself are
stood in. A GGUF file is judged by its own header on every engine. A model
folder has no GGUF header, so it passes to its engine.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from litestar import Litestar
from litestar.testing import AsyncTestClient

import lilbee.app.engine_choice as engine_choice
from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.providers.base import ChatResult, FinishReason
from lilbee.providers.fleet import provider as fleet
from lilbee.server import auth as _auth_mod
from lilbee.server.chat_completions_api.routes import completions_router
from lilbee.server.chat_dispatch.concurrency import chat_gate

_TOOL = {
    "type": "function",
    "function": {"name": "f", "description": "d", "parameters": {"type": "object"}},
}
_NO_TOOLS_ERROR = "model_does_not_support_tools"
_NO_ENGINE_MESSAGE = (
    "Engine mistral.rs for chat is not installed. No installed engine loads the safetensors format."
)


@pytest.fixture(autouse=True)
def _route_state() -> Iterator[None]:
    """Fix the token and drop the module caches, so each cell starts clean."""
    previous = (_auth_mod.session_manager.token, _auth_mod.session_manager._initialized)
    _auth_mod.session_manager.token = "t" * 50
    _auth_mod.session_manager._initialized = True
    chat_gate.cache_clear()
    fleet._supports_tools_cached.cache_clear()
    yield
    chat_gate.cache_clear()
    fleet._supports_tools_cached.cache_clear()
    _auth_mod.session_manager.token, _auth_mod.session_manager._initialized = previous
    set_services(None)


def _plain_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "plain.gguf"
    path.write_bytes(b"GGUF")
    return path


def _tool_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "tools.gguf"
    path.write_bytes(b"GGUF")
    return path


def _corrupt_file(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.gguf"
    path.write_bytes(b"not a gguf file at all" * 4)
    return path


# What the external header parser answers for each file: the parser binary is
# sealed off in unit tests, so its answer is stood in, keyed on the file name.
_HEADERS: dict[str, dict[str, str] | None] = {
    "plain.gguf": {"chat_template": "{{ messages }}"},
    "tools.gguf": {"chat_template": "{% for t in tools %}{{ t }}{% endfor %}"},
    "corrupt.gguf": None,
}


def _model_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "snapshot"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"x" * 64)
    return folder


async def _ask(
    monkeypatch: pytest.MonkeyPatch,
    model_path: Path,
    ref: str,
    engine: str,
    installed: set[str],
    *,
    tools: bool = True,
) -> tuple[Any, MagicMock]:
    """POST one chat request for *model_path* and return the response and the chat stand-in."""
    from tests.conftest import make_mock_services

    provider = fleet.FleetProvider()
    chat = MagicMock(
        return_value=ChatResult(text="ok", tool_calls=(), finish_reason=FinishReason.STOP)
    )
    monkeypatch.setattr(provider, "chat", chat)
    services = make_mock_services(provider=provider)
    services.registry.resolve = MagicMock(return_value=model_path)
    services.known_models.resolve = MagicMock(side_effect=lambda m: m)
    services.known_models.refs = MagicMock(return_value={ref})
    set_services(services)
    monkeypatch.setattr(
        "lilbee.providers.gguf_meta.read_gguf_metadata", lambda p: _HEADERS[Path(p).name]
    )
    monkeypatch.setattr(cfg, "chat_model", ref)
    monkeypatch.setattr(cfg, "engine_choices", json.dumps({ref: engine}))
    monkeypatch.setattr(
        engine_choice,
        "installed_engines",
        lambda: {
            name: adapter.formats
            for name, adapter in engine_choice.ADAPTERS.items()
            if name in installed
        },
    )
    body: dict[str, Any] = {"model": ref, "messages": [{"role": "user", "content": "hi"}]}
    if tools:
        body["tools"] = [_TOOL]
    app = Litestar(route_handlers=[completions_router], logging_config=None)
    async with AsyncTestClient(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {_auth_mod.session_manager.token}"},
        )
    return response, chat


@pytest.mark.parametrize(
    ("make_model", "ref", "engine", "installed", "status", "code"),
    [
        (_plain_gguf, "org/repo/plain.gguf", "llama.cpp", "llama.cpp", 400, _NO_TOOLS_ERROR),
        (_tool_gguf, "org/repo/tools.gguf", "llama.cpp", "llama.cpp", 200, None),
        (_corrupt_file, "org/repo/corrupt.gguf", "llama.cpp", "llama.cpp", 400, _NO_TOOLS_ERROR),
        (_model_folder, "org/repo#safetensors", "mistral.rs", "mistral.rs", 200, None),
        (_model_folder, "org/repo#safetensors", "mistral.rs", "llama.cpp", 400, "invalid_request"),
        (_plain_gguf, "org/repo/plain.gguf", "mistral.rs", "mistral.rs", 400, _NO_TOOLS_ERROR),
    ],
    ids=[
        "plain-gguf-on-llama",
        "tool-gguf-on-llama",
        "corrupt-file-on-llama",
        "folder-on-mistral",
        "folder-with-no-engine",
        "plain-gguf-on-mistral",
    ],
)
async def test_tool_request_status_per_model_and_engine(
    monkeypatch, tmp_path, make_model, ref, engine, installed, status, code
) -> None:
    path = make_model(tmp_path)
    response, chat = await _ask(monkeypatch, path, ref, engine, {installed})
    assert response.status_code == status
    if code is None:
        assert response.json()["choices"][0]["message"]["content"] == "ok"
        assert chat.call_count == 1
    else:
        assert response.json()["error"]["code"] == code
        assert chat.call_count == 0


async def test_folder_with_no_engine_names_the_problem(monkeypatch, tmp_path) -> None:
    response, _ = await _ask(
        monkeypatch,
        _model_folder(tmp_path),
        "org/repo#safetensors",
        "mistral.rs",
        {"llama.cpp"},
    )
    assert response.json()["error"] == {
        "message": _NO_ENGINE_MESSAGE,
        "type": "invalid_request_error",
        "code": "invalid_request",
    }
