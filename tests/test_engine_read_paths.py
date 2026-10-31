"""Reads of health, status and models report a role with no engine; a run refuses."""

from __future__ import annotations

import ast
import asyncio
import inspect
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from lilbee.app import services as services_mod
from lilbee.app.placement import chat_engine_ready, preview_placement
from lilbee.app.services import Services, build_services
from lilbee.app.status import gather_status, serving_roles
from lilbee.cli.commands.setup import _resolved_provider_kwargs
from lilbee.cli.launchers.server import planned_chat_ctx
from lilbee.core.config import cfg
from lilbee.core.health_warnings import HealthWarning, WarningCode
from lilbee.mcp_server import status as mcp_status
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import NoEngineError
from lilbee.providers.roles import WorkerRole
from lilbee.providers.routing_provider import RoutingProvider
from lilbee.server.handlers import health
from lilbee.server.handlers import status as status_handler
from lilbee.server.handlers.models import models_show
from tests.conformance.fake_adapter import BundledFakeAdapter

MLX_REF = "org/repo#mlx"
GGUF_REF = "org/repo/file.gguf"
ROLES = {WorkerRole.CHAT: "chat_model", WorkerRole.EMBED: "embedding_model"}
NO_ENGINE_TEXT = "No installed engine loads the mlx format"


class _GgufEngine(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    architectures = frozenset({"gguf_arch"})
    served_roles = frozenset(WorkerRole)


@pytest.fixture(params=list(ROLES), ids=lambda role: role.value)
def no_engine(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Services]:
    """A container where one role names a format no engine loads, and the other a GGUF."""
    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX, raising=False)
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _GgufEngine())  # type: ignore[arg-type]
    monkeypatch.setattr(cfg, "reranker_model", GGUF_REF)
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "documents_dir", tmp_path / "documents")
    monkeypatch.setattr(cfg, "data_dir", tmp_path / "data")
    monkeypatch.setattr(cfg, "lancedb_dir", tmp_path / "data" / "lancedb")
    monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    for role, field in ROLES.items():
        monkeypatch.setattr(cfg, field, MLX_REF if role is request.param else GGUF_REF)
    services = build_services(cfg)
    monkeypatch.setattr(services_mod._state, "singleton", services)
    yield services
    services.store.close()


def _no_engine_warnings(warnings: list[HealthWarning]) -> list[str]:
    return [w.message for w in warnings if NO_ENGINE_TEXT in w.message]


READS: dict[str, Callable[[Services], Any]] = {
    "embed_token_cap": lambda s: s.provider.embed_token_cap(),
    "list_models": lambda s: s.provider.list_models(),
    "show_model": lambda s: s.provider.show_model(cfg.chat_model),
    "show_route": lambda s: asyncio.run(models_show(cfg.chat_model)),
    "get_capabilities": lambda s: s.provider.get_capabilities(cfg.chat_model),
    "supports_rerank": lambda s: s.provider.supports_rerank(),
    "warm_pending": lambda s: s.provider.warm_pending(),
    "spawn_listener": lambda s: s.add_pool_listener(),
    "chat_engine_ready": lambda s: chat_engine_ready(),
    "gather_status": lambda s: gather_status(),
    "serving_roles": lambda s: serving_roles(),
    "status_route": lambda s: asyncio.run(status_handler()),
    "mcp_status": lambda s: mcp_status(),
    "preview_placement": lambda s: preview_placement(),
    "planned_chat_ctx": lambda s: planned_chat_ctx(),
    "setup_provider_kwargs": lambda s: _resolved_provider_kwargs(),
}


@pytest.mark.parametrize("path", list(READS))
def test_a_read_does_not_raise_for_a_role_with_no_engine(no_engine: Services, path: str) -> None:
    READS[path](no_engine)


def test_the_health_warnings_hold_one_notice(no_engine: Services) -> None:
    warnings = no_engine.provider.health_warnings()
    assert len(_no_engine_warnings(warnings)) == 1
    assert {w.code for w in warnings} == {WarningCode.ENGINE_FALLBACK}


def test_the_health_route_answers_with_one_notice(no_engine: Services) -> None:
    response = asyncio.run(health())
    assert response.status == "ok"
    assert response.chat_ready is False
    assert len(_no_engine_warnings(response.warnings)) == 1


def test_a_run_of_the_role_refuses_with_the_engine_text(no_engine: Services) -> None:
    run: Callable[[], Any] = (
        (lambda: no_engine.provider.chat([{"role": "user", "content": "hi"}]))
        if cfg.chat_model == MLX_REF
        else (lambda: no_engine.provider.embed(["hi"]))
    )
    with pytest.raises(NoEngineError, match=NO_ENGINE_TEXT):
        run()


def test_a_listener_added_with_no_engine_reaches_the_engine_once_it_builds(
    no_engine: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    def on_spawning(role: WorkerRole) -> None: ...

    no_engine.add_pool_listener(on_spawning=on_spawning)
    monkeypatch.setattr(cfg, "chat_model", GGUF_REF)
    monkeypatch.setattr(cfg, "embedding_model", GGUF_REF)
    assert no_engine.provider._get_local()._on_spawning is on_spawning


# The methods that send work to the local engine, and may refuse with the engine text.
_RUN_METHODS = {
    "_pick_backend",
    "rerank",
    "warm_up_pool",
    "_local_for_read",
}


def test_only_a_run_reaches_the_call_that_builds_the_engine() -> None:
    tree = ast.parse(inspect.getsource(RoutingProvider))
    callers = {
        method.name
        for method in ast.walk(tree)
        if isinstance(method, ast.FunctionDef)
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute) and node.attr == "_get_local"
    }
    assert callers == _RUN_METHODS
