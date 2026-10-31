"""The log line, the status and the launch each name the engine of their own role."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from lilbee.app import engine_choice
from lilbee.app.status import chat_engine, serving_roles
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import HostFacts, NoEngineError
from lilbee.providers.roles import WorkerRole
from lilbee.providers.routing_provider import RoutingProvider
from lilbee.providers.serving import UNKNOWN_ENGINE
from tests.conformance.fake_adapter import BundledFakeAdapter, FakeAdapter

MLX_REF = "org/repo#mlx"
GGUF_REF = "org/repo/file.gguf"


class _GgufEngine(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})


class _MlxEngine(BundledFakeAdapter):
    name = "mlx"
    formats = frozenset({"mlx"})


GGUF_ENGINE = _GgufEngine()
MLX_ENGINE = _MlxEngine()


@pytest.fixture(autouse=True)
def two_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, GGUF_ENGINE)  # type: ignore[arg-type]
    monkeypatch.setitem(choice.ADAPTERS, EngineName.MLX, MLX_ENGINE)  # type: ignore[arg-type]
    monkeypatch.setattr(
        engine_choice, "host_facts", lambda: HostFacts(system="darwin", machine="arm64")
    )
    monkeypatch.setattr(cfg, "chat_model", MLX_REF)
    monkeypatch.setattr(cfg, "embedding_model", GGUF_REF)
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")


def _chat_engine() -> FakeAdapter:
    return engine_choice.EngineBook(cfg).for_role(WorkerRole.CHAT).adapter  # type: ignore[return-value]


def _served(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("served by")]


def test_one_request_names_each_roles_engine_in_log_status_and_launch(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider = RoutingProvider(engine=_chat_engine)
    local = MagicMock()
    local.embed.return_value = [[0.0]]
    local.chat.return_value = "Rome"
    monkeypatch.setattr(provider, "_get_local", lambda: local)
    with caplog.at_level(logging.INFO, logger="lilbee.providers.routing_provider"):
        provider.chat([], model=None)
        provider.embed(["x"])
    lines = _served(caplog)
    assert len(lines) == 2
    assert "engine=mlx " in lines[0] and "role=chat" in lines[0]
    assert "engine=llama.cpp " in lines[1] and "role=embed" in lines[1]
    status = {row.role: row.engine for row in serving_roles()}
    assert status == {WorkerRole.CHAT: "mlx", WorkerRole.EMBED: "llama.cpp"}
    assert chat_engine() == "mlx"


def test_the_launch_runs_the_chat_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    built = MagicMock()
    monkeypatch.setattr("lilbee.providers.fleet.provider.FleetProvider", built)
    RoutingProvider(engine=_chat_engine)._get_local()
    assert built.call_args.kwargs["engine"] is MLX_ENGINE


def test_the_log_follows_a_config_change_at_the_next_call(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider = RoutingProvider(engine=_chat_engine)
    monkeypatch.setattr(provider, "_get_local", lambda: MagicMock())
    with caplog.at_level(logging.INFO, logger="lilbee.providers.routing_provider"):
        provider.chat([], model=None)
        monkeypatch.setattr(cfg, "chat_model", GGUF_REF)
        provider.chat([], model=None)
    first, second = _served(caplog)
    assert "engine=mlx " in first
    assert "engine=llama.cpp " in second


def test_a_role_with_no_engine_still_reports_the_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delitem(choice.ADAPTERS, EngineName.MLX)
    provider = RoutingProvider(engine=_chat_engine)
    with (
        caplog.at_level(logging.INFO, logger="lilbee.providers.routing_provider"),
        pytest.raises(NoEngineError, match="For chat: No installed engine loads the mlx"),
    ):
        provider.chat([], model=None)
    assert f"engine={UNKNOWN_ENGINE} " in _served(caplog)[0]
    assert not any(isinstance(r.exc_info, tuple) and r.exc_info[0] for r in caplog.records)
    assert issubclass(NoEngineError, ProviderError)
    assert {row.role: row.engine for row in serving_roles()}[WorkerRole.CHAT] == UNKNOWN_ENGINE
