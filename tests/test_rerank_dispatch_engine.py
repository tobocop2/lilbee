"""The rerank dispatch reads the engine of the rerank role, not the engine of chat."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import collect
from lilbee.providers.roles import WorkerRole
from lilbee.providers.routing_provider import RoutingProvider
from tests.conformance.fake_adapter import BundledFakeAdapter

MLX_REF = "org/repo#mlx"
RERANK_REF = "org/rerank/rerank-Q8_0.gguf"


class _ChatOnlyEngine(BundledFakeAdapter):
    """The engine of chat: it loads MLX models and serves no rerank role."""

    name = "mlx"
    formats = frozenset({"mlx"})
    served_roles = frozenset({WorkerRole.CHAT})


class _RerankEngine(BundledFakeAdapter):
    """The engine of rerank: it loads GGUF models and serves the rerank role."""

    name = "llama.cpp"
    formats = frozenset({"gguf"})
    served_roles = frozenset({WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK})


@pytest.fixture(autouse=True)
def two_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat runs on one adapter and the reranker on another."""
    monkeypatch.setitem(choice.ADAPTERS, EngineName.MLX, _ChatOnlyEngine())  # type: ignore[arg-type]
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _RerankEngine())  # type: ignore[arg-type]
    monkeypatch.setattr(
        runnable,
        "session_host",
        lambda: collect(system="Darwin", machine="arm64", cpu_info={}, nvidia=None),
    )
    monkeypatch.setattr(cfg, "chat_model", MLX_REF)
    monkeypatch.setattr(cfg, "embedding_model", "")
    monkeypatch.setattr(cfg, "reranker_model", RERANK_REF)
    monkeypatch.setattr(cfg, "vision_model", "")


def _provider(
    monkeypatch: pytest.MonkeyPatch, local_scores: list[float]
) -> tuple[RoutingProvider, MagicMock, MagicMock]:
    provider = RoutingProvider()
    local = MagicMock()
    local.rerank.return_value = local_scores
    local.supports_rerank.return_value = True
    sdk = MagicMock()
    sdk.supports_rerank.return_value = False
    monkeypatch.setattr(provider, "_get_local", lambda: local)
    monkeypatch.setattr(provider, "_get_sdk_provider", lambda: sdk)
    return provider, local, sdk


def test_a_rerank_request_goes_to_the_rerank_engine_while_chat_runs_elsewhere(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider, local, sdk = _provider(monkeypatch, [0.75, 0.25])
    with caplog.at_level(logging.INFO, logger="lilbee.providers.routing_provider"):
        scores = provider.rerank("q", ["a", "b"])
    assert scores == [0.75, 0.25]
    local.rerank.assert_called_once_with("q", ["a", "b"])
    sdk.rerank.assert_not_called()
    served = [r.getMessage() for r in caplog.records if r.getMessage().startswith("served by")]
    assert len(served) == 1
    assert "engine=llama.cpp " in served[0] and "role=rerank" in served[0]


def test_the_rerank_capability_reads_the_rerank_engine_while_chat_runs_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, local, sdk = _provider(monkeypatch, [])
    assert provider.supports_rerank() is True
    local.supports_rerank.assert_called_once_with()
    sdk.supports_rerank.assert_not_called()
