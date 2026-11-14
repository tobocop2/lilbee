"""An installed-model memory estimate reads the folder on disk and never asks the Hub."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.utils import get_session

from lilbee.catalog import hf_client
from lilbee.core.config.enums import KvCacheType
from lilbee.providers import engine_params
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines.base import EngineAdapter
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.roles import WorkerRole

_CONFIG = {
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 2560,
    "max_position_embeddings": 40960,
}
_ENGINES = [MISTRAL, MLX, TEI]


@pytest.fixture
def requests_seen(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every Hub request attempted from now on, counted at the HTTP client and at name lookup."""
    seen: list[str] = []

    def send(_self: object, request: Any, **_kwargs: object) -> Any:
        seen.append(f"http {request.url}")
        raise ConnectionError("no network in a test")

    def lookup(host: object, *_args: object, **_kwargs: object) -> list[object]:
        seen.append(f"dns {host}")
        raise OSError("no network in a test")

    monkeypatch.setattr(type(get_session()), "send", send)
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    return seen


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_CONFIG), encoding="utf-8")
    with (model / "model.safetensors").open("wb") as shard:
        shard.truncate(1_000_000)
    return model


def _estimate(engine: EngineAdapter, folder: Path) -> None:
    engine.estimate_memory(MemoryRequest(folder, 8192, 1, -1, True, KvCacheType.F16))


class TestTheGuardCatchesARequest:
    """The positive control: the counter sees a request made by the code under test."""

    def test_a_config_read_is_counted(self, requests_seen: list[str]) -> None:
        assert hf_client.read_model_config("org/model") is None
        assert requests_seen
        assert "config.json" in requests_seen[0]

    def test_a_tensor_count_read_is_counted(self, requests_seen: list[str]) -> None:
        hf_client.read_parameter_count("org/model")
        assert requests_seen

    def test_a_hub_read_injected_into_an_estimate_is_counted(
        self, requests_seen: list[str], folder: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        honest = MISTRAL.estimate_memory

        def reads_the_hub(request: MemoryRequest):
            hf_client.read_parameter_count("org/model")
            return honest(request)

        monkeypatch.setattr(type(MISTRAL), "estimate_memory", lambda _self, r: reads_the_hub(r))
        _estimate(MISTRAL, folder)
        assert requests_seen


class TestAnEstimateMakesNoRequest:
    @pytest.mark.parametrize("engine", _ENGINES, ids=lambda e: e.name)
    def test_the_adapter_estimate_reads_only_the_folder(
        self, engine: EngineAdapter, folder: Path, requests_seen: list[str]
    ) -> None:
        _estimate(engine, folder)
        assert requests_seen == []

    @pytest.mark.parametrize("engine", [MISTRAL, MLX], ids=lambda e: e.name)
    def test_the_planner_estimate_reads_only_the_folder(
        self,
        engine: EngineAdapter,
        folder: Path,
        requests_seen: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: folder)
        roles: list[tuple[WorkerRole, Callable[[], object]]] = [
            (
                role,
                lambda role=role: planning._estimate_role(role, "ref", engine=engine, slots=1),
            )
            for role in (WorkerRole.CHAT, WorkerRole.EMBED)
        ]
        assert roles
        for _role, estimate in roles:
            estimate()
        assert requests_seen == []
