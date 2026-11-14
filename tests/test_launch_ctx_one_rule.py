"""A launch, the pull verdict and the placement estimate price one chat context.

Below the 8192 placement floor a fixed-reserve engine serves the target it was
given, so the verdict and the estimate must charge that same window.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.catalog import hf_client
from lilbee.catalog.pull_fit import chat_pull_fit
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.roles import WorkerRole

_GB = 1024**3
_TARGET = 4096
_QWEN3_4B = {
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 2560,
    "max_position_embeddings": 40960,
    "sliding_window": None,
}


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A model folder the planner resolves every reference to, at a 4096 target."""
    model = tmp_path / "Qwen3-4B"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_QWEN3_4B), encoding="utf-8")
    header = json.dumps({"__metadata__": {}}).encode()
    (model / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: model)
    monkeypatch.setattr(planning, "plan_sizing_budget", lambda *_a, **_k: 64 * _GB)
    monkeypatch.setattr(type(MISTRAL), "binary", lambda self: Path("/x/engine"))
    monkeypatch.setattr(type(MLX), "binary", lambda self: Path("/x/engine"))
    monkeypatch.setattr(
        type(MISTRAL), "_device_flags", lambda self, target: ["--paged-attn", "off"]
    )
    monkeypatch.setattr(
        type(MISTRAL), "pick_build", lambda self, host: SimpleNamespace(backend="cuda-12")
    )
    monkeypatch.setattr(cfg, "chat_n_ctx_target", _TARGET)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "num_ctx_max", None)
    return model


def _flag(argv: list[str], name: str) -> int:
    return int(argv[argv.index(name) + 1])


def _chat_launch(engine, cards: int):
    devices = {i: FleetDevice("CUDA", i, f"gpu{i}", 24 * _GB, 23 * _GB) for i in range(cards)}
    plan = InstancePlan(
        role=WorkerRole.CHAT,
        devices=tuple(devices),
        tensor_split=(1,) * cards if cards > 1 else (),
    )
    return planning._launch_for(plan, "ref", devices, engine=engine)


def _verdict_ctx(monkeypatch: pytest.MonkeyPatch, engine) -> int:
    monkeypatch.setattr(hf_client, "read_model_config", lambda repo, token=None: dict(_QWEN3_4B))
    monkeypatch.setattr(hf_client, "read_parameter_count", lambda repo, token=None: {"BF16": 100})
    return chat_pull_fit("org/model", engine).ctx


@pytest.mark.parametrize("cards", [1, 2])
def test_a_mistral_launch_below_the_floor_passes_the_target_as_its_window(
    folder: Path, cards: int
) -> None:
    devices = {i: FleetDevice("CUDA", i, f"gpu{i}", 24 * _GB, 23 * _GB) for i in range(cards)}
    plan = InstancePlan(
        role=WorkerRole.CHAT,
        devices=tuple(devices),
        tensor_split=(1,) * cards if cards > 1 else (),
    )
    launch = planning._launch_for(plan, "ref", devices, engine=MISTRAL)
    assert _flag(list(launch.argv), "--max-model-len") == _TARGET


@pytest.mark.parametrize("engine", [MISTRAL, MLX], ids=["mistral", "mlx"])
def test_the_verdict_prices_the_context_the_launch_serves(
    folder: Path, monkeypatch: pytest.MonkeyPatch, engine
) -> None:
    assert _verdict_ctx(monkeypatch, engine) == _chat_launch(engine, 1).ctx == _TARGET


@pytest.mark.parametrize("engine", [MISTRAL, MLX], ids=["mistral", "mlx"])
def test_the_placement_estimate_prices_the_context_the_launch_serves(folder: Path, engine) -> None:
    estimate = planning._placement_estimate_ctx(
        WorkerRole.CHAT, folder, engine.model_meta(folder), engine=engine
    )
    assert estimate == _chat_launch(engine, 1).ctx == _TARGET


def test_a_sized_engine_keeps_the_floor_in_its_placement_reserve(folder: Path) -> None:
    estimate = planning._placement_estimate_ctx(
        WorkerRole.CHAT, folder, {"context_length": "40960"}, engine=LLAMA
    )
    assert estimate == planning._MIN_USABLE_CHAT_CTX


def test_a_pin_is_priced_as_it_is_for_every_engine(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "num_ctx", 3000)
    for engine in (MISTRAL, LLAMA):
        assert (
            planning._placement_estimate_ctx(
                WorkerRole.CHAT, folder, {"context_length": "40960"}, engine=engine
            )
            == 3000
        )
