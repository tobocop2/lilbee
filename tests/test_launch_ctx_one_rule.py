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
from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers import engine_params, model_cache
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import UNKNOWN_SIZE_RESERVE_BYTES, estimate_memory
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
    monkeypatch.setattr(MISTRAL, "binary", lambda: Path("/x/engine"))
    monkeypatch.setattr(MLX, "binary", lambda: Path("/x/engine"))
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


_STATED = MemoryEstimate(
    kind=MemoryKind.FIXED_RESERVE, device_bytes=_GB, host_bytes=0, shared_bytes=_GB
)


def _priced_contexts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the context of every memory request an engine is asked to price."""
    seen: list[int] = []

    def estimate(self, request: MemoryRequest) -> MemoryEstimate:
        seen.append(request.ctx)
        return _STATED

    for engine in (MISTRAL, MLX, LLAMA):
        monkeypatch.setattr(type(engine), "estimate_memory", estimate)
    return seen


@pytest.mark.parametrize("engine", [MISTRAL, MLX], ids=["mistral", "mlx"])
def test_the_role_estimate_prices_the_context_the_launch_serves(
    folder: Path, monkeypatch: pytest.MonkeyPatch, engine
) -> None:
    launched = _chat_launch(engine, 1).ctx
    seen = _priced_contexts(monkeypatch)
    planning._estimate_role(WorkerRole.CHAT, "ref", engine=engine, slots=1)
    assert seen == [launched] == [_TARGET]


@pytest.mark.parametrize("engine", [MISTRAL, MLX], ids=["mistral", "mlx"])
def test_the_peak_estimator_prices_the_context_the_launch_serves(
    folder: Path, monkeypatch: pytest.MonkeyPatch, engine
) -> None:
    launched = _chat_launch(engine, 1).ctx
    seen = _priced_contexts(monkeypatch)
    planning._peak_estimator({WorkerRole.CHAT: "ref"}, {WorkerRole.CHAT: engine})(
        WorkerRole.CHAT, (1,)
    )
    assert seen == [launched] == [_TARGET]


@pytest.mark.parametrize("ctx", [_TARGET, 40960])
def test_an_xdna_model_keeps_the_unknown_size_reserve_at_any_context(
    folder: Path, ctx: int
) -> None:
    request = MemoryRequest(
        model_path=folder,
        ctx=ctx,
        slots=1,
        gpu_layers=-1,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )
    estimate = estimate_memory(ModelFormat.XDNA, request)
    assert estimate.device_bytes == UNKNOWN_SIZE_RESERVE_BYTES


@pytest.mark.parametrize("engine", [MISTRAL, MLX], ids=["mistral", "mlx"])
def test_the_split_objective_aims_at_the_context_the_launch_serves(folder: Path, engine) -> None:
    _fit, target = planning._chat_split_ctx_objective(
        {WorkerRole.CHAT: "ref"}, {WorkerRole.CHAT: engine}
    )
    assert target == _chat_launch(engine, 1).ctx == _TARGET


@pytest.mark.parametrize("engine", [MISTRAL, MLX], ids=["mistral", "mlx"])
def test_the_fallback_floor_charges_the_cache_of_the_context_the_launch_serves(
    folder: Path, engine
) -> None:
    weights = 4 * _GB
    meta = engine.model_meta(folder)
    slots = planning._placement_estimate_slots(WorkerRole.CHAT, meta)
    cache_per_token = model_cache.kv_bytes_per_token(meta, 2, 2)
    overhead = int(weights * model_cache._BUFFER_OVERHEAD_FRACTION)
    floor = planning._fallback_floor_for(WorkerRole.CHAT, "ref", weights, engine=engine)
    assert floor == weights + cache_per_token * _TARGET * slots + overhead
