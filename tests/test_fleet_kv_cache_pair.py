"""A chat launch always gives llama-server one KV cache type for both K and V.

llama-server refuses different K and V types on an MLA model (DeepSeek), turns
flash attention on by itself for a quantized V under ``auto``, and refuses a
quantized V when flash attention is ``off``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines import llama as llama_mod
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.fleet.vram import GgufVramEstimate
from lilbee.providers.roles import WorkerRole

_GB = 1024**3
_DENSE_META = {"general.architecture": "llama"}
_MLA_META = {"general.architecture": "deepseek2", "deepseek2.attention.kv_lora_rank": "512"}


def _chat_argv(tmp_path: Path, monkeypatch, *, backend: str, flash: bool | None, meta) -> list[str]:
    """The argv the fleet builds for a single-card chat launch on *backend*."""
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x" * 2048)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: dict(meta))
    monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a, engine: 4096)
    monkeypatch.setattr(planning_mod, "_fleet_backend", lambda engine: backend)
    monkeypatch.setattr(cfg, "flash_attention", flash)
    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.Q8_0)
    device = FleetDevice(backend, 0, "gpu", 24 * _GB, 23 * _GB)
    plan = InstancePlan(role=WorkerRole.CHAT, devices=(0,))
    launch = planning_mod._launch_for(
        plan, "ref", Path("/bin/llama-server"), {0: device}, engine=LLAMA
    )
    return launch.argv


def _flag(argv: list[str], flag: str) -> str:
    """The value after *flag*, or f16 (llama-server's default) when it is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else KvCacheType.F16.value


# (backend, flash_attention setting, --flash-attn sent, the one KV type for K and V)
_CASES = [
    pytest.param("CUDA", None, "on", "q8_0", id="on"),
    pytest.param("Vulkan", None, "auto", "f16", id="auto"),
    pytest.param("CUDA", False, "off", "f16", id="off"),
]


@pytest.mark.parametrize("meta", [_DENSE_META, _MLA_META], ids=["dense", "mla"])
@pytest.mark.parametrize(("backend", "flash", "flash_flag", "kv_type"), _CASES)
def test_chat_launch_sends_one_kv_type_for_k_and_v(
    tmp_path, monkeypatch, meta, backend, flash, flash_flag, kv_type
) -> None:
    argv = _chat_argv(tmp_path, monkeypatch, backend=backend, flash=flash, meta=meta)

    assert _flag(argv, "--flash-attn") == flash_flag
    assert _flag(argv, "--cache-type-k") == _flag(argv, "--cache-type-v") == kv_type


def test_auto_does_not_send_a_quantized_v_that_forces_flash_attention_on(
    tmp_path, monkeypatch
) -> None:
    """Under auto a quantized V makes llama-server force flash attention on."""
    argv = _chat_argv(tmp_path, monkeypatch, backend="Vulkan", flash=None, meta=_MLA_META)

    assert "--cache-type-v" not in argv
    assert "--cache-type-k" not in argv


def _estimate() -> GgufVramEstimate:
    """What gguf-parser would return; the test reads the KV type it was asked for."""
    return GgufVramEstimate(vram_bytes=1, ram_bytes=0, unified_bytes=1)


@pytest.mark.parametrize(("backend", "flash", "flash_flag", "kv_type"), _CASES)
def test_chat_placement_estimate_sizes_the_kv_type_the_launch_sends(
    tmp_path, monkeypatch, backend, flash, flash_flag, kv_type
) -> None:
    """The placement estimate and the launch argv agree on the one KV type."""
    argv = _chat_argv(tmp_path, monkeypatch, backend=backend, flash=flash, meta=_DENSE_META)
    asked: list[KvCacheType] = []

    def _record(*_a, kv_cache_type, **_k):
        asked.append(kv_cache_type)
        return _estimate()

    monkeypatch.setattr(llama_mod, "estimate_instance_footprint", _record)
    monkeypatch.setattr(planning_mod, "_placement_estimate_ctx", lambda _r, _p, _m: 4096)

    planning_mod._estimate_role(WorkerRole.CHAT, "ref", slots=1, engine=LLAMA)

    assert asked == [KvCacheType(_flag(argv, "--cache-type-k"))]
    assert _flag(argv, "--cache-type-v") == kv_type
