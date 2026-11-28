"""The TEI adapter: host gate, builds, roles, memory, argv, and its rerank route."""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import json
import logging
import re
import stat
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from pytest_httpserver import HTTPServer
from werkzeug import Request, Response

from lilbee.app import engine_service
from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.engine_params import EMBED_FALLBACK_CTX
from lilbee.providers.fleet import client as fleet_client
from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.client import LlamaServerClient
from lilbee.providers.fleet.engine_store import EngineStore, PinLock
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines import tei as tei_module
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.fleet.host_caps import HostCaps, collect
from lilbee.providers.fleet.memory_types import MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import ENCODER_RUNTIME_BYTES, UNKNOWN_SIZE_RESERVE_BYTES
from lilbee.providers.fleet.placement import ModelPlacementInput, Placement, plan_placement
from lilbee.providers.roles import RoleTarget, WorkerRole
from lilbee.providers.truncation import note_truncated, truncation_scope
from tests._safetensors import safetensors_bytes, safetensors_mixed_bytes
from tests._stated_sizes import state_size

FIXTURES = Path(__file__).parent / "conformance" / "fixtures" / "engines"
L4 = ("NVIDIA L4", (8, 9))
GTX = ("NVIDIA GeForce GTX 1070 Ti", (6, 1))
T4 = ("Tesla T4", (7, 5))
A100 = ("A100", (8, 0))


def _caps(
    system: str = "Linux",
    machine: str = "x86_64",
    gpu: tuple[str, tuple[int, int]] | None = None,
    cuda: int = 12080,
) -> HostCaps:
    nvidia = None
    if gpu is not None:
        nvidia = {
            "driver": "570.0",
            "cuda_driver": cuda,
            "gpus": [{"name": gpu[0], "cc": list(gpu[1])}],
        }
    return collect(
        system=system,
        machine=machine,
        cpu_info={"flags": ["avx", "avx2"], "brand_raw": "Test CPU"},
        nvidia=nvidia,
    )


@pytest.mark.parametrize(
    ("compute", "backend"),
    [
        ((7, 5), "cuda-sm75"),
        ((8, 0), "cuda-sm80"),
        ((8, 6), "cuda-sm86"),
        ((8, 9), "cuda-sm89"),
        ((9, 0), "cuda-sm90"),
        ((10, 0), "cuda-sm100"),
        ((12, 0), "cuda-sm120"),
        ((8, 7), "cuda-sm86"),
    ],
)
def test_cuda_host_gets_the_build_of_its_compute_capability(
    compute: tuple[int, int], backend: str
) -> None:
    pick = TEI.pick_build(_caps(gpu=("GPU", compute)))
    assert pick.backend == backend


def test_a_host_with_two_cards_gets_the_build_of_the_newest_card() -> None:
    nvidia = {
        "driver": "570.0",
        "cuda_driver": 12080,
        "gpus": [{"name": "NVIDIA T4", "cc": [7, 5]}, {"name": L4[0], "cc": list(L4[1])}],
    }
    host = collect(
        system="Linux",
        machine="x86_64",
        cpu_info={"flags": ["avx", "avx2"], "brand_raw": "Test CPU"},
        nvidia=nvidia,
    )
    assert TEI.pick_build(host).backend == "cuda-sm89"


def test_gpu_older_than_turing_gets_the_cpu_build_with_the_floor_in_the_reason() -> None:
    pick = TEI.pick_build(_caps(gpu=GTX))
    assert pick.backend == "cpu"
    assert "7.5" in pick.reason
    assert "6.1" in pick.reason


def test_gpu_with_no_build_for_its_major_version_gets_the_cpu_build() -> None:
    pick = TEI.pick_build(_caps(gpu=("GPU", (11, 0))))
    assert pick.backend == "cpu"
    assert "11.0" in pick.reason


def test_host_with_no_gpu_gets_the_cpu_build() -> None:
    pick = TEI.pick_build(_caps())
    assert pick.backend == "cpu"
    assert "NVIDIA" in pick.reason


def test_apple_silicon_gets_the_metal_build() -> None:
    assert TEI.pick_build(_caps("Darwin", "arm64")).backend == "metal"


@pytest.mark.parametrize(
    ("system", "machine", "named"),
    [
        ("Windows", "AMD64", "Windows"),
        ("Darwin", "x86_64", "macOS"),
        ("Linux", "aarch64", "architecture"),
    ],
)
def test_host_with_no_build_is_refused_naming_the_cause(
    system: str, machine: str, named: str
) -> None:
    caps = _caps(system, machine, gpu=("GPU", (8, 6)))
    pick = TEI.pick_build(caps)
    gate = TEI.host_gate(caps)
    assert pick.backend is None
    assert named in pick.reason
    assert (gate.ok, gate.reason) == (False, pick.reason)


def test_host_gate_passes_a_host_with_a_build() -> None:
    assert TEI.host_gate(_caps(gpu=L4)).ok


def test_unknown_operating_system_has_no_build() -> None:
    assert TEI.pick_build(_caps("Plan9", "x86_64")).backend is None


def test_serves_embed_and_rerank_only() -> None:
    assert TEI.served_roles == frozenset({WorkerRole.EMBED, WorkerRole.RERANK})
    assert TEI.role_spec(WorkerRole.RERANK, None, None).endpoint_path == "/rerank"
    assert TEI.role_spec(WorkerRole.EMBED, None, None).endpoint_path == "/v1/embeddings"
    with pytest.raises(ValueError, match="serves embed and rerank, not chat"):
        TEI.role_spec(WorkerRole.CHAT, None, None)


def test_choosing_tei_for_chat_is_refused_naming_chat_and_the_roles_it_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "org/repo/file.gguf")
    with pytest.raises(engine_service.EngineRefusedError) as raised:
        engine_service.use_for_role(WorkerRole.CHAT, "org/repo/file.gguf", "tei")
    assert str(raised.value) == "Engine tei does not serve chat. It serves embed and rerank."


def test_choosing_auto_for_a_role_skips_the_served_roles_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "chat_model", "org/repo/file.gguf")
    assert engine_service.use_for_role(WorkerRole.CHAT, "org/repo/file.gguf", "auto") == "auto"


def _encoder_folder(root: Path, weight_bytes: int, dtype: str | None) -> Path:
    folder = root / "encoder"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(safetensors_bytes(weight_bytes))
    config = {"torch_dtype": dtype} if dtype is not None else {}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


def _encoder_request(model_path: Path, ctx: int = 512) -> MemoryRequest:
    return MemoryRequest(
        model_path=model_path,
        ctx=ctx,
        slots=1,
        gpu_layers=0,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )


def test_memory_is_the_large_vocabulary_charge_for_a_config_with_no_vocabulary_on_a_cpu_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    folder = _encoder_folder(tmp_path, 4_000, "float32")
    estimate = TEI.estimate_memory(_encoder_request(folder))
    assert estimate.kind is MemoryKind.FIXED_RESERVE
    charge = 4_000 * 5 // 4 + 1280 * _MIB
    assert estimate.device_bytes == estimate.shared_bytes == charge
    assert estimate.size_known is True


_MIB = 1024**2


def _sparse_encoder_folder(
    root: Path, weight_mib: float, vocab_size: int | None, dtype: str | None = None
) -> Path:
    """A folder whose weight file states its size, so a large size costs no disk or memory.

    The test must use the ``stated_sizes`` fixture.
    """
    folder = root / "sparse-encoder"
    folder.mkdir()
    state_size(folder / "model.safetensors", int(weight_mib * _MIB))
    config: dict[str, Any] = {} if vocab_size is None else {"vocab_size": vocab_size}
    if dtype is not None:
        config["torch_dtype"] = dtype
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


@pytest.mark.usefixtures("stated_sizes")
def test_a_large_vocabulary_is_charged_its_load_peak_on_a_cpu_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    folder = _sparse_encoder_folder(tmp_path, 2135, 250_002)
    charge = TEI.estimate_memory(_encoder_request(folder)).device_bytes
    assert 3754 * _MIB <= charge <= 3754 * _MIB * 1.3


@pytest.mark.usefixtures("stated_sizes")
def test_a_small_vocabulary_is_charged_within_1_3_times_its_load_peak_on_a_cpu_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    folder = _sparse_encoder_folder(tmp_path, 127, 30_522)
    charge = TEI.estimate_memory(_encoder_request(folder)).device_bytes
    assert 222 * _MIB <= charge <= 222 * _MIB * 1.3


@pytest.mark.usefixtures("stated_sizes")
def test_a_config_with_no_vocabulary_is_charged_as_a_large_one_on_a_cpu_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    missing = TEI.estimate_memory(_encoder_request(_sparse_encoder_folder(tmp_path, 418, None)))
    other = tmp_path / "other"
    other.mkdir()
    large = TEI.estimate_memory(_encoder_request(_sparse_encoder_folder(other, 418, 250_002)))
    assert missing.device_bytes == large.device_bytes >= 1280 * _MIB


@pytest.mark.usefixtures("stated_sizes")
def test_a_cuda_build_keeps_the_weights_plus_allowance_charge_for_a_large_vocabulary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=L4))
    folder = _sparse_encoder_folder(tmp_path, 1060.7, 250_002, "bfloat16")
    weights = int(1060.7 * _MIB)
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        weights + ENCODER_RUNTIME_BYTES
    )


@pytest.mark.parametrize("gpu", [L4, A100], ids=["sm89", "sm80"])
@pytest.mark.parametrize(
    ("data_bytes", "config", "halved"),
    [
        ({"F32": 3_000}, {}, 1_500),
        ({"F32": 3_000}, {"torch_dtype": "float16"}, 1_500),
        ({"F16": 3_000}, {"torch_dtype": "float32"}, 0),
        ({"F32": 2_000, "F16": 1_000}, {}, 1_000),
    ],
    ids=["f32-no-dtype", "f32-float16-config", "f16-float32-config", "mixed"],
)
def test_a_newer_cuda_build_holds_the_float32_bytes_of_the_header_as_float16(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gpu: tuple[str, tuple[int, int]],
    data_bytes: dict[str, int],
    config: dict[str, Any],
    halved: int,
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=gpu))
    folder, size = _mixed_encoder_folder(tmp_path, data_bytes, config)
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        size - halved + ENCODER_RUNTIME_BYTES
    )


def test_a_cuda_build_leaves_bfloat16_weights_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=L4))
    folder = _encoder_folder(tmp_path, 4_000, "bfloat16")
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        4_000 + ENCODER_RUNTIME_BYTES
    )


def _mixed_encoder_folder(
    root: Path, data_bytes: dict[str, int], config: dict[str, Any] | None
) -> tuple[Path, int]:
    """A folder whose one weight file holds a tensor per dtype; also the file's size."""
    folder = root / "mixed-encoder"
    folder.mkdir()
    blob = safetensors_mixed_bytes(data_bytes)
    (folder / "model.safetensors").write_bytes(blob)
    (folder / "config.json").write_text(json.dumps(config or {}), encoding="utf-8")
    return folder, len(blob)


@pytest.mark.parametrize(
    ("data_bytes", "config", "doubled"),
    [
        ({"F16": 3_000}, {}, 3_000),
        ({"BF16": 3_000}, {}, 3_000),
        ({"F32": 3_000}, {}, 0),
        ({"F16": 3_000}, {"torch_dtype": "float32"}, 3_000),
        ({"BF16": 3_000}, {"dtype": "float32"}, 3_000),
        ({"F32": 3_000}, {"torch_dtype": "float16"}, 0),
        ({"F16": 1_000, "BF16": 500, "F32": 2_000, "I64": 100}, {}, 1_500),
        ({"F16": 1_000, "F32": 2_000}, {"torch_dtype": "float32"}, 1_000),
    ],
    ids=[
        "f16-no-dtype",
        "bf16-no-dtype",
        "f32-no-dtype",
        "f16-under-a-float32-config",
        "bf16-under-a-float32-config",
        "f32-under-a-float16-config",
        "mixed-no-dtype",
        "mixed-float32-config",
    ],
)
def test_a_turing_build_doubles_the_half_precision_bytes_in_the_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    data_bytes: dict[str, int],
    config: dict[str, Any],
    doubled: int,
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=T4))
    folder, size = _mixed_encoder_folder(tmp_path, data_bytes, config)
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        size + doubled + ENCODER_RUNTIME_BYTES
    )


@pytest.mark.parametrize("gpu", [L4, A100], ids=["sm89", "sm80"])
def test_a_newer_cuda_build_keeps_the_stored_size_of_a_half_precision_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gpu: tuple[str, tuple[int, int]]
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=gpu))
    folder, size = _mixed_encoder_folder(tmp_path, {"F16": 3_000, "BF16": 500}, {})
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        size + ENCODER_RUNTIME_BYTES
    )


@pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
@pytest.mark.parametrize(
    ("gpu", "flags"),
    [(T4, ["float32"]), (("GeForce RTX 2080", (7, 5)), ["float32"]), (A100, []), (None, [])],
    ids=["t4", "rtx2080", "a100", "no-gpu"],
)
def test_the_planner_launch_for_a_role_carries_the_dtype_of_the_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: WorkerRole, gpu, flags: list[str]
) -> None:
    from lilbee.providers.fleet import planning

    caps = _caps(gpu=gpu) if gpu else _caps()
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: caps)
    monkeypatch.setattr(
        type(TEI), "binary", lambda _self: Path("/engines/bin/text-embeddings-router")
    )
    folder = _encoder_folder(tmp_path, 64, None)
    argv = planning.build_single_role_launch(role, folder, engine=TEI).argv
    assert [argv[i + 1] for i, word in enumerate(argv[:-1]) if word == "--dtype"] == flags


def _small_card_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, refuse_oversize: bool
) -> Placement:
    """The planner's answer for a float32 embedder and reranker on one 4 GiB Turing card."""
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=T4))
    folder = _sparse_encoder_folder(tmp_path, 2300, 250_002, "float32")
    charge = TEI.estimate_memory(_encoder_request(folder)).device_bytes
    models = [
        ModelPlacementInput(role, charge, kind=MemoryKind.FIXED_RESERVE)
        for role in (WorkerRole.EMBED, WorkerRole.RERANK)
    ]

    def no_split(role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
        raise AssertionError("a fixed reserve is never split")

    return plan_placement(
        models, [(0, 4096 * _MIB)], estimate_peak=no_split, refuse_oversize=refuse_oversize
    )


@pytest.mark.usefixtures("stated_sizes")
def test_two_float32_encoders_that_do_not_share_a_4_gib_card_load_one_tight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = _small_card_roles(monkeypatch, tmp_path, refuse_oversize=False)
    assert {inst.role for inst in plan.instances} == {WorkerRole.EMBED, WorkerRole.RERANK}
    assert len(plan.tight_roles) == 1
    assert plan.unplaceable_roles == ()


@pytest.mark.usefixtures("stated_sizes")
def test_two_float32_encoders_beyond_a_4_gib_card_leave_one_unplaced_when_oversize_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = _small_card_roles(monkeypatch, tmp_path, refuse_oversize=True)
    assert len(plan.instances) == 1
    assert len(plan.unplaceable_roles) == 1
    assert plan.instances[0].role not in plan.unplaceable_roles


def test_a_metal_build_is_charged_by_vocabulary_like_a_cpu_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps("Darwin", "arm64"))
    folder = _encoder_folder(tmp_path, 4_000, "float32")
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        4_000 * 5 // 4 + 1280 * _MIB
    )


def test_a_host_with_no_build_is_charged_by_vocabulary_like_a_cpu_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps("Windows"))
    folder = _encoder_folder(tmp_path, 4_000, "float32")
    assert TEI.estimate_memory(_encoder_request(folder)).device_bytes == (
        4_000 * 5 // 4 + 1280 * _MIB
    )


def test_a_longer_context_leaves_the_charge_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    folder = _encoder_folder(tmp_path, 4_000, "float32")
    assert TEI.estimate_memory(_encoder_request(folder, 512)) == TEI.estimate_memory(
        _encoder_request(folder, 32768)
    )


def test_a_model_with_no_weights_is_the_unknown_size_reserve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    estimate = TEI.estimate_memory(_encoder_request(tmp_path / "absent"))
    assert estimate.device_bytes == UNKNOWN_SIZE_RESERVE_BYTES
    assert estimate.size_known is False


def test_launch_raises_the_server_batch_limit_to_the_client_batch(tmp_path: Path) -> None:
    request = LaunchRequest(
        binary=tmp_path / "router",
        model_path=tmp_path / "model",
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=512,
    )
    argv = TEI.launch_argv(TEI.role_spec(WorkerRole.EMBED, None, None), request)
    assert argv == [
        str(tmp_path / "router"),
        "--model-id",
        str(tmp_path / "model"),
        "--hostname",
        "127.0.0.1",
        "--max-client-batch-size",
        "64",
        "--max-batch-tokens",
        "640",
        "--tokenization-workers",
        "2",
    ]


def test_pin_names_the_picked_build_and_refuses_a_lock_with_no_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=L4))
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    pin = TEI.pin_record()
    assert (pin["engine"], pin["version"], pin["backend"]) == ("tei", "1.9.4", "cuda-sm89")
    assert pin["licenses"] == ["Apache-2.0"]
    with pytest.raises(choice.NoPublishedPackError, match=r"No pack of tei is published yet\."):
        choice.pin_lock(pin)


def test_binary_with_no_published_pack_says_there_is_nothing_to_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {})
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    with pytest.raises(
        ProviderError,
        match=r"Engine tei is not installed\. No pack of tei is published yet, so there is "
        r"nothing to install\.",
    ) as error:
        TEI.binary()
    assert error.value.kind is ProviderErrorKind.NOT_FOUND


def test_binary_comes_from_the_store_record_of_the_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps())
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    with pytest.raises(ProviderError, match="Run lilbee engine install tei"):
        TEI.binary()
    wheel = tmp_path / "tei.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        tool = zipfile.ZipInfo("tei_fx/bin/text-embeddings-router")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("tei_fx-1.dist-info/METADATA", "License-Expression: Apache-2.0\n")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = PinLock("tei", "1.9.4", "cpu", {}, ("Apache-2.0",), digest, "text-embeddings-router")
    EngineStore(tmp_path / "engines").install_from_wheel(lock, wheel)
    want = tmp_path / "engines" / "tei" / "1.9.4-cpu" / "tei_fx" / "bin"
    assert TEI.binary() == want / "text-embeddings-router"


def test_other_probes_of_the_adapter_answer_without_a_server(tmp_path: Path) -> None:
    probe = TEI.discover_devices(tmp_path / "router")
    assert probe.devices == []
    assert TEI.identity(tmp_path / "router") == "tei@1.9.4"
    assert TEI.name is EngineName.TEI
    assert TEI.supports_architecture("bert")
    assert TEI.model_meta(tmp_path) is None
    assert TEI.assert_loadable("org/repo", "file", None) is None
    assert TEI.runtime_env() == {}
    assert not TEI.check_readback(tmp_path)
    assert not TEI.supports_fit_target(tmp_path)
    assert not TEI.supports_shared_kv(tmp_path)
    assert TEI.swap_extras(None) == {"checkEndpoint": "/health"}  # type: ignore[arg-type]


def test_profile_has_no_tokenize_route_and_names_the_rerank_route() -> None:
    profile = TEI.request_profile(WorkerRole.RERANK)
    assert (profile.tokenize_path, profile.detokenize_path) == (None, None)
    assert (profile.rerank_path, profile.batch_limit) == ("/rerank", 64)
    assert profile.embed_fields == {}
    assert profile.batch_overflow is None
    assert profile.batch_count_refusal is not None
    assert profile.batch_count_refusal.search("batch size 40 > maximum allowed batch size 32")
    assert profile.context_overflow.search("`inputs` must have less than 512 tokens. Given: 600")
    assert profile.context_overflow.search(
        "`inputs` must have less than 2048 characters. Given: 3000"
    )
    assert not profile.context_overflow.search("`inputs` cannot be empty")
    assert (profile.rerank_fields, profile.context_overflow_status) == ({"truncate": False}, 422)


def _recorded_rerank(name: str) -> tuple[dict, str]:
    recording = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    exchange = next(e for e in recording["exchanges"] if e["path"] == "/rerank")
    return exchange["request"], exchange["body"]


@pytest.mark.parametrize("recording", ["tei-cpu", "tei-cuda"])
def test_recorded_rerank_answer_gives_scores_in_candidate_order(recording: str) -> None:
    request, body = _recorded_rerank(recording)
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    client = LlamaServerClient(
        "http://gpu0",
        request["model"],
        http=http,
        profile=TEI.request_profile(WorkerRole.RERANK),
    )
    scores = client.rerank(request["query"], request["texts"])
    by_index = {item["index"]: item["score"] for item in json.loads(body)}
    assert scores == [by_index[0], by_index[1], by_index[2]]
    assert max(range(3), key=scores.__getitem__) == 1
    assert [r.url.path for r in seen] == [f"/upstream/{request['model']}/rerank"]
    assert json.loads(seen[0].content) == {
        "query": request["query"],
        "texts": request["texts"],
        "truncate": False,
    }


def _base64_vectors(vectors: list[list[float]]) -> dict[str, object]:
    return {
        "data": [
            {
                "embedding": base64.b64encode(np.asarray(vec, dtype="<f4").tobytes()).decode(),
                "index": index,
            }
            for index, vec in enumerate(vectors)
        ]
    }


def _tei_client(handler, role: WorkerRole = WorkerRole.RERANK, **changes) -> LlamaServerClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://gpu0")
    profile = dataclasses.replace(TEI.request_profile(role), **changes)
    return LlamaServerClient("http://gpu0", "m", http=http, profile=profile)


def _refuse_past(
    limit: int, seen: list[int], req: httpx.Request, key: str
) -> httpx.Response | None:
    count = len(json.loads(req.content)[key])
    seen.append(count)
    if count > limit:
        return httpx.Response(
            422, json={"error": f"batch size {count} > maximum allowed batch size {limit}"}
        )
    return None


def test_two_hundred_candidates_go_out_in_requests_the_managed_server_accepts() -> None:
    sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        refused = _refuse_past(64, sizes, req, "texts")
        if refused is not None:
            return refused
        texts = json.loads(req.content)["texts"]
        scored = [{"index": i, "score": float(text)} for i, text in enumerate(texts)]
        return httpx.Response(200, json=scored[::-1])

    client = _tei_client(handler)
    candidates = [str(i) for i in range(200)]
    assert client.rerank("q", candidates) == [float(i) for i in range(200)]
    assert sizes == [64, 64, 64, 8]


def test_two_hundred_texts_embed_in_requests_the_managed_server_accepts() -> None:
    sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        refused = _refuse_past(64, sizes, req, "input")
        if refused is not None:
            return refused
        texts = json.loads(req.content)["input"]
        vectors = [[float(text)] for text in texts]
        return httpx.Response(200, json=_base64_vectors(vectors))

    client = _tei_client(handler, WorkerRole.EMBED, input_limit_path=None)
    out = client.embed([str(i) for i in range(200)])
    assert [float(vec[0]) for vec in out] == [float(i) for i in range(200)]
    assert sizes == [64, 64, 64, 8]


def test_a_batch_size_refusal_is_a_bad_request_sent_once_not_a_context_overflow() -> None:
    sizes: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        refused = _refuse_past(32, sizes, req, "input")
        assert refused is not None
        return refused

    client = _tei_client(handler, WorkerRole.EMBED, batch_limit=None, input_limit_path=None)
    with pytest.raises(ProviderError, match=r"too many inputs\. Send fewer inputs") as raised:
        client.embed(["x"] * 33)
    assert raised.value.kind is ProviderErrorKind.BAD_REQUEST
    assert sizes == [33]


def test_the_launch_argv_carries_the_figure_the_profile_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tei_module, "_PROFILE", dataclasses.replace(tei_module._PROFILE, batch_limit=17)
    )
    request = LaunchRequest(
        binary=tmp_path / "router",
        model_path=tmp_path / "model",
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=512,
    )
    argv = TEI.launch_argv(TEI.role_spec(WorkerRole.EMBED, None, None), request)
    assert argv[argv.index("--max-client-batch-size") + 1] == "17"


def _batch_tokens(tmp_path: Path, ctx: int, target: RoleTarget = RoleTarget.GPU) -> int:
    request = LaunchRequest(
        binary=tmp_path / "router",
        model_path=tmp_path / "model",
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=ctx,
        target=target,
    )
    argv = TEI.launch_argv(TEI.role_spec(WorkerRole.EMBED, None, None), request)
    return int(argv[argv.index("--max-batch-tokens") + 1])


@pytest.mark.parametrize("target", [RoleTarget.CPU, RoleTarget.GPU])
def test_the_default_chunk_size_bounds_the_warm_up_batch(
    tmp_path: Path, target: RoleTarget
) -> None:
    """The default chunk size of 512 gives 512 chunk tokens plus 128 for a rerank query."""
    assert cfg.chunk_size == 512
    assert _batch_tokens(tmp_path, EMBED_FALLBACK_CTX, target) == 640


def test_a_larger_chunk_size_raises_the_warm_up_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cfg, "chunk_size", 1024)
    assert _batch_tokens(tmp_path, EMBED_FALLBACK_CTX) == 1152


@pytest.mark.parametrize("ctx", [512, EMBED_FALLBACK_CTX, 8192, 32768])
def test_the_warm_up_batch_does_not_follow_the_context_per_slot(tmp_path: Path, ctx: int) -> None:
    assert _batch_tokens(tmp_path, ctx) == _batch_tokens(tmp_path, 512)


def test_the_launch_argv_keeps_the_default_truncation_for_long_inputs(tmp_path: Path) -> None:
    """With truncation off, TEI refuses to start when the model limit exceeds the batch."""
    request = LaunchRequest(
        binary=tmp_path / "router",
        model_path=tmp_path / "model",
        devices=(),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=512,
    )
    argv = TEI.launch_argv(TEI.role_spec(WorkerRole.EMBED, None, None), request)
    assert "--auto-truncate" not in argv
    assert not any(a.startswith("--auto-truncate=") for a in argv)


def _dtype_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu, role: WorkerRole, target: RoleTarget
) -> list[str]:
    """The values that follow each `--dtype` in the launch command."""
    monkeypatch.setattr(
        "lilbee.catalog.runnable.session_host", lambda: _caps(gpu=gpu) if gpu else _caps()
    )
    request = LaunchRequest(
        binary=tmp_path / "router",
        model_path=tmp_path / "model",
        devices=(0,),
        n_gpu_layers=0,
        slots=1,
        ctx_per_slot=512,
        target=target,
    )
    argv = TEI.launch_argv(TEI.role_spec(role, None, None), request)
    return [argv[i + 1] for i, word in enumerate(argv[:-1]) if word == "--dtype"]


@pytest.mark.parametrize("role", [WorkerRole.EMBED, WorkerRole.RERANK])
def test_a_turing_build_launches_with_float32_for_embed_and_rerank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: WorkerRole
) -> None:
    assert _dtype_flags(tmp_path, monkeypatch, T4, role, RoleTarget.GPU) == ["float32"]


@pytest.mark.parametrize("compute", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)])
def test_a_build_for_compute_8_or_newer_launches_with_no_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, compute: tuple[int, int]
) -> None:
    flags = _dtype_flags(tmp_path, monkeypatch, ("GPU", compute), WorkerRole.RERANK, RoleTarget.GPU)
    assert flags == []


@pytest.mark.parametrize(
    ("gpu", "target"),
    [(None, RoleTarget.CPU), (GTX, RoleTarget.GPU), (T4, RoleTarget.CPU)],
    ids=["no-gpu", "pascal-cpu-build", "turing-host-cpu-target"],
)
def test_a_cpu_build_and_a_cpu_target_launch_with_no_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gpu, target: RoleTarget
) -> None:
    assert _dtype_flags(tmp_path, monkeypatch, gpu, WorkerRole.RERANK, target) == []


def test_an_unreadable_platform_stands_and_is_never_auto_selected() -> None:
    gate = TEI.host_gate(_caps("", ""))
    assert (gate.ok, gate.known, gate.auto_selectable) == (True, False, False)


def test_a_cuda_build_runs_a_role_on_the_card_or_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lilbee.catalog.runnable.session_host", lambda: _caps(gpu=("GPU", (8, 6))))
    assert TEI.device_targets(Path("/b/router")) == {RoleTarget.GPU, RoleTarget.CPU}


def test_a_cpu_role_hides_every_card_from_the_cuda_build() -> None:
    assert TEI.target_env(RoleTarget.CPU, 1) == {"CUDA_VISIBLE_DEVICES": ""}
    assert TEI.target_env(RoleTarget.GPU, 1) == {}


def test_a_cpu_role_charges_the_fixed_reserve_to_system_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lilbee.providers.fleet import planning

    model = tmp_path / "embed.bin"
    model.write_bytes(b"\0" * 4096)
    monkeypatch.setattr(planning.engine_params, "resolve_model_path", lambda _ref: model)
    sized = planning._estimate_cpu_role(
        WorkerRole.EMBED, "org/embed", engine=TEI, skipped_not_installed={}
    )
    assert sized is not None
    assert sized.est_vram_bytes == 0
    assert sized.est_ram_bytes >= 2 * 1024**3


_OVER_LIMIT = {"error": "Input validation error: `inputs` must have less than 24 tokens. Given: 31"}


def _rerank_exchange(over_limit: bool) -> tuple[list[dict], LlamaServerClient]:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        if over_limit and body.get("truncate") is False:
            return httpx.Response(422, json=_OVER_LIMIT)
        return httpx.Response(200, json=[{"index": 0, "score": 0.25}])

    return bodies, _tei_client(handler)


def test_rerank_that_fits_sends_one_request_with_truncation_off_and_no_warning(caplog) -> None:
    bodies, client = _rerank_exchange(over_limit=False)
    with caplog.at_level("WARNING"):
        assert client.rerank("q", ["a"]) == [0.25]
    assert bodies == [{"query": "q", "texts": ["a"], "truncate": False}]
    assert caplog.records == []


def test_rerank_over_the_limit_warns_once_and_is_sent_again_for_the_server_to_truncate(
    caplog,
) -> None:
    bodies, client = _rerank_exchange(over_limit=True)
    with caplog.at_level("WARNING"):
        assert client.rerank("q", ["a"]) == [0.25]
    assert bodies == [
        {"query": "q", "texts": ["a"], "truncate": False},
        {"query": "q", "texts": ["a"]},
    ]
    [record] = caplog.records
    assert record.getMessage().startswith("Rerank query and candidate exceed the reranker's token")
    assert "the server truncates the input" in record.getMessage()
    assert "must have less than 24 tokens. Given: 31" in record.getMessage()


def test_rerank_over_the_character_limit_is_sent_again_for_the_server_to_truncate() -> None:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        if body.get("truncate") is False:
            error = "`inputs` must have less than 2048 characters. Given: 3000"
            return httpx.Response(422, json={"error": error})
        return httpx.Response(200, json=[{"index": 0, "score": 0.5}])

    assert _tei_client(handler).rerank("q", ["a"]) == [0.5]
    assert len(bodies) == 2


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, "`inputs` cannot be empty"),
        (422, "`texts` must be a list"),
        (400, _OVER_LIMIT["error"]),
    ],
    ids=["empty-input", "other-validation-error", "overflow-body-wrong-status"],
)
def test_rerank_answer_that_is_not_the_over_limit_answer_is_an_error_sent_once(
    status: int, error: str, caplog
) -> None:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(status, json={"error": error})

    with caplog.at_level("WARNING"), pytest.raises(ProviderError) as raised:
        _tei_client(handler).rerank("q", ["a"])
    assert raised.value.kind is not ProviderErrorKind.CONTEXT_OVERFLOW
    assert len(bodies) == 1
    assert caplog.records == []


def test_rerank_gateway_retry_after_an_overflow_still_warns_once(monkeypatch, caplog) -> None:
    monkeypatch.setattr("lilbee.providers.fleet.client.time.sleep", lambda _s: None)
    resends = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal resends
        if json.loads(req.content).get("truncate") is False:
            return httpx.Response(422, json=_OVER_LIMIT)
        resends += 1
        if resends == 1:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json=[{"index": 0, "score": 0.75}])

    with caplog.at_level("WARNING"):
        assert _tei_client(handler).rerank("q", ["a"]) == [0.75]
    assert resends == 2
    assert len(caplog.records) == 1


def _long_candidates(count: int, long_at: set[int]) -> list[str]:
    return [("long" if i in long_at else "short") + str(i) for i in range(count)]


def _batched_rerank_exchange() -> tuple[list[dict], LlamaServerClient]:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        over = any(text.startswith("long") for text in body["texts"])
        if over and body.get("truncate") is False:
            return httpx.Response(422, json=_OVER_LIMIT)
        scores = [{"index": i, "score": 0.5} for i in range(len(body["texts"]))]
        return httpx.Response(200, json=scores)

    return bodies, _tei_client(handler)


@pytest.mark.parametrize(
    ("count", "long_at", "overflowing"),
    [(200, set(range(200)), 4), (65, {0, 64}, 2), (200, {70, 130}, 2), (200, set(), 0)],
    ids=["every-batch", "both-batches", "some-batches", "no-batch"],
)
def test_rerank_in_batches_warns_once_for_the_whole_call(
    count: int, long_at: set[int], overflowing: int, caplog
) -> None:
    bodies, client = _batched_rerank_exchange()
    with caplog.at_level("WARNING"):
        scores = client.rerank("q", _long_candidates(count, long_at))
    assert scores == [0.5] * count
    batches = -(-count // 64)
    assert len(bodies) == batches + overflowing
    if overflowing == 0:
        assert caplog.records == []
        return
    [record] = caplog.records
    assert f"in {overflowing} of {batches} batches" in record.getMessage()
    assert "must have less than 24 tokens. Given: 31" in record.getMessage()


def test_rerank_overflow_from_an_engine_with_no_extra_fields_is_not_sent_again() -> None:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(422, json=_OVER_LIMIT)

    client = _tei_client(handler, rerank_fields={})
    with pytest.raises(ProviderError) as raised:
        client.rerank("q", ["a"])
    assert raised.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW
    assert bodies == [{"query": "q", "texts": ["a"]}]


def test_the_422_over_limit_answer_is_a_context_overflow() -> None:
    client = _tei_client(lambda req: httpx.Response(422, json=_OVER_LIMIT), WorkerRole.EMBED)
    with pytest.raises(ProviderError, match="context window") as raised:
        client._raise_for_status(client._http.post("/v1/embeddings", json={}))
    assert raised.value.kind is ProviderErrorKind.CONTEXT_OVERFLOW


def test_the_400_status_is_not_the_over_limit_answer_of_this_engine() -> None:
    client = _tei_client(lambda req: httpx.Response(400, json=_OVER_LIMIT), WorkerRole.EMBED)
    with pytest.raises(ProviderError) as raised:
        client._raise_for_status(client._http.post("/v1/embeddings", json={}))
    assert raised.value.kind is not ProviderErrorKind.CONTEXT_OVERFLOW


_ROUTER_LIMIT = 8


class _Embedder:
    """A fake router as measured: it embeds any input, cuts it silently, counts a word a token.

    ``special`` tokens flank each input, flagged as the router flags them. With ``by_char``
    a character is a token.
    """

    def __init__(self, limit: int = _ROUTER_LIMIT, special: int = 0, by_char: bool = False) -> None:
        self.limit = limit
        self.special = special
        self.by_char = by_char
        self.paths: list[str] = []
        self.timeouts: dict[str, float | None] = {}
        self.tokenized: list[list[str]] = []

    def _tokens(self, text: str) -> list[dict[str, Any]]:
        pieces = list(text) if self.by_char else text.split()
        tokens = [{"id": i, "special": False} for i, _ in enumerate(pieces)]
        specials = [{"id": 0, "special": True} for _ in range(self.special)]
        return specials[:1] + tokens + specials[1:]

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.paths.append(req.url.path)
        self.timeouts[req.url.path] = req.extensions["timeout"]["read"]
        body = json.loads(req.content) if req.content else {}
        if req.url.path.endswith("/info"):
            return httpx.Response(200, json={"max_input_length": self.limit})
        if req.url.path.endswith("/tokenize"):
            self.tokenized.append(body["inputs"])
            return httpx.Response(200, json=[self._tokens(t) for t in body["inputs"]])
        vector = base64.b64encode(np.array([0.5, 0.5], dtype="<f4").tobytes()).decode()
        data = [{"index": i, "embedding": vector} for i, _ in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data})


def _words(count: int) -> str:
    return " ".join(["w"] * count)


def _past_limit(client: LlamaServerClient, texts: list[str]) -> int:
    """How many of *texts* the check after the embed tallies past the router's limit."""
    limit, path = client._input_limit(), client._profile.input_count_path
    assert limit is not None and path is not None
    return sum(client._counts_past_limit(texts, limit, path))


def test_embed_inputs_past_the_router_limit_give_one_warning_for_the_call(caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        vectors = client.embed([_words(30), _words(3), _words(20)])
    assert len(vectors) == 3
    [record] = caplog.records
    assert "2 of 3" in record.getMessage()
    assert "8-token limit" in record.getMessage()
    assert "the cut text is not in the vectors" in record.getMessage()


def test_embed_inputs_within_the_router_limit_give_no_warning(caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(3), _words(4)])
    assert caplog.records == []
    assert router.tokenized == [[_words(3), _words(4)]]


def test_embed_counts_every_input_in_one_request() -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    short, long = "a b", _words(30)
    client.embed([short, long])
    assert router.tokenized == [[short, long]]


class _ByteLevelEmbedder(_Embedder):
    """A router whose tokenizer makes one token per UTF-8 byte, so a character can be many."""

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/tokenize"):
            body = json.loads(req.content)
            return httpx.Response(
                200, json=[[{"id": b} for b in t.encode("utf-8")] for t in body["inputs"]]
            )
        return super().__call__(req)


def test_embed_warns_for_a_text_with_few_characters_and_many_tokens(caplog) -> None:
    """Six characters are twelve bytes: past the limit of eight, though not in characters."""
    client = _tei_client(_ByteLevelEmbedder(), WorkerRole.EMBED)
    text = "é" * 6
    assert len(text) < _ROUTER_LIMIT
    with caplog.at_level("WARNING"):
        client.embed([text])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


def test_embed_asks_the_router_for_its_limit_once() -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1


def test_embeds_in_one_scope_give_one_warning_that_adds_their_counts(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"), truncation_scope():
        client.embed([_words(30), _words(2)])
        client.embed([_words(40)])
    [record] = caplog.records
    assert record.getMessage().startswith("2 of 3")
    assert "8-token limit" in record.getMessage()


def test_the_count_of_a_call_with_several_batches_adds_the_cut_inputs_of_each(caplog) -> None:
    """The cut input is in the first batch and the last batch holds none."""
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)] + [_words(2)] * 70)
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 71")


@pytest.mark.parametrize("special", [0, 1, 2, 3])
def test_an_input_the_router_cuts_only_with_its_special_tokens_gives_a_warning(
    special, caplog
) -> None:
    """The router adds *special* tokens, so one character a token is cut at limit - special + 1."""
    router = _Embedder(special=special, by_char=True)
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed(["x" * (_ROUTER_LIMIT - special + 1)])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


@pytest.mark.parametrize("special", [0, 1, 2, 3])
def test_an_input_that_fits_with_the_special_tokens_gives_no_warning(special, caplog) -> None:
    router = _Embedder(special=special, by_char=True)
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed(["x" * (_ROUTER_LIMIT - special)])
    assert router.tokenized == [["x" * (_ROUTER_LIMIT - special)]]
    assert caplog.records == []


@pytest.mark.parametrize("route", ["/info", "/tokenize"])
def test_a_side_route_request_asks_for_an_answer_within_five_seconds(route) -> None:
    router = _Embedder(special=2, by_char=True)
    http = httpx.Client(transport=httpx.MockTransport(router), base_url="http://gpu0", timeout=60.0)
    client = LlamaServerClient(
        "http://gpu0", "m", http=http, profile=TEI.request_profile(WorkerRole.EMBED)
    )
    client.embed(["x" * 7])
    assert [t for path, t in router.timeouts.items() if path.endswith(route)] == [5.0]


def test_a_count_for_chunk_sizing_waits_as_long_as_the_client_does() -> None:
    router = _Embedder(by_char=True)
    http = httpx.Client(transport=httpx.MockTransport(router), base_url="http://gpu0", timeout=60.0)
    client = LlamaServerClient(
        "http://gpu0", "m", http=http, profile=TEI.request_profile(WorkerRole.EMBED)
    )
    assert client.count_text_tokens("abc") == 3
    assert [t for path, t in router.timeouts.items() if path.endswith("/tokenize")] == [60.0]


class _RecoveringRouter(_Embedder):
    """A router whose /info answers with the status in ``info_status`` until the test changes it."""

    def __init__(self, info_status: int) -> None:
        super().__init__()
        self.info_status = info_status

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/info") and self.info_status != 200:
            self.paths.append(req.url.path)
            return httpx.Response(self.info_status, json={"error": "down"})
        return super().__call__(req)


def test_a_busy_router_pauses_the_check_for_a_cool_down(caplog) -> None:
    router = _RecoveringRouter(503)
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    router.info_status = 200
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1
    assert caplog.records == []


def test_the_check_runs_again_when_the_cool_down_is_over(monkeypatch, caplog) -> None:
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _RecoveringRouter(503)
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    router.info_status = 200
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


class _FrozenClock:
    """Stands in for the ``time`` module of the client with a clock that never moves."""

    def __init__(self, now: float) -> None:
        self.now = now
        self.sleep = time.sleep

    def monotonic(self) -> float:
        return self.now


def test_a_client_checks_on_its_first_embed_whatever_the_clock_reads(monkeypatch, caplog) -> None:
    monkeypatch.setattr(fleet_client, "time", _FrozenClock(0.5))
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


def test_a_cool_down_of_no_time_ends_at_once(monkeypatch, caplog) -> None:
    monkeypatch.setattr(fleet_client, "time", _FrozenClock(100.0))
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _RecoveringRouter(503)
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    router.info_status = 200
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    [record] = caplog.records
    assert record.getMessage().startswith("1 of 1")


def test_a_count_route_answer_with_entries_that_are_not_lists_is_named_as_unexpected(
    caplog,
) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, json=[1]))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.getMessage() == (
        "Embed truncation check skipped: the router's /tokenize route gave an unexpected "
        "answer (the body is not a list of token lists: [1]). "
        "Embed inputs the router cuts are not reported."
    )


def test_a_router_that_cannot_be_reached_is_asked_again_after_the_cool_down(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _BrokenSideRoute("/tokenize", httpx.ConnectError("refused"))
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    router.answer = httpx.Response(200, json=[[{"id": 0, "special": False}] * 30])
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    assert [r.getMessage()[:6] for r in caplog.records] == ["1 of 1"]


def test_a_route_the_router_does_not_have_stops_the_check_for_the_client_after_a_cool_down(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _BrokenSideRoute("/info", httpx.Response(404, json={}))
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1


def test_an_input_of_exactly_the_limit_in_characters_is_counted_and_not_cut(caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed(["a" * _ROUTER_LIMIT])
    assert router.tokenized == [["a" * _ROUTER_LIMIT]]
    assert caplog.records == []


def test_an_input_of_exactly_the_limit_in_tokens_gives_no_warning(caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(_ROUTER_LIMIT)])
    assert router.tokenized == [[_words(_ROUTER_LIMIT)]]
    assert caplog.records == []


def test_two_scopes_on_one_client_each_give_one_warning(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        for _ in range(2):
            with truncation_scope():
                client.embed([_words(30)])
                client.embed([_words(40)])
    assert len(caplog.records) == 2


def test_embeds_outside_a_scope_each_warn(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
        client.embed([_words(40)])
    assert len(caplog.records) == 2


def test_a_scope_with_nothing_cut_logs_nothing(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"), truncation_scope():
        client.embed([_words(3)])
    assert caplog.records == []


def test_embed_of_a_dense_text_the_count_finds_within_the_limit_gives_no_warning(caplog) -> None:
    """Thirty-nine characters could be thirty-nine tokens; the router count says twenty."""
    router = _Embedder(limit=38)
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(20)])
    assert router.tokenized == [[_words(20)]]
    assert caplog.records == []


@pytest.mark.parametrize(
    "routes",
    [
        {"input_count_path": None, "input_limit_path": None},
        {"input_count_path": None},
    ],
    ids=["no-routes", "limit-route-only"],
)
def test_an_engine_with_no_count_route_gives_no_warning(routes, caplog) -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED, **routes)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    assert router.tokenized == []
    assert caplog.records == []
    assert all(path.endswith(("/info", "/embeddings")) for path in router.paths)


class _BrokenSideRoute(_Embedder):
    """A router whose embeddings work while one side route answers as the test sets."""

    def __init__(self, route: str, answer: httpx.Response | Exception) -> None:
        super().__init__()
        self.route = route
        self.answer = answer

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith(self.route):
            self.paths.append(req.url.path)
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer
        return super().__call__(req)


_BROKEN_ANSWERS = {
    "404": httpx.Response(404, json={"error": "no route"}),
    "500": httpx.Response(500, text="boom"),
    "503": httpx.Response(503, text="restarting"),
    "not-json": httpx.Response(200, text="<html>"),
    "connection": httpx.ConnectError("refused"),
}
_BROKEN_INFO = {
    **_BROKEN_ANSWERS,
    "no-limit-key": httpx.Response(200, json={"model_id": "x"}),
    "limit-not-a-number": httpx.Response(200, json={"max_input_length": "many"}),
    "limit-null": httpx.Response(200, json={"max_input_length": None}),
    "body-a-list": httpx.Response(200, json=[1, 2]),
}
_BROKEN_TOKENIZE = {
    **_BROKEN_ANSWERS,
    "body-an-object": httpx.Response(200, json={"tokens": 1}),
    "entries-not-lists": httpx.Response(200, json=[1]),
}


_DEAD_ROUTER_KINDS = {"503", "connection"}


def _skipped_level(kind: str) -> str:
    """A router that is down costs a debug line; an answer the check cannot use is a warning."""
    return "DEBUG" if kind in _DEAD_ROUTER_KINDS else "WARNING"


@pytest.mark.parametrize("kind", list(_BROKEN_INFO))
def test_embed_returns_its_vectors_when_the_limit_route_fails(kind, caplog) -> None:
    router = _BrokenSideRoute("/info", _BROKEN_INFO[kind])
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        vectors = client.embed([_words(30), _words(3)])
    assert len(vectors) == 2
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == _skipped_level(kind)


@pytest.mark.parametrize("kind", list(_BROKEN_TOKENIZE))
def test_embed_returns_its_vectors_when_the_count_route_fails(kind, caplog) -> None:
    router = _BrokenSideRoute("/tokenize", _BROKEN_TOKENIZE[kind])
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        vectors = client.embed([_words(30), _words(3)])
    assert len(vectors) == 2
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == _skipped_level(kind)


def test_a_tokenize_body_of_an_empty_object_is_a_failed_check(caplog) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, json={}))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    assert sum("check skipped" in r.getMessage() for r in caplog.records) == 1


def test_an_unexpected_count_route_answer_is_a_warning_naming_the_route_and_the_body(
    caplog,
) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, text='{"tokens": 1}'))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert record.getMessage() == (
        "Embed truncation check skipped: the router's /tokenize route gave an unexpected "
        'answer (the body is not a list of token lists: {"tokens": 1}). '
        "Embed inputs the router cuts are not reported."
    )


def test_a_missing_limit_key_is_a_warning_naming_the_limit_route(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(200, json={"model_id": "x"}))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert record.getMessage() == (
        "Embed truncation check skipped: the router's /info route gave an unexpected "
        "answer ('max_input_length'). Embed inputs the router cuts are not reported."
    )


def test_a_router_that_is_down_costs_a_debug_line_and_no_warning(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(503, text="restarting"))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


def test_an_absent_route_is_one_warning_that_names_the_404(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(404, text="no route"))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert "/info route" in record.getMessage()
    assert "(HTTP 404): no route" in record.getMessage()


def test_a_dead_router_debug_line_carries_the_reason(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.ConnectError("refused"))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "DEBUG"
    assert record.getMessage().startswith("Embed truncation check skipped: ")
    assert "refused" in record.getMessage()


def test_an_absent_route_warning_quotes_at_most_200_characters_of_the_body(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(404, text="x" * 300))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert "(HTTP 404): " + "x" * 200 + ")" in record.getMessage()


def test_a_bad_token_list_warning_quotes_at_most_200_characters_of_the_body(caplog) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, json="y" * 300))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert '"' + "y" * 199 + ")" in record.getMessage()


def test_a_404_on_a_side_route_is_a_not_found_error_that_names_the_router() -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with pytest.raises(ProviderError) as raised:
        client._raise_for_side_route(httpx.Response(404, text="gone"))
    assert raised.value.kind is ProviderErrorKind.NOT_FOUND


def test_an_unexpected_answer_is_asked_again_after_the_cool_down_and_warns_once(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_RETRY_S", 0.0)
    router = _BrokenSideRoute("/info", httpx.Response(200, json={"model_id": "x"}))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
        client.embed([_words(30)])
        router.answer = httpx.Response(200, json={"max_input_length": _ROUTER_LIMIT})
        client.embed([_words(30)])
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert sum(p.endswith("/info") for p in router.paths) == 3
    assert len(warnings) == 2
    assert warnings[0].getMessage().startswith("Embed truncation check skipped")
    assert warnings[1].getMessage().startswith("1 of 1")


def test_an_unexpected_answer_is_not_asked_again_inside_the_cool_down() -> None:
    router = _BrokenSideRoute("/info", httpx.Response(200, json={"model_id": "x"}))
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1


def test_a_second_embed_after_a_failed_limit_read_logs_no_second_skip(caplog) -> None:
    router = _BrokenSideRoute("/info", httpx.Response(404, json={}))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        client.embed([_words(30)])
        client.embed([_words(30)])
    assert sum("check skipped" in r.getMessage() for r in caplog.records) == 1


def test_a_failed_limit_read_is_not_tried_again_by_the_same_client() -> None:
    router = _BrokenSideRoute("/info", httpx.Response(404, json={}))
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    client.embed([_words(30)])
    assert sum(p.endswith("/info") for p in router.paths) == 1


def test_a_later_embed_with_no_limit_keeps_the_limit_the_scope_has(caplog) -> None:
    with caplog.at_level("WARNING"), truncation_scope():
        note_truncated(2, 8, 512)
        note_truncated(0, 5, None)
    [record] = caplog.records
    assert "2 of 13" in record.getMessage()
    assert "512-token limit" in record.getMessage()


def test_a_cut_with_no_known_limit_never_prints_none(caplog) -> None:
    with caplog.at_level("WARNING"), truncation_scope():
        note_truncated(1, 4, None)
    [record] = caplog.records
    assert "None" not in record.getMessage()
    assert "1 of 4 embed inputs are past the embedder's limit and" in record.getMessage()


def test_the_warning_names_the_chunk_size_setting_and_a_larger_limit_model(caplog) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        client.embed([_words(30)])
    message = caplog.records[0].getMessage()
    assert "Lower chunk_size, or use an embedding model with a larger input limit." in message


_HANG_S = 3.0
_CHECK_BOUND_S = 1.5


@pytest.fixture
def hung_side_route(monkeypatch: pytest.MonkeyPatch):
    """A real threaded router whose /info or /tokenize answer waits ``_HANG_S``."""
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_TIMEOUT_S", 0.2)
    release = threading.Event()
    server = HTTPServer(host="127.0.0.1", port=0, threaded=True)
    server.start()
    paths: list[str] = []

    def serve(
        hung: str, hang_s: float = _HANG_S, token_cap: int | None = None
    ) -> LlamaServerClient:
        def answer(request: Request) -> Response:
            paths.append(request.path)
            if request.path.endswith(hung):
                release.wait(hang_s)
            if request.path.endswith("/info"):
                return Response(json.dumps({"max_input_length": _ROUTER_LIMIT}))
            inputs = json.loads(request.data)
            if request.path.endswith("/tokenize"):
                tokens = [
                    [{"id": i, "special": False} for i, _ in enumerate(t.split())]
                    for t in inputs["inputs"]
                ]
                return Response(json.dumps(tokens))
            return Response(json.dumps(_base64_vectors([[0.5, 0.5]] * len(inputs["input"]))))

        server.expect_request(re.compile(".*")).respond_with_handler(answer)
        profile = TEI.request_profile(WorkerRole.EMBED)
        base = f"http://127.0.0.1:{server.port}"
        return LlamaServerClient(base, "m", timeout=_HANG_S, profile=profile, token_cap=token_cap)

    yield serve, paths
    release.set()
    server.clear()
    server.stop()


def _timed_embed(client: LlamaServerClient) -> float:
    start = time.monotonic()
    vectors = client.embed([_words(30), _words(3)])
    assert len(vectors) == 2
    return time.monotonic() - start


@pytest.mark.parametrize("hung", ["/info", "/tokenize"])
def test_a_hung_side_route_never_holds_an_embed_back(hung, hung_side_route) -> None:
    serve, paths = hung_side_route
    client = serve(hung)
    first, second = _timed_embed(client), _timed_embed(client)
    assert first < _CHECK_BOUND_S, f"first embed took {first:.2f} s with {hung} hung"
    assert second < _CHECK_BOUND_S, f"second embed took {second:.2f} s with {hung} hung"
    assert sum(path.endswith(hung) for path in paths) == 1
    assert sum(path.endswith("/embeddings") for path in paths) == 2


def _run_together(calls: list[Callable[[], Any]]) -> list[Any]:
    """The result of each call, with every call started at the same moment on its own thread."""
    start = threading.Barrier(len(calls))
    results: list[Any] = [None] * len(calls)

    def run(i: int) -> None:
        start.wait()
        results[i] = calls[i]()

    workers = [threading.Thread(target=run, args=(i,)) for i in range(len(calls))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return results


@pytest.mark.parametrize("hung", ["/info", "/tokenize"])
def test_four_first_embeds_make_one_request_to_a_hung_route(hung, hung_side_route) -> None:
    serve, paths = hung_side_route
    client = serve(hung)
    seconds = _run_together([lambda: _timed_embed(client)] * 4)
    assert max(seconds) < _CHECK_BOUND_S, f"slowest embed took {max(seconds):.2f} s"
    assert sum(path.endswith(hung) for path in paths) == 1
    assert sum(path.endswith("/embeddings") for path in paths) == 4


def test_four_first_embeds_of_different_texts_make_one_request_to_a_hung_count_route(
    hung_side_route,
) -> None:
    serve, paths = hung_side_route
    client = serve("/tokenize")
    calls = [lambda i=i: client.embed([_words(30 + i)]) for i in range(4)]
    started = time.monotonic()
    _run_together(calls)
    total = time.monotonic() - started
    assert sum(path.endswith("/tokenize") for path in paths) == 1
    assert sum(path.endswith("/embeddings") for path in paths) == 4
    assert total < _CHECK_BOUND_S, f"four embeds took {total:.2f} s in all with /tokenize hung"


def test_a_thread_that_asks_as_a_failed_probe_ends_sends_no_second_request(
    monkeypatch, hung_side_route
) -> None:
    """The pause is set before the probe lets go, so a late thread waits on it, not on the route."""
    serve, paths = hung_side_route
    client = serve("/tokenize")
    pause = client._pause_truncation_check
    late: list[threading.Thread] = []
    embedded: list[int] = []

    def pause_after_a_late_ask(exc: Exception, route: str | None) -> None:
        thread = threading.Thread(target=lambda: embedded.append(len(client.embed([_words(31)]))))
        late.append(thread)
        thread.start()
        thread.join(0.5)
        pause(exc, route)

    monkeypatch.setattr(client, "_pause_truncation_check", pause_after_a_late_ask)
    assert len(client.embed([_words(30)])) == 1
    late[0].join()
    assert embedded == [1]
    assert sum(path.endswith("/tokenize") for path in paths) == 1


def test_a_failure_that_is_not_a_router_answer_reaches_the_caller_and_pauses_nothing(
    monkeypatch,
) -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)

    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("not a router failure")

    monkeypatch.setattr(client, "_count_route_tokens", broken)
    with pytest.raises(RuntimeError, match="not a router failure"):
        client.embed([_words(30)])
    assert client._check_resume_at == 0.0
    assert client._flights == {}


def test_once_the_count_route_has_answered_threads_count_without_waiting_on_each_other(
    monkeypatch, hung_side_route
) -> None:
    serve, paths = hung_side_route
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_TIMEOUT_S", 5.0)
    client = serve("/tokenize", hang_s=0.4)
    client.embed([_words(30)])
    paths.clear()
    started = time.monotonic()
    _run_together([lambda i=i: client.embed([_words(30 + i)]) for i in range(1, 3)])
    total = time.monotonic() - started
    assert sum(path.endswith("/tokenize") for path in paths) == 2
    assert total < 0.7, f"two embeds took {total:.2f} s in all; each count waits 0.4 s"


@pytest.mark.parametrize("slow", ["/info", "/tokenize"])
def test_embeds_that_wait_on_a_slow_route_use_its_one_answer(
    slow, monkeypatch, hung_side_route, caplog
) -> None:
    serve, paths = hung_side_route
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_TIMEOUT_S", 5.0)
    noted: list[int] = []
    monkeypatch.setattr(fleet_client, "note_truncated", lambda cut, *_rest: noted.append(cut))
    client = serve(slow, hang_s=0.4)
    with caplog.at_level("DEBUG"):
        _run_together([lambda: _timed_embed(client)] * 4)
    assert sum(path.endswith(slow) for path in paths) == 1
    assert not [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert noted == [1, 1, 1, 1]


def test_embeds_of_different_texts_each_count_their_own(monkeypatch, hung_side_route) -> None:
    serve, paths = hung_side_route
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_TIMEOUT_S", 5.0)
    client = serve("/tokenize", hang_s=0.4)
    long, short = [_words(30)], [_words(3)]
    cuts = _run_together(
        [
            lambda: _past_limit(client, long),
            lambda: _past_limit(client, short),
        ]
    )
    assert cuts == [1, 0]
    assert sum(path.endswith("/tokenize") for path in paths) == 2


_HOLD_S = 5.0
"""The longest a fake route waits for the threads a test arranges; it bounds a red run only."""


class _ProbeWatch(dict[tuple[Any, ...], Any]):
    """The client's table of requests in flight; it notes each thread that finds the probe sent.

    ``waiting`` is set once ``waiters`` threads wait on the first count of the route.
    """

    def __init__(self, waiters: int) -> None:
        super().__init__()
        self.waiters = waiters
        self.waiting = threading.Event()
        self.found: list[tuple[int, Any]] = []

    def get(self, key: tuple[Any, ...], default: Any = None) -> Any:
        flight = super().get(key, default)
        if flight is not None and key[0] == "first-probe":
            self.found.append((threading.get_ident(), flight))
            if len(self.found) == self.waiters:
                self.waiting.set()
        return flight


class _HeldCountRoute(_Embedder):
    """A router whose count route answers each request in order as the test sets.

    The first count request waits for ``hold``. ``failing`` maps a request's place, from 0,
    to the answer it gets; any other request is counted. With ``paired`` each request after
    the first waits until another one has arrived.
    """

    def __init__(
        self,
        hold: threading.Event | None = None,
        failing: dict[int, httpx.Response] | None = None,
        paired: bool = False,
    ) -> None:
        super().__init__()
        self.hold = hold
        self.failing = failing or {}
        self.pair = threading.Barrier(2) if paired else None
        self.counted = 0
        self._lock = threading.Lock()

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if not req.url.path.endswith("/tokenize"):
            return super().__call__(req)
        with self._lock:
            place = self.counted
            self.counted += 1
        if place == 0 and self.hold is not None:
            self.hold.wait(_HOLD_S)
        elif place > 0 and self.pair is not None:
            with contextlib.suppress(threading.BrokenBarrierError):
                self.pair.wait(_HOLD_S)
        if place in self.failing:
            return self.failing[place]
        return super().__call__(req)


def _skip_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "check skipped" in r.getMessage()]


def test_a_thread_waiting_on_a_failed_probe_finds_the_pause_set_when_it_wakes(
    monkeypatch,
) -> None:
    watch = _ProbeWatch(waiters=1)
    router = _HeldCountRoute(hold=watch.waiting, failing={0: httpx.Response(503, text="busy")})
    client = _tei_client(router, WorkerRole.EMBED)
    client._flights = watch
    noted = threading.Event()
    paused_at_note: dict[int, bool] = {}
    pause = client._pause_truncation_check

    def pause_once_a_woken_waiter_noted(exc: Exception, route: str | None) -> None:
        # A waiter that can wake before the pause is set gets to note its embed first.
        if any(flight.done() for _, flight in watch.found):
            noted.wait(_HOLD_S)
        pause(exc, route)

    def note(*_args: Any) -> None:
        ident = threading.get_ident()
        paused_at_note[ident] = time.monotonic() < client._check_resume_at
        if ident in {waiter for waiter, _ in watch.found}:
            noted.set()

    monkeypatch.setattr(client, "_pause_truncation_check", pause_once_a_woken_waiter_noted)
    monkeypatch.setattr(fleet_client, "note_truncated", note)
    _run_together([lambda: client.embed([_words(30)])] * 2)

    assert len(watch.found) == 1
    assert list(paused_at_note.values()) == [True, True]
    assert router.counted == 1


@pytest.mark.parametrize("words", [30, 31], ids=["same-texts", "other-texts"])
def test_a_count_that_fails_after_the_route_answered_pauses_the_check(words, caplog) -> None:
    router = _HeldCountRoute(failing={1: httpx.Response(500, text="boom")})
    client = _tei_client(router, WorkerRole.EMBED)
    client.embed([_words(30)])
    with caplog.at_level("WARNING"):
        client.embed([_words(words)])
        client.embed([_words(words)])
    assert router.counted == 2
    [record] = _skip_warnings(caplog)
    assert record.levelname == "WARNING"


def test_a_failed_count_of_other_texts_than_a_waited_probe_pauses_the_check(caplog) -> None:
    watch = _ProbeWatch(waiters=1)
    router = _HeldCountRoute(hold=watch.waiting, failing={1: httpx.Response(500, text="boom")})
    client = _tei_client(router, WorkerRole.EMBED)
    client._flights = watch
    with caplog.at_level("WARNING"):
        _run_together([lambda: client.embed([_words(30)]), lambda: client.embed([_words(31)])])
        client.embed([_words(32)])
    assert len(watch.found) == 1
    assert router.counted == 2
    [record] = _skip_warnings(caplog)
    assert record.levelname == "WARNING"


def test_threads_with_other_texts_than_a_waited_probe_each_get_their_own_count(
    monkeypatch,
) -> None:
    watch = _ProbeWatch(waiters=2)
    router = _HeldCountRoute(hold=watch.waiting, paired=True)
    client = _tei_client(router, WorkerRole.EMBED)
    client._flights = watch
    cuts: dict[int, int] = {}
    monkeypatch.setattr(
        fleet_client, "note_truncated", lambda cut, *_: cuts.__setitem__(threading.get_ident(), cut)
    )

    def cut_of(texts: list[str]) -> int:
        client.embed(texts)
        return cuts[threading.get_ident()]

    batches = [[_words(30)], [_words(31), _words(32)], [_words(33), _words(34), _words(35)]]
    assert _run_together([lambda b=b: cut_of(b) for b in batches]) == [1, 2, 3]
    assert len(watch.found) == 2
    assert router.counted == 3


_WIDE_CAP = 10


class _FailingCounts(_Embedder):
    """A router that holds each count and fails those *fails* picks, by order and inputs."""

    def __init__(
        self,
        failure: int | Exception | httpx.Response,
        fails: Callable[[int, list[str]], bool],
        hold_s: float = 0.4,
    ) -> None:
        super().__init__()
        self.failure = failure
        self.fails = fails
        self.hold_s = hold_s
        self.counted: list[list[str]] = []
        self.embedded: list[str] = []
        self.active = 0
        self.peak = 0
        self.first_sent = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        if req.url.path.endswith("/embeddings"):
            self.embedded.extend(body["input"])
        if not req.url.path.endswith("/tokenize"):
            return super().__call__(req)
        with self._lock:
            self.counted.append(body["inputs"])
            order = len(self.counted)
            self.active += 1
            self.peak = max(self.peak, self.active)
        self.first_sent.set()
        time.sleep(self.hold_s)
        with self._lock:
            self.active -= 1
        if not self.fails(order, body["inputs"]):
            return super().__call__(req)
        if isinstance(self.failure, Exception):
            raise self.failure
        if isinstance(self.failure, httpx.Response):
            return self.failure
        return httpx.Response(self.failure, json={"error": "not this one"})

    def client(self) -> LlamaServerClient:
        http = httpx.Client(transport=httpx.MockTransport(self), base_url="http://gpu0")
        profile = TEI.request_profile(WorkerRole.EMBED)
        return LlamaServerClient(
            "http://gpu0", "m", http=http, profile=profile, token_cap=_WIDE_CAP
        )


def _wide_within_limit(word: str) -> str:
    """Eight tokens, the router's limit, of a four-letter *word*: an estimate past the cap."""
    return " ".join([word] * 8)


_FOUR_WIDE_TEXTS = [_wide_within_limit(word) for word in ("east", "west", "nord", "sued")]


@pytest.mark.parametrize(
    "failure",
    [401, 403, 404, 409, 424, 500, 503, httpx.ReadTimeout("hung"), httpx.ConnectError("down")],
    ids=str,
)
def test_a_first_count_that_fails_for_the_route_takes_one_request_for_every_call(failure) -> None:
    router = _FailingCounts(failure, lambda *_: True)
    client = router.client()
    _run_together([lambda text=text: client.embed([text]) for text in _FOUR_WIDE_TEXTS])
    assert len(router.counted) == 1
    assert sorted(router.embedded) == sorted(_FOUR_WIDE_TEXTS)  # each input reaches it whole


def test_calls_of_one_text_share_one_count_after_the_first_is_refused() -> None:
    router = _FailingCounts(429, lambda order, _inputs: order == 1)
    client = router.client()
    _run_together([lambda: client.embed([_FOUR_WIDE_TEXTS[0]])] * 4)
    assert router.embedded == [_FOUR_WIDE_TEXTS[0]] * 4  # each input reaches the server whole
    assert len(router.counted) == 2
    assert client._count_route_answered  # the second count ends the probe for later ones


def test_threads_that_wait_for_a_probe_of_other_texts_each_get_their_own_counts(
    monkeypatch, hung_side_route
) -> None:
    serve, paths = hung_side_route
    monkeypatch.setattr(fleet_client, "_TRUNCATION_CHECK_TIMEOUT_S", 5.0)
    client = serve("/tokenize", hang_s=0.4)
    texts = [_words(30), _words(3), _words(31), _words(4)]
    cuts = _run_together([lambda text=text: _past_limit(client, [text]) for text in texts])
    assert cuts == [1, 0, 1, 0]
    assert sum(path.endswith("/tokenize") for path in paths) == 4


@pytest.mark.parametrize("failure", [500, httpx.ReadTimeout("hung")], ids=str)
def test_a_count_that_fails_after_the_route_has_answered_pauses_the_check(failure) -> None:
    router = _FailingCounts(failure, lambda order, _inputs: order > 1, hold_s=0.0)
    client = router.client()
    good, failed, paused = _FOUR_WIDE_TEXTS[:3]
    client.embed([good])
    client.embed([failed])
    client.embed([paused])
    assert router.counted == [[good], [failed]]
    assert router.embedded == [good, failed, paused]  # each input reaches the server whole


def test_a_thread_whose_own_count_fails_after_the_probe_pauses_the_check() -> None:
    good, failed, paused = _FOUR_WIDE_TEXTS[:3]
    router = _FailingCounts(500, lambda _order, inputs: failed in inputs)
    client = router.client()
    probe = threading.Thread(target=lambda: client.embed([good]))
    probe.start()
    assert router.first_sent.wait(5.0)
    client.embed([failed])
    probe.join()
    client.embed([paused])
    assert router.counted == [[good], [failed]]
    assert sorted(router.embedded) == sorted([good, failed, paused])  # each one whole


def test_a_failed_count_after_the_embed_still_reports_the_inputs_of_the_call(caplog) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(500, text="boom"))
    http = httpx.Client(transport=httpx.MockTransport(router), base_url="http://gpu0")
    client = LlamaServerClient(
        "http://gpu0",
        "m",
        http=http,
        token_cap=640,
        profile=TEI.request_profile(WorkerRole.EMBED),
    )
    with caplog.at_level("WARNING"), truncation_scope():
        assert len(client.embed([_words(3), _words(4), _words(30)])) == 3
        note_truncated(1, 1, _ROUTER_LIMIT)
    assert router.paths[0].endswith("/embeddings")  # no input is past the estimate
    assert sum(path.endswith("/tokenize") for path in router.paths) == 1
    [record] = [r for r in caplog.records if "embed inputs" in r.getMessage()]
    assert "1 of 4 embed inputs" in record.getMessage()


def test_a_count_answer_of_the_wrong_length_after_the_embed_is_a_failed_check(caplog) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, json=[[{"id": 1}]]))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("WARNING"):
        assert len(client.embed([_words(30), _words(3)])) == 2
    [record] = caplog.records
    assert "check skipped" in record.getMessage()
    assert "1 token lists for 2 inputs" in record.getMessage()


def test_a_busy_count_route_costs_a_debug_line_that_says_it_is_busy_and_no_warning(caplog) -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(429, text="slow down"))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        assert len(client.embed([_words(30)])) == 1
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "DEBUG"
    assert "The model is busy with too many requests" in record.getMessage()


def test_a_422_from_the_count_route_is_a_warning_that_carries_the_body(caplog) -> None:
    body = {"error": "inputs are not text", "error_type": "Validation"}
    router = _BrokenSideRoute("/tokenize", httpx.Response(422, json=body))
    client = _tei_client(router, WorkerRole.EMBED)
    with caplog.at_level("DEBUG"):
        assert len(client.embed([_words(30)])) == 1
    [record] = [r for r in caplog.records if "check skipped" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert "returned HTTP 422" in record.getMessage()
    assert "inputs are not text" in record.getMessage()


@pytest.mark.parametrize(
    ("answer", "kind", "said"),
    [
        (httpx.Response(429, text="slow down"), ProviderErrorKind.RATE_LIMIT, "busy"),
        (
            httpx.Response(422, text="`inputs` must have less than 512 tokens. Given: 600"),
            ProviderErrorKind.CONTEXT_OVERFLOW,
            "exceeds this model's context window",
        ),
    ],
    ids=["busy", "overflow"],
)
def test_counting_one_text_keeps_the_kind_and_the_words_of_a_refused_count(
    answer, kind, said
) -> None:
    client = _tei_client(_BrokenSideRoute("/tokenize", answer), WorkerRole.EMBED)
    with pytest.raises(ProviderError) as raised:
        client.count_text_tokens(_words(30))
    assert raised.value.kind is kind
    assert said in str(raised.value)


def test_a_refused_count_does_not_end_the_probe_for_the_counts_after_it() -> None:
    refusal = _FailingCounts(429, lambda order, _inputs: order == 1)
    client = refusal.client()
    with pytest.raises(ProviderError):
        _past_limit(client, [_FOUR_WIDE_TEXTS[2]])
    refusal.peak = 0
    _run_together([lambda text=text: _past_limit(client, [text]) for text in _FOUR_WIDE_TEXTS[:2]])
    assert refusal.peak == 1  # one probe, then the other count: a refusal is no answer


def test_a_count_of_the_wrong_length_does_not_end_the_probe_for_the_counts_after_it() -> None:
    router = _FailingCounts(httpx.Response(200, json=[]), lambda order, _inputs: order == 1)
    client = router.client()
    with pytest.raises(ValueError, match="0 token lists for 1 inputs"):
        _past_limit(client, [_FOUR_WIDE_TEXTS[2]])
    router.peak = 0
    _run_together([lambda text=text: _past_limit(client, [text]) for text in _FOUR_WIDE_TEXTS[:2]])
    assert router.peak == 1  # one probe, then the other count: a wrong length is no answer


def test_threads_that_wait_on_a_refused_probe_count_their_own_texts_at_once() -> None:
    router = _FailingCounts(429, lambda order, _inputs: order == 1)
    client = router.client()

    def count(text: str) -> int | ProviderError:
        try:
            return _past_limit(client, [text])
        except ProviderError as exc:
            return exc

    results = _run_together([lambda text=text: count(text) for text in _FOUR_WIDE_TEXTS[:3]])
    assert sum(isinstance(result, ProviderError) for result in results) == 1
    assert len(router.counted) == 3
    assert router.peak == 2  # the two waiters send together, after the refusal


def test_an_input_that_comes_twice_past_the_limit_is_counted_once_and_tallied_twice() -> None:
    router = _Embedder()
    client = _tei_client(router, WorkerRole.EMBED)
    text = _words(30)
    assert _past_limit(client, [text, text]) == 2
    assert router.tokenized == [[text]]


def test_a_count_answer_with_more_lists_than_inputs_is_a_failed_check() -> None:
    router = _BrokenSideRoute("/tokenize", httpx.Response(200, json=[[{"id": 1}] * 9, [{"id": 2}]]))
    client = _tei_client(router, WorkerRole.EMBED)
    with pytest.raises(ValueError, match="2 token lists for 1 inputs"):
        _past_limit(client, [_words(30)])
    assert not client._count_route_answered
    assert time.monotonic() < client._check_resume_at  # the failed answer paused the check


def test_threads_that_wait_on_a_probe_with_a_wrong_length_answer_get_the_failure() -> None:
    watch = _ProbeWatch(waiters=1)
    router = _HeldCountRoute(hold=watch.waiting, failing={0: httpx.Response(200, json=[])})
    client = _tei_client(router, WorkerRole.EMBED)
    client._flights = watch

    def count(text: str) -> str:
        try:
            _past_limit(client, [text])
        except ValueError as exc:
            return str(exc)
        return "counted"

    results = _run_together([lambda text=text: count(text) for text in (_words(30), _words(31))])
    assert results == ["0 token lists for 1 inputs"] * 2
    assert router.counted == 1  # the waiter sent no request of its own


class _SingleFlightProbe:
    """A request that blocks until released, and counts how often it was sent."""

    def __init__(self) -> None:
        self.sent = 0
        self.release = threading.Event()

    def __call__(self) -> int:
        self.sent += 1
        self.release.wait(5.0)
        return 7


def test_threads_that_ask_for_one_key_together_send_one_request() -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    probe = _SingleFlightProbe()
    threading.Timer(0.3, probe.release.set).start()
    results = _run_together([lambda: client._single_flight(("k",), probe)] * 4)
    assert results == [7, 7, 7, 7]
    assert probe.sent == 1


def test_threads_that_wait_on_a_failed_request_get_its_failure() -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    release = threading.Event()
    sent: list[int] = []

    def boom() -> int:
        sent.append(1)
        release.wait(5.0)
        raise ValueError("the route failed")

    def ask() -> str:
        try:
            client._single_flight(("k",), boom)
        except ValueError as exc:
            return str(exc)
        return "no failure"

    threading.Timer(0.3, release.set).start()
    assert _run_together([ask] * 3) == ["the route failed"] * 3
    assert len(sent) == 1


def test_a_request_for_another_key_does_not_wait_on_one_in_flight() -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    probe = _SingleFlightProbe()
    holder = threading.Thread(target=lambda: client._single_flight(("a",), probe))
    holder.start()
    deadline = time.monotonic() + 5.0
    while not probe.sent and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client._single_flight(("b",), lambda: 3) == 3
    probe.release.set()
    holder.join()


def test_a_failed_request_clears_its_slot_so_the_next_caller_sends_a_new_one() -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)

    def boom() -> int:
        raise RuntimeError("not a router failure")

    with pytest.raises(RuntimeError):
        client._single_flight(("k",), boom)
    assert client._flights == {}
    assert client._single_flight(("k",), lambda: 5) == 5


def test_a_finished_request_clears_its_slot() -> None:
    client = _tei_client(_Embedder(), WorkerRole.EMBED)
    assert client._single_flight(("k",), lambda: 1) == 1
    assert client._single_flight(("k",), lambda: 2) == 2
    assert client._flights == {}
