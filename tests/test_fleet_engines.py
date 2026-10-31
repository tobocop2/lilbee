"""The llama.cpp engine adapter: each duty answers the way the code behind it does."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import pytest

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.fleet import adapters
from lilbee.providers.fleet.engines import LLAMA, EngineName, LaunchRequest
from lilbee.providers.fleet.engines.base import (
    MemoryEstimate,
    MemoryKind,
    MemoryRequest,
    ModelFormat,
)
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.vram import GgufVramEstimate
from lilbee.providers.roles import RerankMode, WorkerRole

_ROOT = Path(__file__).resolve().parents[1]
_LLAMA_MODULE = "lilbee.providers.fleet.engines.llama"
_SWEEP_INPUTS = 400


def test_llama_identity_fields() -> None:
    assert LLAMA.name is EngineName.LLAMA
    assert LLAMA.formats == {ModelFormat.GGUF}


def test_llama_supports_known_and_rejects_unknown_architecture() -> None:
    assert LLAMA.architectures
    known = sorted(LLAMA.architectures)[0]
    assert LLAMA.supports_architecture(known)
    assert not LLAMA.supports_architecture("no-such-architecture")


def test_llama_serves_every_role() -> None:
    assert LLAMA.served_roles == set(WorkerRole)


def test_llama_binary_resolves_through_the_bundled_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_LLAMA_MODULE}.resolve_llama_server", lambda: Path("/bin/srv"))
    assert LLAMA.binary() == Path("/bin/srv")


def test_llama_identity_digests_the_binary(tmp_path: Path) -> None:
    binary = tmp_path / "srv"
    binary.write_bytes(b"engine")
    assert re.fullmatch(rf"{re.escape(str(binary))}@[0-9a-f]{{64}}", LLAMA.identity(binary))
    assert LLAMA.identity(tmp_path / "missing").endswith("@unreadable")


def test_llama_pin_record_version_is_the_pinned_ref() -> None:
    env = (_ROOT / "engine-versions.env").read_text(encoding="utf-8")
    ref = re.search(r"^ENGINE_LLAMA_CPP_REF=(.+)$", env, re.M)
    assert ref is not None
    record = LLAMA.pin_record()
    assert record["version"] == ref.group(1)
    assert {"engine", "version", "backend", "launch", "platforms", "memory", "licenses"} <= set(
        record
    )


def test_llama_role_spec_picks_rerank_embed_or_default() -> None:
    llm = LLAMA.role_spec(WorkerRole.RERANK, RerankMode.LLM, None)
    assert llm is adapters.LLM_RERANK_SPEC
    assert LLAMA.role_spec(WorkerRole.RERANK, None, None) is adapters.ROLE_SPECS[WorkerRole.RERANK]
    decoder = LLAMA.role_spec(WorkerRole.EMBED, None, {"architecture": "qwen3"})
    assert decoder.extra_args == ("--embeddings", "--pooling", "last")
    assert LLAMA.role_spec(WorkerRole.CHAT, None, None) is adapters.ROLE_SPECS[WorkerRole.CHAT]


def test_llama_role_spec_ignores_a_rerank_mode_on_other_roles() -> None:
    spec = LLAMA.role_spec(WorkerRole.CHAT, RerankMode.LLM, None)
    assert spec is adapters.ROLE_SPECS[WorkerRole.CHAT]


def test_llama_launch_argv_matches_the_argv_builder(tmp_path: Path) -> None:
    binary = tmp_path / "srv"
    binary.write_text("#!/bin/sh\necho '  --model X  text'\n", encoding="utf-8")
    binary.chmod(0o755)
    spec = adapters.ROLE_SPECS[WorkerRole.CHAT]
    request = LaunchRequest(
        binary=binary,
        model_path=Path("/m/a.gguf"),
        devices=(0, 1),
        device_names=("CUDA0", "CUDA1"),
        n_gpu_layers=-1,
        slots=2,
        ctx_per_slot=4096,
        tensor_split=(1, 1),
        n_cpu_moe=3,
        fit_target_mib=(512, 512),
        kv_unified_per_slot=4096,
    )
    assert LLAMA.launch_argv(spec, request) == adapters.build_server_argv(
        binary=binary,
        spec=spec,
        model_path=Path("/m/a.gguf"),
        devices=(0, 1),
        device_names=("CUDA0", "CUDA1"),
        n_gpu_layers=-1,
        slots=2,
        ctx_per_slot=4096,
        tensor_split=(1, 1),
        n_cpu_moe=3,
        fit_target_mib=(512, 512),
        kv_unified_per_slot=4096,
    )


def test_llama_discover_devices_probes_the_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def probe(binary: Path) -> str:
        seen.append(binary)
        return "probed"

    monkeypatch.setattr(f"{_LLAMA_MODULE}.probe_devices", probe)
    assert LLAMA.discover_devices(Path("/bin/srv")) == "probed"
    assert seen == [Path("/bin/srv")]


def test_llama_estimate_memory_passes_every_request_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def estimate(model_path: Path, **kwargs: Any) -> GgufVramEstimate:
        seen["model_path"] = model_path
        seen.update(kwargs)
        return GgufVramEstimate(
            vram_bytes=40,
            ram_bytes=5,
            unified_bytes=60,
            per_device_vram=(30, 10),
            per_device_unified=(35, 25),
        )

    monkeypatch.setattr(f"{_LLAMA_MODULE}.estimate_instance_footprint", estimate)
    request = MemoryRequest(
        model_path=Path("/m/a.gguf"),
        ctx=8192,
        slots=2,
        gpu_layers=33,
        flash_attn=True,
        kv_cache_type=KvCacheType.Q8_0,
        mmproj_path=Path("/m/p.gguf"),
        tensor_split=(2, 1),
        batch_size=512,
        expert_offload=("blk.*",),
    )
    assert LLAMA.estimate_memory(request) == MemoryEstimate(
        kind=MemoryKind.SIZED,
        device_bytes=40,
        host_bytes=5,
        shared_bytes=60,
        per_device=(30, 10),
        per_device_shared=(35, 25),
    )
    assert seen == {
        "model_path": Path("/m/a.gguf"),
        "ctx": 8192,
        "slots": 2,
        "gpu_layers": 33,
        "flash_attn": True,
        "kv_cache_type": KvCacheType.Q8_0,
        "mmproj_path": Path("/m/p.gguf"),
        "tensor_split": (2, 1),
        "batch_size": 512,
        "expert_offload": ("blk.*",),
    }


def test_llama_memory_estimate_charges_what_the_parser_estimate_charged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = random.Random(1180)
    request = MemoryRequest(
        model_path=Path("/m/a.gguf"),
        ctx=4096,
        slots=1,
        gpu_layers=-1,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )
    compared = 0
    for _ in range(_SWEEP_INPUTS):
        width = rng.randint(0, 4)
        parsed = GgufVramEstimate(
            vram_bytes=rng.randint(0, 10**11),
            ram_bytes=rng.randint(0, 10**10),
            unified_bytes=rng.randint(0, 10**11),
            per_device_vram=tuple(rng.randint(0, 10**10) for _ in range(width)),
            per_device_unified=tuple(rng.randint(0, 10**10) for _ in range(width)),
        )
        monkeypatch.setattr(
            f"{_LLAMA_MODULE}.estimate_instance_footprint",
            lambda *_a, _parsed=parsed, **_k: _parsed,
        )
        estimate = LLAMA.estimate_memory(request)
        assert estimate.host_bytes == parsed.ram_bytes
        for unified in (False, True):
            assert estimate.footprint(unified=unified) == parsed.footprint(unified=unified)
            assert estimate.peak_footprint(unified=unified) == parsed.peak_footprint(
                unified=unified
            )
            compared += 1
    assert compared == 2 * _SWEEP_INPUTS


def test_llama_model_meta_reads_a_gguf_header(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def _read(path: Path) -> dict[str, str]:
        seen.append(path)
        return {"architecture": "llama"}

    monkeypatch.setattr(f"{_LLAMA_MODULE}.read_gguf_metadata", _read)
    assert LLAMA.model_meta(Path("/m/model.GGUF")) == {"architecture": "llama"}
    assert seen == [Path("/m/model.GGUF")]


def test_llama_model_meta_skips_a_file_with_no_gguf_header(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> dict[str, str]:
        raise AssertionError(f"the parser must not run on {path}")

    monkeypatch.setattr(f"{_LLAMA_MODULE}.read_gguf_metadata", _read)
    assert LLAMA.model_meta(Path("/m/model.safetensors")) is None


def test_llama_assert_loadable_asks_the_gguf_parser_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        f"{_LLAMA_MODULE}.assert_engine_can_load", lambda *args: calls.append(args)
    )
    LLAMA.assert_loadable("org/repo", "m.gguf", "tok")
    assert calls == [("org/repo", "m.gguf", "tok")]


def test_llama_runtime_env_is_the_server_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_LLAMA_MODULE}.llama_server_runtime_env", lambda: {"LD_LIBRARY_PATH": "/x"})
    assert LLAMA.runtime_env() == {"LD_LIBRARY_PATH": "/x"}


def test_llama_swap_extras_is_empty() -> None:
    launch = InstanceLaunch(role=WorkerRole.CHAT, argv=["/b"], env_overrides={}, model="m")
    assert LLAMA.swap_extras(launch) == {}


def test_llama_request_profile_holds_the_client_constants() -> None:
    profile = LLAMA.request_profile(WorkerRole.EMBED)
    assert dict(profile.embed_fields) == {"embd_normalize": -1}
    assert dict(profile.stream_fields) == {"return_progress": True}
    assert (profile.tokenize_path, profile.detokenize_path, profile.apply_template_path) == (
        "/tokenize",
        "/detokenize",
        "/apply-template",
    )
    assert profile.label == "llama-server"
    assert profile.reasoning_field == "reasoning_content"
    assert profile.context_overflow.search("The request exceeds the available context size")
    assert profile.context_overflow.search('{"type": "exceed_context_size_error"}')
    assert not profile.context_overflow.search("bad request")
    assert profile.batch_overflow.search("input (4096 tokens) is too large to process")
    assert not profile.batch_overflow.search("out of memory")
    assert "llama-server b4391" in profile.unreadable_embedding_hint
