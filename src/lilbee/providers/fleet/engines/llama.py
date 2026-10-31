"""The llama.cpp adapter; the only module that names llama.cpp's tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lilbee._generated.engine_archs import ENGINE_LLAMA_CPP_REF, SUPPORTED_ARCHS
from lilbee.providers.fleet.adapters import (
    FLAG_FIT_TARGET,
    FLAG_KV_UNIFIED_PER_SLOT,
    ROLE_SPECS,
    RoleServerSpec,
    build_server_argv,
    embed_spec,
    rerank_spec,
)
from lilbee.providers.fleet.binary import (
    engine_binary_identity,
    llama_server_runtime_env,
    resolve_llama_server,
)
from lilbee.providers.fleet.devices import DeviceProbe, probe_devices
from lilbee.providers.fleet.engines.base import (
    EngineName,
    LaunchRequest,
    MemoryEstimate,
    MemoryKind,
    MemoryRequest,
    ModelFormat,
    RequestProfile,
)
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.loadability import assert_engine_can_load
from lilbee.providers.fleet.readback import engine_advertises, supports_memory_readback
from lilbee.providers.fleet.vram import estimate_instance_footprint
from lilbee.providers.gguf_meta import read_gguf_metadata
from lilbee.providers.roles import RerankMode, WorkerRole

_PROFILE = RequestProfile(
    embed_fields={"embd_normalize": -1},
    stream_fields={"return_progress": True},
    tokenize_path="/tokenize",
    detokenize_path="/detokenize",
    apply_template_path="/apply-template",
    label="llama-server",
    context_overflow=re.compile("exceed_context_size|context size", re.IGNORECASE),
    batch_overflow=re.compile("too large to process"),
    unreadable_embedding_hint="base64 embedding responses need llama-server b4391 or newer.",
    reasoning_field="reasoning_content",
)
_GGUF_SUFFIX = ".gguf"
_PIN_LAUNCH = ["{binary}", "--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"]
_PIN_SIZING = ["gguf-parser", "--model", "{model}"]


class LlamaCppAdapter:
    """llama-server behind the engine contract."""

    name = EngineName.LLAMA
    formats = frozenset({ModelFormat.GGUF})
    architectures = frozenset(SUPPORTED_ARCHS)
    served_roles = frozenset(role for role, spec in ROLE_SPECS.items() if spec.server_capable)

    def supports_architecture(self, architecture: str) -> bool:
        return architecture in self.architectures

    def binary(self) -> Path:
        return resolve_llama_server()

    def identity(self, binary: Path) -> str:
        return engine_binary_identity(binary)

    def pin_record(self) -> Mapping[str, Any]:
        return {
            "engine": self.name.value,
            "version": ENGINE_LLAMA_CPP_REF,
            "backend": None,
            "launch": _PIN_LAUNCH,
            "platforms": {},
            "memory": {"sizing": _PIN_SIZING, "rule": None},
            "licenses": ["MIT"],
        }

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role is WorkerRole.RERANK and rerank_mode is not None:
            return rerank_spec(rerank_mode)
        if role is WorkerRole.EMBED:
            return embed_spec(meta)
        return ROLE_SPECS[role]

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        return build_server_argv(
            binary=request.binary,
            spec=spec,
            model_path=request.model_path,
            devices=request.devices,
            device_names=request.device_names,
            n_gpu_layers=request.n_gpu_layers,
            slots=request.slots,
            ctx_per_slot=request.ctx_per_slot,
            tensor_split=request.tensor_split,
            mmproj=request.mmproj,
            flash_attn=request.flash_attn,
            cache_type=request.cache_type,
            batch_size=request.batch_size,
            no_mmap=request.no_mmap,
            cpu_moe=request.cpu_moe,
            n_cpu_moe=request.n_cpu_moe,
            memory_endpoint=request.memory_endpoint,
            fit_target_mib=request.fit_target_mib,
            kv_unified_per_slot=request.kv_unified_per_slot,
        )

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return probe_devices(binary)

    def model_meta(self, path: Path) -> Mapping[str, str] | None:
        if path.suffix.lower() != _GGUF_SUFFIX:
            return None
        return read_gguf_metadata(path)

    def assert_loadable(self, repo: str, filename: str, token: str | None) -> None:
        assert_engine_can_load(repo, filename, token)

    def runtime_env(self) -> Mapping[str, str]:
        return llama_server_runtime_env()

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        parsed = estimate_instance_footprint(
            request.model_path,
            ctx=request.ctx,
            slots=request.slots,
            gpu_layers=request.gpu_layers,
            flash_attn=request.flash_attn,
            kv_cache_type=request.kv_cache_type,
            mmproj_path=request.mmproj_path,
            tensor_split=request.tensor_split,
            batch_size=request.batch_size,
            expert_offload=request.expert_offload,
        )
        return MemoryEstimate(
            kind=MemoryKind.SIZED,
            device_bytes=parsed.vram_bytes,
            host_bytes=parsed.ram_bytes,
            shared_bytes=parsed.unified_bytes,
            per_device=parsed.per_device_vram,
            per_device_shared=parsed.per_device_unified,
        )

    def check_readback(self, binary: Path) -> bool:
        return supports_memory_readback(binary)

    def supports_fit_target(self, binary: Path) -> bool:
        return engine_advertises(binary, FLAG_FIT_TARGET)

    def supports_shared_kv(self, binary: Path) -> bool:
        return engine_advertises(binary, FLAG_KV_UNIFIED_PER_SLOT)

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


LLAMA = LlamaCppAdapter()
