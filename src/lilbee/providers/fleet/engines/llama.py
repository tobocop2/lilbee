"""The llama.cpp adapter; the only module that names llama.cpp's tools."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lilbee._generated.engine_archs import ENGINE_LLAMA_CPP_REF, SUPPORTED_ARCHS
from lilbee.providers.fleet.adapters import (
    ROLE_SPECS,
    RoleServerSpec,
    build_server_argv,
    embed_spec,
    rerank_spec,
)
from lilbee.providers.fleet.binary import (
    engine_binary_identity,
    resolve_gguf_parser,
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
    ReadbackKind,
    RequestProfile,
)
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.vram import estimate_instance_footprint
from lilbee.providers.roles import RerankMode, WorkerRole

_EMBD_NORMALIZE_NONE = -1
_PROFILE = RequestProfile(
    embd_normalize=_EMBD_NORMALIZE_NONE,
    return_progress=True,
    tokenize_path="/tokenize",
    detokenize_path="/detokenize",
    apply_template_path="/apply-template",
)
_PIN_LAUNCH = ["{binary}", "--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"]
_PIN_SIZING = ["gguf-parser", "--model", "{model}"]


class LlamaCppAdapter:
    """llama-server behind the engine contract."""

    name = EngineName.LLAMA
    formats = frozenset({ModelFormat.GGUF})
    architectures = frozenset(SUPPORTED_ARCHS)
    served_roles = frozenset(role for role, spec in ROLE_SPECS.items() if spec.server_capable)
    readback = ReadbackKind.ENDPOINT

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

    def header_reader(self) -> Path:
        return resolve_gguf_parser()

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

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


LLAMA = LlamaCppAdapter()
