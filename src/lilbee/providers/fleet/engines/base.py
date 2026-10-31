"""The contract an inference engine implements for the fleet."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from lilbee.core.config.enums import KvCacheType
from lilbee.providers.roles import RerankMode, WorkerRole

if TYPE_CHECKING:
    from lilbee.providers.fleet.adapters import RoleServerSpec
    from lilbee.providers.fleet.devices import DeviceProbe
    from lilbee.providers.fleet.launch import InstanceLaunch
    from lilbee.providers.fleet.vram import GgufVramEstimate


class EngineName(StrEnum):
    """The engines the fleet can launch."""

    LLAMA = "llama.cpp"


class ModelFormat(StrEnum):
    """A weight file format an engine loads."""

    GGUF = "gguf"


class ReadbackKind(StrEnum):
    """How the fleet reads an instance's real memory use back from the engine."""

    ENDPOINT = "endpoint"
    LOG = "log"
    NONE = "none"


@dataclass(frozen=True)
class LaunchRequest:
    """Placement and load choices for one instance; the adapter turns them into argv."""

    binary: Path
    model_path: Path
    devices: tuple[int, ...]
    n_gpu_layers: int
    slots: int
    ctx_per_slot: int
    device_names: tuple[str, ...] = ()
    tensor_split: tuple[int, ...] = ()
    mmproj: Path | None = None
    flash_attn: str | None = None
    cache_type: str | None = None
    batch_size: int | None = None
    no_mmap: bool = False
    cpu_moe: bool = False
    n_cpu_moe: int | None = None
    memory_endpoint: bool = False
    fit_target_mib: tuple[int, ...] = ()
    kv_unified_per_slot: int | None = None


@dataclass(frozen=True)
class MemoryRequest:
    """What the memory estimate needs to know about one instance."""

    model_path: Path
    ctx: int
    slots: int
    gpu_layers: int
    flash_attn: bool
    kv_cache_type: KvCacheType
    mmproj_path: Path | None = None
    tensor_split: tuple[int, ...] = ()
    batch_size: int | None = None
    expert_offload: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestProfile:
    """The request fields and paths the client uses against an engine's server."""

    embd_normalize: int | None
    return_progress: bool
    tokenize_path: str
    detokenize_path: str
    apply_template_path: str


class EngineAdapter(Protocol):
    """One inference engine behind the fleet.

    The provider holds one adapter. Call order: ``binary()`` resolves the
    executable, ``discover_devices`` probes it, ``estimate_memory`` sizes a
    plan, ``role_spec`` and ``launch_argv`` build each command line, and
    ``readback`` says how the running instance reports its memory.
    """

    name: EngineName
    formats: frozenset[ModelFormat]
    architectures: frozenset[str]
    served_roles: frozenset[WorkerRole]
    readback: ReadbackKind

    def supports_architecture(self, architecture: str) -> bool: ...

    def binary(self) -> Path: ...

    def identity(self, binary: Path) -> str: ...

    def pin_record(self) -> Mapping[str, Any]: ...

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec: ...

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]: ...

    def discover_devices(self, binary: Path) -> DeviceProbe: ...

    def estimate_memory(self, request: MemoryRequest) -> GgufVramEstimate: ...

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]: ...

    def request_profile(self, role: WorkerRole) -> RequestProfile: ...
