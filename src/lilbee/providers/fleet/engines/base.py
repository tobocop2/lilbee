"""The contract an inference engine implements for the fleet."""

from __future__ import annotations

import re
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


class EngineName(StrEnum):
    """The engines the fleet can launch."""

    LLAMA = "llama.cpp"


class ModelFormat(StrEnum):
    """A weight file format an engine loads."""

    GGUF = "gguf"


class MemoryKind(StrEnum):
    """How much an engine's memory figures can be trusted."""

    SIZED = "sized"
    FIXED_RESERVE = "fixed_reserve"


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
class MemoryEstimate:
    """One instance's memory footprint under both memory models.

    A tensor-split instance fails on its busiest device, so the planner fits
    ``peak_footprint`` (the largest device), never the sum.
    """

    kind: MemoryKind
    device_bytes: int
    """Discrete-GPU model: bytes resident in device memory, summed over devices."""
    host_bytes: int
    """Discrete-GPU model: bytes held in host memory."""
    shared_bytes: int
    """Unified-memory model: the whole resident footprint."""
    per_device: tuple[int, ...] = ()
    """Discrete-GPU model: bytes on each device."""
    per_device_shared: tuple[int, ...] = ()
    """Unified-memory model: bytes on each device."""

    def footprint(self, *, unified: bool) -> int:
        """Total bytes to charge against a shared budget for this memory model."""
        return self.shared_bytes if unified else self.device_bytes

    def peak_footprint(self, *, unified: bool) -> int:
        """The busiest single device's bytes, or the total without a per-device breakdown."""
        per_device = self.per_device_shared if unified else self.per_device
        return max(per_device) if per_device else self.footprint(unified=unified)


@dataclass(frozen=True)
class RequestProfile:
    """The request fields and paths the client uses against an engine's server."""

    embed_fields: Mapping[str, Any]
    """Extra fields on every embeddings request body."""
    stream_fields: Mapping[str, Any]
    """Extra fields on a streamed chat request body."""
    tokenize_path: str
    detokenize_path: str
    apply_template_path: str
    label: str
    """The name of the engine's server in error messages."""
    context_overflow: re.Pattern[str]
    """Matches a 400 body that reports a prompt past the context window."""
    batch_overflow: re.Pattern[str]
    """Matches a 500 body that reports an input past the physical batch."""
    unreadable_embedding_hint: str
    """What to tell the user when embedding vectors do not decode."""
    reasoning_field: str
    """The response field that carries reasoning text apart from the answer."""


class EngineAdapter(Protocol):
    """One inference engine behind the fleet.

    The provider holds one adapter. Call order: ``binary()`` resolves the
    executable, ``discover_devices`` probes it, ``estimate_memory`` sizes a
    plan, ``role_spec`` and ``launch_argv`` build each command line, and
    ``check_readback`` says whether the running instance reports its memory.
    """

    @property
    def name(self) -> str: ...

    @property
    def formats(self) -> frozenset[str]: ...

    architectures: frozenset[str]
    served_roles: frozenset[WorkerRole]

    def supports_architecture(self, architecture: str) -> bool: ...

    def binary(self) -> Path: ...

    def identity(self, binary: Path) -> str: ...

    def pin_record(self) -> Mapping[str, Any]: ...

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec: ...

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]: ...

    def discover_devices(self, binary: Path) -> DeviceProbe: ...

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate: ...

    def check_readback(self, binary: Path) -> bool:
        """Whether *binary* can report its memory use by endpoint."""
        ...

    def supports_fit_target(self, binary: Path) -> bool:
        """Whether *binary* places layers itself, leaving a stated margin free."""
        ...

    def supports_shared_kv(self, binary: Path) -> bool:
        """Whether *binary* can give its slots one KV pool with a per-slot cap."""
        ...

    def model_meta(self, path: Path) -> Mapping[str, str] | None:
        """Header metadata of the model file at *path*, or ``None`` when it has none."""
        ...

    def assert_loadable(self, repo: str, filename: str, token: str | None) -> None:
        """Raise ``UnsupportedQuantError`` when the engine cannot load the remote file."""
        ...

    def runtime_env(self) -> Mapping[str, str]:
        """Extra environment variables for a spawned engine process."""
        ...

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]: ...

    def request_profile(self, role: WorkerRole) -> RequestProfile: ...
