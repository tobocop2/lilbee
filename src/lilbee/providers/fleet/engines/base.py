"""The contract an inference engine implements for the fleet."""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.roles import RerankMode, WorkerRole

if TYPE_CHECKING:
    from lilbee.providers.fleet.adapters import RoleServerSpec
    from lilbee.providers.fleet.devices import DeviceProbe
    from lilbee.providers.fleet.engines.host_gate import HostGate
    from lilbee.providers.fleet.host_caps import HostCaps
    from lilbee.providers.fleet.launch import InstanceLaunch


class EngineName(StrEnum):
    """The engines the fleet can launch."""

    LLAMA = "llama.cpp"
    IK_LLAMA = "ik_llama.cpp"
    MLX = "mlx"
    OPENVINO = "openvino"


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


class EngineAdapter(Protocol, Hashable):
    """One inference engine behind the fleet; hashable, so per-engine caches can key on it.

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
    memory_kind: MemoryKind
    """How far this engine's memory figures can be trusted, known without an estimate."""
    takes_device_names: bool
    """Whether ``launch_argv`` puts ``LaunchRequest.device_names`` on the command line."""

    def supports_architecture(self, architecture: str) -> bool: ...

    def host_gate(self, host: HostCaps) -> HostGate:
        """Whether *host* can run this engine; callable at any time, before ``binary()``."""
        ...

    def binary(self) -> Path: ...

    def identity(self, binary: Path) -> str: ...

    def pin_record(self) -> Mapping[str, Any]: ...

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec: ...

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]: ...

    def discover_devices(self, binary: Path) -> DeviceProbe: ...

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        """Size one instance; ``per_device`` follows a ratio only if ``launch_argv`` honours it."""
        ...

    def check_readback(self, binary: Path) -> bool:
        """Whether *binary* can report its memory use by endpoint."""
        ...

    def supports_fit_target(self, binary: Path) -> bool:
        """Whether *binary* places layers itself, leaving a stated margin free."""
        ...

    def supports_shared_kv(self, binary: Path) -> bool:
        """Whether *binary* can give its slots one KV pool with a per-slot cap."""
        ...

    def model_meta(self, path: Path) -> dict[str, str] | None:
        """Header metadata of the model file at *path*, or ``None`` when it has none."""
        ...

    def assert_loadable(self, repo: str, filename: str, token: str | Literal[False] | None) -> None:
        """Raise ``UnsupportedQuantError`` when the engine cannot load the remote file."""
        ...

    def runtime_env(self) -> Mapping[str, str]:
        """Extra environment variables for a spawned engine process."""
        ...

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]: ...

    def request_profile(self, role: WorkerRole) -> RequestProfile: ...
