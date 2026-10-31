"""The text-embeddings-inference adapter; the only module that names its server and builds."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.fleet.adapters import RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engine_store import (
    EngineNotInstalledError,
    EngineStore,
    InstalledEngine,
    engines_root,
)
from lilbee.providers.fleet.engines.base import (
    EngineName,
    LaunchRequest,
    RequestProfile,
)
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS, detect
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.roles import RerankMode, WorkerRole

_VERSION = "1.9.4"
# The digest of the pack; None until a pack is published, and an install is refused then.
PACK_SHA256: str | None = None
_CPU = "cpu"
_METAL = "metal"
_CUDA_PREFIX = "cuda-sm"
_MIN_COMPUTE = (7, 5)
_CUDA_CELLS = (75, 80, 86, 89, 90, 100, 120)
"""The compute capabilities (major * 10 + minor) a CUDA build exists for."""
_RESERVE_BYTES = 2 * 1024**3
"""The figure charged for one instance; two small models peaked at 886 MiB in the recording."""
_BATCH_LIMIT = 64
"""The inputs per request: the launch argv sets it on the server, the client splits at it."""
_SERVED = frozenset({WorkerRole.EMBED, WorkerRole.RERANK})
_ENDPOINTS = {WorkerRole.EMBED: "/v1/embeddings", WorkerRole.RERANK: "/rerank"}

_PROFILE = RequestProfile(
    embed_fields={},
    stream_fields={},
    tokenize_path=None,
    detokenize_path=None,
    apply_template_path="/apply-template",
    label="text-embeddings-inference",
    context_overflow=re.compile(r"must have less than \d+ tokens"),
    batch_overflow=None,
    unreadable_embedding_hint="the server returns vectors that could not be decoded.",
    reasoning_field="reasoning_content",
    rerank_path="/rerank",
    batch_limit=_BATCH_LIMIT,
    batch_count_refusal=re.compile(r"maximum allowed batch size"),
)
_PIN_LAUNCH = ["{binary}", "--model-id", "{model}", "--hostname", "127.0.0.1", "--port", "{port}"]


@dataclass(frozen=True)
class BuildPick:
    """The build chosen for a host: its backend, or none with the reason."""

    backend: str | None
    reason: str = ""


def _cuda_cell(compute: tuple[int, int]) -> int | None:
    """The newest CUDA build of the same major version that does not exceed *compute*."""
    major, minor = compute
    fits = [c for c in _CUDA_CELLS if c // 10 == major and c % 10 <= minor]
    return max(fits, default=None)


def _gpu_pick(host: HostCaps) -> BuildPick:
    """The CUDA build for the newest GPU, else the CPU build with the reason."""
    if not host.gpus:
        return BuildPick(_CPU, host.nvidia_reason or "no NVIDIA GPU")
    best = max(gpu.compute for gpu in host.gpus)
    if best < _MIN_COMPUTE:
        found = ".".join(map(str, best))
        needed = ".".join(map(str, _MIN_COMPUTE))
        return BuildPick(
            _CPU, f"CUDA builds need compute capability {needed} or newer; found {found}"
        )
    cell = _cuda_cell(best)
    if cell is None:
        found = ".".join(map(str, best))
        return BuildPick(_CPU, f"No CUDA build covers compute capability {found}")
    return BuildPick(f"{_CUDA_PREFIX}{cell}")


class TeiAdapter:
    """text-embeddings-inference behind the engine contract; embed and rerank only."""

    name = EngineName.TEI
    formats = frozenset({ModelFormat.SAFETENSORS})
    architectures: frozenset[str] = frozenset()
    served_roles = _SERVED
    memory_kind = MemoryKind.FIXED_RESERVE
    takes_device_names = False
    bundled = False

    def supports_architecture(self, architecture: str) -> bool:
        return True

    def pick_build(self, host: HostCaps) -> BuildPick:
        """The build for *host*: CUDA by compute capability, metal on Apple silicon, else CPU."""
        if host.os is HostOS.WINDOWS:
            return BuildPick(None, "No build exists for Windows.")
        if host.os is HostOS.MACOS:
            if host.arch is HostArch.ARM64:
                return BuildPick(_METAL)
            return BuildPick(None, "No build exists for macOS on this CPU architecture.")
        if host.os is HostOS.LINUX and host.arch is HostArch.X86_64:
            return _gpu_pick(host)
        return BuildPick(None, "No build exists for this operating system and CPU architecture.")

    def host_gate(self, host: HostCaps) -> HostGate:
        pick = self.pick_build(host)
        return PASS if pick.backend is not None else HostGate(False, pick.reason)

    def binary(self) -> Path:
        record = self._record()
        if record is None or record.binary is None:
            raise EngineNotInstalledError(
                f"Engine {self.name} is not installed. Run lilbee engine install {self.name}."
            )
        return record.binary

    def _record(self) -> InstalledEngine | None:
        pin = self.pin_record()
        store = EngineStore(engines_root(cfg.models_dir))
        for build in store.installed():
            if (build.engine, build.version, build.backend) == (
                pin["engine"],
                pin["version"],
                pin["backend"],
            ):
                return build
        return None

    def identity(self, binary: Path) -> str:
        return f"{self.name}@{_VERSION}"

    def pin_record(self) -> Mapping[str, Any]:
        return {
            "engine": self.name.value,
            "version": _VERSION,
            "backend": self.pick_build(detect()).backend,
            "sha256": PACK_SHA256,
            "launch": _PIN_LAUNCH,
            "platforms": {},
            "memory": {"sizing": None, "rule": f"fixed {_RESERVE_BYTES} bytes"},
            "licenses": ["Apache-2.0"],
        }

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role not in _SERVED:
            raise ValueError(f"Engine {self.name} serves embed and rerank, not {role}.")
        return RoleServerSpec(
            role=role, endpoint_path=_ENDPOINTS[role], extra_args=(), server_capable=True
        )

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        return [
            str(request.binary),
            "--model-id",
            str(request.model_path),
            "--hostname",
            "127.0.0.1",
            "--max-client-batch-size",
            str(_PROFILE.batch_limit),
            *spec.extra_args,
        ]

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=[], output="", spoke_protocol=False)

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        return MemoryEstimate(
            kind=self.memory_kind,
            device_bytes=_RESERVE_BYTES,
            host_bytes=0,
            shared_bytes=_RESERVE_BYTES,
        )

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def model_meta(self, path: Path) -> dict[str, str] | None:
        return None

    def assert_loadable(self, repo: str, filename: str, token: str | Literal[False] | None) -> None:
        return None

    def runtime_env(self) -> Mapping[str, str]:
        return {}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {"checkEndpoint": "/health"}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


TEI = TeiAdapter()
