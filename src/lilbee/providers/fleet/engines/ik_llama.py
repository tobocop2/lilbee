"""The ik_llama.cpp adapter, composed from the llama.cpp adapter; opt-in, installed from a pack."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.adapters import RoleServerSpec
from lilbee.providers.fleet.binary import EngineTool
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engine_manifest import pack_digest, pack_platforms
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest, RequestProfile
from lilbee.providers.fleet.engines.host_gate import HostGate, gate_on
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

# A commit of the upstream main branch, never "latest": a bump follows a passing harness run.
IK_LLAMA_COMMIT = "13cffc006b82e03e4d89cbb8c7327f0255ae3d18"
_VERSION_HEX_DIGITS = 12
VERSION = f"0+git{IK_LLAMA_COMMIT[:_VERSION_HEX_DIGITS]}"
# The CUDA pack is built with CUDA 12.4 for compute capability 7.5 up (the cuda build script
# names the same floors); a test reads that script and fails when they stop agreeing.
_CUDA_DRIVER_FLOOR = (12, 4)
_CUDA_MIN_COMPUTE = (7, 5)
_CUDA_LIBRARIES = ["ggml", "llama", "cudart", "cublas", "cublasLt"]
_CPU_LICENCES = ["MIT"]
_CUDA_LICENCES = ["MIT AND LicenseRef-NVIDIA-CUDA-EULA"]
_PIN_LAUNCH = ["{binary}", "--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"]
_PIN_SIZING = ["gguf-parser", "--model", "{model}"]
_REFUSAL_TAIL = "ik_llama.cpp needs a CPU with AVX2 or NEON."
_PROFILE = dataclasses.replace(
    LLAMA.request_profile(WorkerRole.CHAT),
    label="ik_llama.cpp server",
    unreadable_embedding_hint="The ik_llama.cpp server answered embeddings that cannot be decoded.",
)


class IkBackend(StrEnum):
    """The packs of the engine, named as the digest table names them."""

    CPU = "cpu"
    CUDA = "cuda"


def pick_backend(host: HostCaps) -> IkBackend:
    """CUDA on Linux x86_64 when each card has compute 7.5+ and the driver offers CUDA 12.4."""
    if (host.os, host.arch) != (HostOS.LINUX, HostArch.X86_64) or not host.nvidia_known:
        return IkBackend.CPU
    if not host.gpus or (host.nvidia_driver_cuda or (0, 0)) < _CUDA_DRIVER_FLOOR:
        return IkBackend.CPU
    if min(gpu.compute for gpu in host.gpus) < _CUDA_MIN_COMPUTE:
        return IkBackend.CPU
    return IkBackend.CUDA


class IkLlamaAdapter:
    """ik_llama.cpp behind the engine contract; everything but the build comes from llama.cpp."""

    name = EngineName.IK_LLAMA
    formats = frozenset({ModelFormat.GGUF})
    architectures = LLAMA.architectures
    served_roles = frozenset({WorkerRole.CHAT, WorkerRole.EMBED})
    memory_kind = MemoryKind.SIZED
    takes_device_names = True
    bundled = False

    def supports_architecture(self, architecture: str) -> bool:
        return architecture in self.architectures

    def host_gate(self, host: HostCaps) -> HostGate:
        """Pass on a CPU with AVX2 (x86) or NEON (ARM); an unreadable CPU stays selectable."""
        feature = host.neon if host.arch is HostArch.ARM64 else host.avx2
        gate = gate_on(feature)
        if gate.ok:
            return gate
        return HostGate(False, f"{gate.reason} {_REFUSAL_TAIL}")

    def binary(self) -> Path:
        """The server of the installed build of the pin; raises when the pack is not installed."""
        from lilbee.catalog.runnable import session_host

        backend = pick_backend(session_host())
        store = EngineStore(engines_root(cfg.models_dir))
        for build in store.installed():
            of_pin = (build.engine, build.version, build.backend) == (self.name, VERSION, backend)
            if of_pin and build.binary is not None:
                return build.binary
        raise ProviderError(
            f"Engine {self.name} is not installed. Run lilbee engine install {self.name}.",
            kind=ProviderErrorKind.NOT_FOUND,
        )

    def identity(self, binary: Path) -> str:
        return LLAMA.identity(binary)

    def pin_record(self) -> Mapping[str, Any]:
        from lilbee.catalog.runnable import session_host

        host = session_host()
        backend = pick_backend(host)
        platforms = pack_platforms(self.name.value, VERSION, host, backend)
        record: dict[str, Any] = {
            "engine": self.name.value,
            "version": VERSION,
            "backend": backend.value,
            "sha256": pack_digest(self.name.value, VERSION, host, backend),
            "tool": EngineTool.LLAMA_SERVER.value,
            "launch": _PIN_LAUNCH,
            "platforms": platforms,
            "memory": {"sizing": _PIN_SIZING, "rule": None},
            "licenses": _CUDA_LICENCES if backend is IkBackend.CUDA else _CPU_LICENCES,
        }
        if backend is IkBackend.CUDA:
            record["libraries"] = _CUDA_LIBRARIES
        return record

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        return LLAMA.role_spec(role, rerank_mode, meta)

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        return LLAMA.launch_argv(spec, request)

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return LLAMA.discover_devices(binary)

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        return LLAMA.device_targets(binary)

    def model_meta(self, path: Path) -> dict[str, str] | None:
        return LLAMA.model_meta(path)

    def assert_loadable(self, repo: str, filename: str, token: str | Literal[False] | None) -> None:
        LLAMA.assert_loadable(repo, filename, token)

    def runtime_env(self) -> Mapping[str, str]:
        return LLAMA.runtime_env()

    def target_env(self, target: RoleTarget, cards: int) -> Mapping[str, str]:
        return LLAMA.target_env(target, cards)

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        return LLAMA.estimate_memory(request)

    def check_readback(self, binary: Path) -> bool:
        return LLAMA.check_readback(binary)

    def supports_fit_target(self, binary: Path) -> bool:
        return LLAMA.supports_fit_target(binary)

    def supports_shared_kv(self, binary: Path) -> bool:
        return LLAMA.supports_shared_kv(binary)

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return LLAMA.swap_extras(launch)

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


IK_LLAMA = IkLlamaAdapter()
ADAPTER = IK_LLAMA
