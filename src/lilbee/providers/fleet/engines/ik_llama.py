"""The ik_llama.cpp adapter, composed from the llama.cpp adapter; opt-in, installed from a pack."""

from __future__ import annotations

import dataclasses
import logging
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
from lilbee.providers.fleet.engines.base import (
    EngineName,
    LaunchRequest,
    RequestProfile,
    ServerProcess,
)
from lilbee.providers.fleet.engines.host_gate import HostGate, gate_on
from lilbee.providers.fleet.engines.llama import LLAMA, has_usable_cuda
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

log = logging.getLogger(__name__)

# A commit of the upstream main branch, never "latest": a bump follows a passing harness run.
IK_LLAMA_COMMIT = "13cffc006b82e03e4d89cbb8c7327f0255ae3d18"
_VERSION_HEX_DIGITS = 12
VERSION = f"0+git{IK_LLAMA_COMMIT[:_VERSION_HEX_DIGITS]}"

# The CUDA build is made with CUDA 12.4 for compute capability 7.5 (Turing) and newer.
# A test reads the build script and fails when it stops agreeing with these two.
_CUDA_DRIVER_FLOOR = (12, 4)
_CUDA_MIN_COMPUTE = (7, 5)
_CUDA_LICENCES = ["MIT AND LicenseRef-NVIDIA-CUDA-EULA AND Apache-2.0 AND MIT-0 AND Unlicense"]
_CPU_LICENCES = ["MIT AND Apache-2.0 AND MIT-0 AND Unlicense"]
_PIN_LAUNCH = ["{binary}", "--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"]
_PIN_SIZING = ["gguf-parser", "--model", "{model}"]
_REFUSAL_TAIL = "ik_llama.cpp needs a CPU with AVX2 or NEON."
_PROFILE = dataclasses.replace(
    LLAMA.request_profile(WorkerRole.CHAT),
    label="ik_llama.cpp server",
    unreadable_embedding_hint="The ik_llama.cpp server answered embeddings that cannot be decoded.",
)


class IkBackend(StrEnum):
    """The builds of the pack: one CPU baseline, or the same server with CUDA."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclasses.dataclass(frozen=True)
class BuildPick:
    """The build chosen for a host, with the reason when it is not the CUDA build."""

    backend: IkBackend
    reason: str = ""


def _version(pair: tuple[int, int]) -> str:
    return f"{pair[0]}.{pair[1]}"


class IkLlamaAdapter:
    """ik_llama.cpp behind the engine contract; everything but the build comes from llama.cpp."""

    name = EngineName.IK_LLAMA
    formats = frozenset({ModelFormat.GGUF})
    architectures = LLAMA.architectures
    served_roles = frozenset({WorkerRole.CHAT, WorkerRole.EMBED})
    memory_kind = MemoryKind.SIZED
    takes_device_names = True
    concatenates_pooling = False
    bundled = False
    server_process = ServerProcess(name="llama-server")

    def supports_architecture(self, architecture: str) -> bool:
        return architecture in self.architectures

    def host_gate(self, host: HostCaps) -> HostGate:
        """Pass on a CPU with AVX2 (x86) or NEON (ARM); an unreadable CPU stays selectable."""
        feature = host.neon if host.arch is HostArch.ARM64 else host.avx2
        gate = gate_on(feature)
        if gate.ok:
            return gate
        return HostGate(False, f"{gate.reason} {_REFUSAL_TAIL}")

    def pick_build(self, host: HostCaps) -> BuildPick:
        """The CUDA build on a Linux x86_64 host with a card and driver that it serves, else CPU."""
        if (host.os, host.arch) != (HostOS.LINUX, HostArch.X86_64):
            return BuildPick(IkBackend.CPU, "No CUDA build exists for this host.")
        if has_usable_cuda(host, _CUDA_DRIVER_FLOOR, _CUDA_MIN_COMPUTE):
            return BuildPick(IkBackend.CUDA)
        if not host.gpus:
            return BuildPick(IkBackend.CPU, host.nvidia_reason or "No NVIDIA GPU found.")
        card = max(host.gpus, key=lambda gpu: gpu.compute)
        if card.compute < _CUDA_MIN_COMPUTE:
            return BuildPick(
                IkBackend.CPU,
                f"The NVIDIA GPU {card.name} has compute capability {_version(card.compute)}, "
                f"and the CUDA build needs {_version(_CUDA_MIN_COMPUTE)} or newer.",
            )
        driver = host.nvidia_driver_cuda or (0, 0)
        return BuildPick(
            IkBackend.CPU,
            f"The NVIDIA driver offers CUDA {_version(driver)}, "
            f"and the CUDA build needs CUDA {_version(_CUDA_DRIVER_FLOOR)}. Update the driver.",
        )

    def binary(self) -> Path:
        """The server of the installed build of the pin; raises when the pack is not installed."""
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import install_advice

        backend = self.pick_build(session_host()).backend
        store = EngineStore(engines_root(cfg.models_dir))
        for build in store.installed():
            of_pin = (build.engine, build.version, build.backend) == (self.name, VERSION, backend)
            if of_pin and build.binary is not None:
                return build.binary
        raise ProviderError(
            f"Engine {self.name} is not installed. {install_advice(self)}",
            kind=ProviderErrorKind.NOT_FOUND,
        )

    def identity(self, binary: Path) -> str:
        return LLAMA.identity(binary)

    def pin_record(self) -> Mapping[str, Any]:
        from lilbee.catalog.runnable import session_host

        host = session_host()
        backend = self.pick_build(host).backend
        platforms = pack_platforms(self.name.value, VERSION, host, backend)
        return {
            "engine": self.name.value,
            "version": VERSION,
            "upstream_ref": IK_LLAMA_COMMIT,
            "backend": backend.value,
            "sha256": pack_digest(self.name.value, VERSION, host, backend),
            "tool": EngineTool.LLAMA_SERVER.value,
            "launch": _PIN_LAUNCH,
            "platforms": platforms,
            "memory": {"sizing": _PIN_SIZING, "rule": None},
            "licenses": _CUDA_LICENCES if backend is IkBackend.CUDA else _CPU_LICENCES,
        }

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        return LLAMA.role_spec(role, rerank_mode, meta)

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        return LLAMA.launch_argv(spec, request)

    def discover_devices(self, binary: Path) -> DeviceProbe:
        """An empty probe: this engine's server has no device listing to run."""
        log.debug("%s has no --list-devices option, so no device probe runs.", binary.name)
        return DeviceProbe([], "")

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
        return LLAMA.estimate_memory(dataclasses.replace(request, on_cpu=False))

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
