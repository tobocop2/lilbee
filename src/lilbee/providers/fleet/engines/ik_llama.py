"""The ik_llama.cpp adapter, composed from the llama.cpp adapter; opt-in, installed from a pack."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.adapters import RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest, RequestProfile
from lilbee.providers.fleet.engines.host_gate import HostGate, gate_on
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.host_caps import HostArch, HostCaps
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

# A commit of the upstream main branch, never "latest": a bump follows a passing harness run.
IK_LLAMA_COMMIT = "13cffc006b82e03e4d89cbb8c7327f0255ae3d18"
_VERSION_HEX_DIGITS = 12
VERSION = f"git-{IK_LLAMA_COMMIT[:_VERSION_HEX_DIGITS]}"
BACKEND = "cpu"
# The digest of the CPU pack; None until a pack is published, and an install is refused then.
PACK_SHA256: str | None = None
_REFUSAL_TAIL = "ik_llama.cpp needs a CPU with AVX2 or NEON. llama.cpp stays in use."
_PROFILE = dataclasses.replace(
    LLAMA.request_profile(WorkerRole.CHAT),
    label="ik_llama.cpp server",
    unreadable_embedding_hint="The ik_llama.cpp server answered embeddings that cannot be decoded.",
)


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
        return HostGate(False, f"{gate.reason}. {_REFUSAL_TAIL}")

    def binary(self) -> Path:
        """The server of the installed build of the pin; raises when the pack is not installed."""
        store = EngineStore(engines_root(cfg.models_dir))
        for build in store.installed():
            of_pin = (build.engine, build.version, build.backend) == (self.name, VERSION, BACKEND)
            if of_pin and build.binary is not None:
                return build.binary
        raise ProviderError(
            f"Engine {self.name} is not installed. Run lilbee engine install {self.name}.",
            kind=ProviderErrorKind.NOT_FOUND,
        )

    def identity(self, binary: Path) -> str:
        return LLAMA.identity(binary)

    def pin_record(self) -> Mapping[str, Any]:
        return {
            **LLAMA.pin_record(),
            "engine": self.name.value,
            "version": VERSION,
            "backend": BACKEND,
            "sha256": PACK_SHA256,
        }

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
