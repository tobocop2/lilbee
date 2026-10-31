"""The mistral.rs adapter; the only module that names mistral.rs's tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.adapters import ROLE_SPECS, RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    RequestProfile,
)
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import UNKNOWN_SIZE_RESERVE_BYTES
from lilbee.providers.roles import RerankMode, WorkerRole

_VERSION = "0.9.5"
_TOOL = "mistralrs"
_HOST = "127.0.0.1"
_LICENCES = ["MIT"]
# The digest of the pack; None until a pack is published, and an install is refused then.
PACK_SHA256: str | None = None
_CPU = "cpu"
# The one id the engine serves a model under; it answers 400 to any other.
_SERVED_NAME = "default"
_GGUF_SUFFIX = ".gguf"
_METAL = "metal"
_CUDA_DRIVER_FLOOR = (12, 8)
# The compute capabilities the release publishes a CUDA 12.8 archive for.
_CUDA_ARCHIVES = frozenset({(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)})
_SUPPORTED_HOSTS = frozenset(
    {
        (HostOS.LINUX, HostArch.X86_64),
        (HostOS.MACOS, HostArch.ARM64),
        (HostOS.WINDOWS, HostArch.X86_64),
    }
)

_PROFILE = RequestProfile(
    embed_fields={},
    stream_fields={},
    tokenize_path=None,
    detokenize_path=None,
    apply_template_path="/apply-template",
    label="mistral.rs",
    context_overflow=re.compile("Prompt sequence length is greater than", re.IGNORECASE),
    batch_overflow=re.compile(r"(?!)"),
    unreadable_embedding_hint="mistral.rs answers embeddings as lists of floats.",
    reasoning_field="reasoning_content",
)


@dataclass(frozen=True)
class BuildPick:
    """The archive for one host: its backend name, or ``None`` with the reason it has none."""

    backend: str | None
    reason: str


def _compute_text(compute: tuple[int, int]) -> str:
    return f"{compute[0]}.{compute[1]}"


def _linux_pick(host: HostCaps) -> BuildPick:
    """The CUDA archive of the host's GPUs, else the CPU archive with the reason."""
    if not host.nvidia_known:
        return BuildPick(_CPU, f"{host.nvidia_reason}, so the CPU archive runs.")
    if not host.gpus:
        return BuildPick(_CPU, "This host has no NVIDIA GPU, so the CPU archive runs.")
    computes = {gpu.compute for gpu in host.gpus}
    if len(computes) > 1:
        return BuildPick(_CPU, "The GPUs differ in compute capability, so the CPU archive runs.")
    compute = next(iter(computes))
    if compute not in _CUDA_ARCHIVES:
        return BuildPick(
            _CPU,
            f"No CUDA archive targets compute capability {_compute_text(compute)}, "
            "so the CPU archive runs.",
        )
    driver = host.nvidia_driver_cuda or (0, 0)
    if driver < _CUDA_DRIVER_FLOOR:
        return BuildPick(
            _CPU,
            f"The CUDA archive needs a driver for CUDA {_compute_text(_CUDA_DRIVER_FLOOR)}; "
            f"this driver offers CUDA {_compute_text(driver)}, so the CPU archive runs.",
        )
    return BuildPick(f"cuda128-sm{compute[0]}{compute[1]}", "")


class MistralAdapter:
    """mistral.rs behind the engine contract; it serves chat, embed and vision, never rerank."""

    name = EngineName.MISTRAL
    formats = frozenset({ModelFormat.GGUF, ModelFormat.SAFETENSORS})
    architectures: frozenset[str] = frozenset()
    served_roles = frozenset({WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION})
    memory_kind = MemoryKind.FIXED_RESERVE
    takes_device_names = False
    bundled = False

    def supports_architecture(self, architecture: str) -> bool:
        return True

    def host_gate(self, host: HostCaps) -> HostGate:
        if host.os is None or host.arch is None:
            return HostGate(
                True, "The OS or CPU architecture of this host could not be read.", known=False
            )
        if (host.os, host.arch) in _SUPPORTED_HOSTS:
            reason = self._archive(host, host.os).reason
            return HostGate(True, reason) if reason else PASS
        return HostGate(False, f"mistral.rs is not offered on {host.os.value} {host.arch.value}.")

    def pick_build(self, host: HostCaps) -> BuildPick:
        """The archive that runs on *host*; ``backend`` is ``None`` when the gate refuses it."""
        gate = self.host_gate(host)
        if not gate.ok or host.os is None:
            return BuildPick(None, gate.reason)
        return self._archive(host, host.os)

    @staticmethod
    def _archive(host: HostCaps, os: HostOS) -> BuildPick:
        if os is HostOS.MACOS:
            return BuildPick(_METAL, "")
        if os is HostOS.WINDOWS:
            return BuildPick(_CPU, "The release has no CUDA archive for Windows.")
        return _linux_pick(host)

    def pin_record(self) -> Mapping[str, Any]:
        from lilbee.catalog.runnable import session_host

        backend = self.pick_build(session_host()).backend or _CPU
        return {
            "engine": self.name.value,
            "version": _VERSION,
            "backend": backend,
            "sha256": PACK_SHA256,
            "launch": [_TOOL, "serve"],
            "platforms": {},
            "memory": {"sizing": None, "rule": MemoryKind.FIXED_RESERVE.value},
            "licenses": _LICENCES,
        }

    def binary(self) -> Path:
        from lilbee.providers.fleet.engines.choice import NoPublishedPackError, pin_lock

        store = EngineStore(engines_root(cfg.models_dir))
        try:
            build = store.complete_build(pin_lock(self.pin_record()))
        except NoPublishedPackError:
            build = None
        tool = next(iter(build.glob(f"*/bin/{_TOOL}*")), None) if build else None
        if tool is None:
            raise ProviderError(
                "The mistral.rs engine is not installed. Run lilbee engine install mistral.rs."
            )
        return tool

    def identity(self, binary: Path) -> str:
        return f"{self.name.value} {_VERSION} {binary}"

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role not in self.served_roles:
            raise ProviderError(f"Engine {self.name.value} does not serve {role}.")
        return RoleServerSpec(role, ROLE_SPECS[role].endpoint_path, (), server_capable=True)

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        """The serve command; the engine places the model itself, so layers and context stay unset.

        The backend of the build decides the device flags: a CPU build gets ``--cpu``, and
        any GPU build turns paged attention off, because its cache takes the whole card
        and the first chat then answers 503. The port flag
        belongs to ``serve``, so it sits before the model subcommand. A GGUF file
        goes by ``-f`` with its folder as ``-m``; how the engine takes a folder with
        several quantizations is open for slice 8.
        """
        argv = [str(request.binary), "serve", "--host", _HOST, "--port", PORT_SLOT]
        if spec.role is WorkerRole.EMBED:
            argv.append("embedding")
        argv += self._device_flags()
        path = request.model_path
        if path.suffix.lower() == _GGUF_SUFFIX:
            return [*argv, "-m", str(path.parent), "-f", path.name]
        return [*argv, "-m", str(path)]

    def _device_flags(self) -> list[str]:
        from lilbee.catalog.runnable import session_host

        if self.pick_build(session_host()).backend in (None, _CPU):
            return ["--cpu"]
        return ["--paged-attn", "off"]

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(devices=[], output="")

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        try:
            weights = request.model_path.stat().st_size
        except OSError:
            weights = 0
        weights = weights if weights > 0 else UNKNOWN_SIZE_RESERVE_BYTES
        return MemoryEstimate(
            kind=MemoryKind.FIXED_RESERVE, device_bytes=weights, host_bytes=0, shared_bytes=weights
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
        return {"useModelName": _SERVED_NAME}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


MISTRAL = MistralAdapter()
