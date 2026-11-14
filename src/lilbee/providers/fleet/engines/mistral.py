"""The mistral.rs adapter; the only module that names mistral.rs's tools."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from lilbee.catalog.types import ModelFormat
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.adapters import ROLE_SPECS, RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe, visible_nvidia_cards
from lilbee.providers.fleet.engine_manifest import pack_digest, pack_platforms
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    RequestProfile,
)
from lilbee.providers.fleet.engines.host_gate import PASS, PLATFORM_UNREAD, HostGate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import UNKNOWN_SIZE_RESERVE_BYTES, estimate_memory
from lilbee.providers.gguf_meta import read_gguf_metadata
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

# The flag that gives each visible card a count of layers; the engine runs equal
# tensor-parallel shards instead of that map unless its collective library is off.
_DEVICE_LAYERS = "--device-layers"
_MIN_SPLIT_CARDS = 2
_NO_COLLECTIVE_ENV = "MISTRALRS_NO_NCCL"
_CONFIG = "config.json"
_LAYERS_KEY = "num_hidden_layers"
_TEXT_CONFIG_KEY = "text_config"
_GGUF_LAYERS_KEY = "block_count"
_VERSION = "0.9.5"
_TOOL = "mistralrs"
_HOST = "127.0.0.1"
_LICENCES = ["MIT"]
# The expression a CUDA pack declares: it bundles NVIDIA runtime libraries.
_CUDA_LICENCES = ["MIT AND LicenseRef-NVIDIA-CUDA-EULA AND BSD-3-Clause"]
_CPU = "cpu"
# The one id the engine serves a model under; it answers 400 to any other.
_SERVED_NAME = "default"
_GGUF_SUFFIX = ".gguf"
# The model subcommand each role names; the device flags go after it.
_SUBCOMMAND = {
    WorkerRole.EMBED: "embedding",
    WorkerRole.VISION: "multimodal",
}
_TEXT = "text"
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


def _positive_int(value: object) -> int | None:
    """*value* as a whole number above zero, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        return None
    try:
        count = int(value)
    except ValueError:
        return None
    return count if count > 0 else None


def _layer_count(path: Path) -> int | None:
    """The layer count a GGUF file or a model folder declares, or ``None``."""
    if path.suffix.lower() == _GGUF_SUFFIX:
        meta = read_gguf_metadata(path)
        return _positive_int(meta.get(_GGUF_LAYERS_KEY)) if meta else None
    try:
        config = json.loads((path / _CONFIG).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    for section in (config, config.get(_TEXT_CONFIG_KEY)):
        count = _positive_int(section.get(_LAYERS_KEY)) if isinstance(section, dict) else None
        if count is not None:
            return count
    return None


def _apportion(layers: int, ratio: tuple[int, ...]) -> tuple[int, ...] | None:
    """Layers per card by largest remainder; ``None`` when a card gets none."""
    total = sum(ratio)
    if total <= 0:
        return None
    shares = [layers * part // total for part in ratio]
    by_remainder = sorted(range(len(ratio)), key=lambda i: (-(layers * ratio[i] % total), i))
    for i in by_remainder[: layers - sum(shares)]:
        shares[i] += 1
    return tuple(shares) if min(shares) > 0 else None


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
            return PLATFORM_UNREAD
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

        host = session_host()
        backend = self.pick_build(host).backend or _CPU
        platforms = pack_platforms(self.name.value, _VERSION, host, backend)
        return {
            "engine": self.name.value,
            "version": _VERSION,
            "backend": backend,
            "sha256": pack_digest(self.name.value, _VERSION, host, backend),
            "tool": _TOOL,
            "launch": [_TOOL, "serve"],
            "platforms": platforms,
            "memory": {"sizing": None, "rule": MemoryKind.FIXED_RESERVE.value},
            "licenses": _LICENCES if backend == _CPU else _CUDA_LICENCES,
        }

    def binary(self) -> Path:
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import store_tool

        tool = store_tool(self.pin_record(), _TOOL, session_host())
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
        """The serve command; context stays unset, and several cards get a layer map.

        A cpu target gets ``--cpu`` on any build. Otherwise the backend of the build
        decides the device flags: a CPU build gets ``--cpu``, and
        any GPU build turns paged attention off, because its cache takes the whole card
        and the first chat then answers 503. The port flag
        belongs to ``serve``, so it sits before the model subcommand. The device flags
        belong to the subcommand: before it the engine ignores ``--cpu`` and takes the
        whole card (20536 MiB measured), so every role names its subcommand. A GGUF file
        goes by ``-f`` with its folder as ``-m``; how the engine takes a folder with
        several quantizations is open for slice 8.
        """
        argv = [str(request.binary), "serve", "--host", _HOST, "--port", PORT_SLOT]
        argv.append(_SUBCOMMAND.get(spec.role, _TEXT))
        argv += self._device_flags(request.target)
        argv += self._split_flags(request)
        path = request.model_path
        if path.suffix.lower() == _GGUF_SUFFIX:
            return [*argv, "-m", str(path.parent), "-f", path.name]
        return [*argv, "-m", str(path)]

    def _device_flags(self, target: RoleTarget) -> list[str]:
        from lilbee.catalog.runnable import session_host

        if target is RoleTarget.CPU or self.pick_build(session_host()).backend in (None, _CPU):
            return ["--cpu"]
        return ["--paged-attn", "off"]

    def _split_flags(self, request: LaunchRequest) -> list[str]:
        """The layer map for a split over several cards; it refuses a split it cannot run."""
        ratio = request.tensor_split
        if request.target is RoleTarget.CPU or len(ratio) < _MIN_SPLIT_CARDS:
            return []
        from lilbee.catalog.runnable import session_host

        name = request.model_path.name
        cards = len(ratio)
        backend = self.pick_build(session_host()).backend
        if backend in (None, _CPU, _METAL):
            label = "Metal" if backend == _METAL else "CPU"
            raise ProviderError(
                f"mistral.rs cannot split {name} across {cards} cards on the {label} build. "
                "Give the role one card."
            )
        layers = _layer_count(request.model_path)
        if layers is None:
            raise ProviderError(
                f"mistral.rs cannot split {name} across {cards} cards: "
                "the model does not list its layer count. Give the role one card."
            )
        shares = _apportion(layers, ratio)
        if shares is None:
            raise ProviderError(
                f"mistral.rs cannot split {name} across {cards} cards: "
                "the split leaves a card with no layer. Give the role fewer cards."
            )
        return [_DEVICE_LAYERS, ";".join(f"{card}:{count}" for card, count in enumerate(shares))]

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        from lilbee.catalog.runnable import session_host

        if self.pick_build(session_host()).backend in (None, _CPU):
            return frozenset({RoleTarget.CPU})
        return frozenset({RoleTarget.CPU, RoleTarget.GPU})

    def discover_devices(self, binary: Path) -> DeviceProbe:
        from lilbee.catalog.runnable import session_host

        if self.pick_build(session_host()).backend in (None, _CPU):
            return DeviceProbe(devices=[], output="")
        return DeviceProbe(devices=visible_nvidia_cards(), output="")

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        whole = self._whole_estimate(request)
        if len(request.tensor_split) < _MIN_SPLIT_CARDS:
            return whole
        layers = _layer_count(request.model_path)
        if layers is None:
            return whole
        shares = _apportion(layers, request.tensor_split)
        if shares is None:
            return whole
        per_device = tuple(whole.device_bytes * count // layers for count in shares)
        return replace(whole, per_device=per_device)

    @staticmethod
    def _whole_estimate(request: MemoryRequest) -> MemoryEstimate:
        if request.model_path.is_dir():
            return estimate_memory(ModelFormat.SAFETENSORS, request)
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
        return {_NO_COLLECTIVE_ENV: "1"}

    def target_env(self, target: RoleTarget) -> Mapping[str, str]:
        return {}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {"useModelName": _SERVED_NAME}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


MISTRAL = MistralAdapter()
ADAPTER = MISTRAL
