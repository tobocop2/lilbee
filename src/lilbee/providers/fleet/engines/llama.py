"""The llama.cpp adapter; the only module that names llama.cpp's tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from lilbee._generated.engine_archs import ENGINE_LLAMA_CPP_REF, SUPPORTED_ARCHS
from lilbee.catalog.header_probe import GGUF_MAGIC
from lilbee.catalog.hf_client import HfToken
from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.adapters import (
    FLAG_DEVICE,
    FLAG_FIT_TARGET,
    FLAG_KV_UNIFIED_PER_SLOT,
    NO_DEVICE,
    ROLE_SPECS,
    RoleServerSpec,
    build_server_argv,
    embed_spec,
    rerank_spec,
)
from lilbee.providers.fleet.binary import (
    EngineTool,
    engine_binary_identity,
    llama_server_runtime_env,
    resolve_llama_server,
)
from lilbee.providers.fleet.devices import DeviceProbe, probe_devices
from lilbee.providers.fleet.engines.base import (
    EngineName,
    LaunchRequest,
    RequestProfile,
    ServerProcess,
)
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.gpu_select import PCIVendorID
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.loadability import assert_engine_can_load
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import estimate_gguf_cpu_decode, estimate_memory
from lilbee.providers.fleet.readback import engine_advertises, supports_memory_readback
from lilbee.providers.gguf_meta import read_gguf_metadata
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

_REQUEST_TOKENS = 2048
"""llama-server runs each input of an embed request as its own task, so --batch-size bounds
one input and not the request. A request of about 2000 tokens ran on a 648-token batch."""
_PROFILE = RequestProfile(
    request_tokens=_REQUEST_TOKENS,
    embed_fields={"embd_normalize": -1},
    stream_fields={"return_progress": True},
    takes_cache_prompt=True,
    takes_seed=True,
    tokenize_path="/tokenize",
    detokenize_path="/detokenize",
    apply_template_path="/apply-template",
    label="llama-server",
    context_overflow=re.compile("exceed_context_size|context size", re.IGNORECASE),
    batch_overflow=re.compile("too large to process"),
    unreadable_embedding_hint="base64 embedding responses need llama-server b4391 or newer.",
    reasoning_field="reasoning_content",
)
_PIN_LAUNCH = ["{binary}", "--model", "{model}", "--host", "127.0.0.1", "--port", "{port}"]
_PIN_LIBRARIES = ["ggml-base"]
_PIN_SIZING = ["gguf-parser", "--model", "{model}"]
_CPU_PACK = "cpu"
# The CUDA pack is built with CUDA 12.4: it needs a driver that offers 12.4 and a card of
# compute capability 5.0 or newer (the oldest architecture the pack targets).
# A test reads the pack build files and fails when they stop agreeing with these two.
_CUDA_DRIVER_FLOOR = (12, 4)
_CUDA_MIN_COMPUTE = (5, 0)
_CUDA_BACKEND = "cuda"
_CUDA_HOSTS = {
    (HostOS.LINUX, HostArch.X86_64),
    (HostOS.WINDOWS, HostArch.X86_64),
}
_VULKAN_VENDORS = frozenset({PCIVendorID.AMD, PCIVendorID.INTEL})
_VULKAN_BACKEND = "vulkan"
_ROCM_BACKEND = "rocm"
# The gfx targets that the ROCm pack carries device code and GEMM kernels for.
# A test reads the pack build files and fails when they stop agreeing with this set.
_ROCM_GFX = frozenset(
    {
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
    }
)
_PACK_LICENCES = ["MIT"]
_UNIX_PACK_TERMS = (
    "MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND MIT-0 AND Unlicense"
)
_WINDOWS_PACK_TERMS = (
    "MIT AND Apache-2.0 WITH LLVM-exception AND LicenseRef-Microsoft-VC-Runtime AND Apache-2.0 "
    "AND BSD-2-Clause AND BSD-3-Clause AND ISC AND MIT-0 AND Unlicense"
)
# Each list is the License-Expression of the published wheel of that cell.
_CELL_LICENCES = {
    "linux-x86_64-cpu": [_UNIX_PACK_TERMS],
    "linux-x86_64-vulkan": [_UNIX_PACK_TERMS],
    "macos-arm64-metal": [_UNIX_PACK_TERMS],
    "linux-x86_64-cuda": [
        "MIT AND LicenseRef-NVIDIA-CUDA-EULA AND Apache-2.0 AND Apache-2.0 WITH LLVM-exception "
        "AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT-0 AND Unlicense"
    ],
    "linux-x86_64-rocm": [
        "MIT AND BSD-3-Clause AND NCSA AND Apache-2.0 WITH LLVM-exception AND Apache-2.0 "
        "AND BSD-2-Clause AND ISC AND MIT-0 AND Unlicense"
    ],
    "windows-x86_64-cpu": [_WINDOWS_PACK_TERMS],
    "windows-x86_64-vulkan": [_WINDOWS_PACK_TERMS],
    "windows-x86_64-cuda": [
        "MIT AND LicenseRef-NVIDIA-CUDA-EULA AND Apache-2.0 WITH LLVM-exception "
        "AND LicenseRef-Microsoft-VC-Runtime AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause "
        "AND BSL-1.0 AND ISC AND MIT-0 AND Unlicense"
    ],
}
# The pack build names the wheel 1.<N>.<release build> for the upstream tag b<N>.
_UPSTREAM_TAG = "b12032"
_WHEEL_VERSION = f"1.{_UPSTREAM_TAG.removeprefix('b')}.1"


def has_usable_cuda(
    host: HostCaps, driver_floor: tuple[int, int], min_compute: tuple[int, int]
) -> bool:
    """Whether the NVIDIA driver and a card of *host* can run a CUDA pack with these floors."""
    if not host.nvidia_known or not host.gpus:
        return False
    if (host.nvidia_driver_cuda or (0, 0)) < driver_floor:
        return False
    return max(gpu.compute for gpu in host.gpus) >= min_compute


def _has_rocm_gpu(host: HostCaps) -> bool:
    """Whether *host* has an AMD GPU that the ROCm pack carries device code for."""
    return any(g in _ROCM_GFX for g in host.amd_gfx)


def _has_usable_rocm(host: HostCaps) -> bool:
    """Whether *host* is Linux with the NUMA library and an AMD GPU that the ROCm pack serves."""
    return host.os is HostOS.LINUX and host.libnuma and _has_rocm_gpu(host)


def rocm_needs_libnuma(host: HostCaps) -> bool:
    """Whether the ROCm pack would serve *host* if it had the NUMA library."""
    return (
        (host.os, host.arch) == (HostOS.LINUX, HostArch.X86_64)
        and not host.libnuma
        and _has_rocm_gpu(host)
    )


def cuda_pack_gap(host: HostCaps) -> str:
    """Why the CUDA pack cannot serve the NVIDIA card of *host*; empty when it can."""
    if (host.os, host.arch) not in _CUDA_HOSTS or not host.nvidia_known or not host.gpus:
        return ""
    if has_usable_cuda(host, _CUDA_DRIVER_FLOOR, _CUDA_MIN_COMPUTE):
        return ""
    card = max(host.gpus, key=lambda gpu: gpu.compute)
    compute = f"{card.compute[0]}.{card.compute[1]}"
    need = f"{_CUDA_MIN_COMPUTE[0]}.{_CUDA_MIN_COMPUTE[1]}"
    floor = f"{_CUDA_DRIVER_FLOOR[0]}.{_CUDA_DRIVER_FLOOR[1]}"
    if card.compute < _CUDA_MIN_COMPUTE:
        return (
            f"The NVIDIA GPU {card.name} has compute capability {compute}, and the llama.cpp "
            f"CUDA build supports {need} or newer. lilbee installs the CPU build."
        )
    # A host that knows its GPUs knows the driver's CUDA version too (host_caps.collect).
    driver_cuda = host.nvidia_driver_cuda or (0, 0)
    driver = f"{driver_cuda[0]}.{driver_cuda[1]}"
    return (
        f"The NVIDIA driver for {card.name} offers CUDA {driver}, and the llama.cpp CUDA "
        f"build needs CUDA {floor}. Update the driver. lilbee installs the CPU build."
    )


def pack_backend(host: HostCaps) -> str:
    """The backend of the llama.cpp pack that *host* installs."""
    if (host.os, host.arch) == (HostOS.MACOS, HostArch.ARM64):
        return "metal"
    if (host.os, host.arch) not in _CUDA_HOSTS:
        return _CPU_PACK
    if has_usable_cuda(host, _CUDA_DRIVER_FLOOR, _CUDA_MIN_COMPUTE):
        return _CUDA_BACKEND
    if _has_usable_rocm(host):
        return _ROCM_BACKEND
    if host.vulkan_loader and host.display_vendors & _VULKAN_VENDORS:
        return _VULKAN_BACKEND
    return _CPU_PACK


def starts_with_gguf_magic(path: Path) -> bool:
    """Whether the file opens with the GGUF magic; the registry stores blobs with no suffix."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(GGUF_MAGIC)) == GGUF_MAGIC
    except OSError:
        return False


class LlamaCppAdapter:
    """llama-server behind the engine contract."""

    name = EngineName.LLAMA
    formats = frozenset({ModelFormat.GGUF})
    architectures = frozenset(SUPPORTED_ARCHS)
    served_roles = frozenset(role for role, spec in ROLE_SPECS.items() if spec.server_capable)
    memory_kind = MemoryKind.SIZED
    takes_device_names = True
    concatenates_pooling = False
    bundled = True
    server_process = ServerProcess(name="llama-server")

    def supports_architecture(self, architecture: str) -> bool:
        return architecture in self.architectures

    def host_gate(self, host: HostCaps) -> HostGate:
        return PASS

    def binary(self) -> Path:
        return resolve_llama_server()

    def identity(self, binary: Path) -> str:
        return engine_binary_identity(binary)

    def pin_record(self) -> Mapping[str, Any]:
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engine_manifest import cell_key, pack_platforms

        host = session_host()
        backend = pack_backend(host)
        platforms = pack_platforms(self.name.value, _WHEEL_VERSION, host, backend)
        return {
            "engine": self.name.value,
            "version": _WHEEL_VERSION,
            "upstream_ref": ENGINE_LLAMA_CPP_REF,
            "backend": None,
            "tool": EngineTool.LLAMA_SERVER.value,
            "libraries": _PIN_LIBRARIES,
            "launch": _PIN_LAUNCH,
            "platforms": platforms,
            "memory": {"sizing": _PIN_SIZING, "rule": None},
            "licenses": _CELL_LICENCES.get(cell_key(host, backend) or "", _PACK_LICENCES),
        }

    def pack_pin(self) -> Mapping[str, Any]:
        """The pin of the llama.cpp pack for the first-use install; its backend follows the host."""
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engine_manifest import pack_digest

        host = session_host()
        backend = pack_backend(host)
        pin = dict(self.pin_record())
        pin.update(
            backend=backend,
            sha256=pack_digest(self.name.value, _WHEEL_VERSION, host, backend),
        )
        return pin

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role is WorkerRole.RERANK and rerank_mode is not None:
            return rerank_spec(rerank_mode)
        if role is WorkerRole.EMBED:
            return embed_spec(meta)
        return ROLE_SPECS[role]

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        if request.target is RoleTarget.CPU:
            # Zero layers alone still opens the card (424 MiB measured), so the
            # adapter pairs them with the no-device name whatever the caller sent.
            request = replace(request, n_gpu_layers=0, fit_target_mib=(), device_names=(NO_DEVICE,))
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

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        """A build whose help lists no device flag keeps the card, so it has no cpu target."""
        if engine_advertises(binary, FLAG_DEVICE):
            return frozenset({RoleTarget.GPU, RoleTarget.CPU})
        return frozenset({RoleTarget.GPU})

    def model_meta(self, path: Path) -> dict[str, str] | None:
        if not starts_with_gguf_magic(path):
            return None
        return read_gguf_metadata(path)

    def assert_loadable(self, repo: str, filename: str, token: HfToken) -> None:
        assert_engine_can_load(repo, filename, token)

    def runtime_env(self) -> Mapping[str, str]:
        return llama_server_runtime_env()

    def target_env(self, target: RoleTarget, cards: int) -> Mapping[str, str]:
        return {}

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        if request.on_cpu and request.batch_size is None:
            return estimate_gguf_cpu_decode(request)
        return estimate_memory(ModelFormat.GGUF, request)

    def check_readback(self, binary: Path) -> bool:
        return supports_memory_readback(binary)

    def supports_fit_target(self, binary: Path) -> bool:
        return engine_advertises(binary, FLAG_FIT_TARGET)

    def supports_shared_kv(self, binary: Path) -> bool:
        return engine_advertises(binary, FLAG_KV_UNIFIED_PER_SLOT)

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


LLAMA = LlamaCppAdapter()
ADAPTER = LLAMA
