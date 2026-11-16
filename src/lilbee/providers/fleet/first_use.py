"""Install the llama.cpp pack the first time a model needs an engine and none is present."""

from __future__ import annotations

import logging
from pathlib import Path

from lilbee.catalog.download import RetryCallback
from lilbee.catalog.download_progress import ProgressCallback
from lilbee.catalog.runnable import session_host
from lilbee.core.config import cfg
from lilbee.core.security import FileLockUnavailableError, file_lock
from lilbee.providers.fleet.engine_manifest import UnsupportedOsError, cell_key, host_cell
from lilbee.providers.fleet.engine_store import (
    ChecksumError,
    EngineConflictError,
    EngineDiskError,
    EngineStore,
    LicenceRefusedError,
    MissingPackError,
    PackageInstallError,
    PackUrlError,
    PinLock,
    UnsafeWheelError,
    UnsupportedPlatformError,
    engines_root,
    fetch_frozen,
)
from lilbee.providers.fleet.engines.choice import pin_lock
from lilbee.providers.fleet.engines.llama import (
    LLAMA,
    cuda_pack_gap,
    pack_backend,
    rocm_needs_libnuma,
)
from lilbee.providers.fleet.gpu_select import PCIVendorID
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.gguf_meta import forget_unanswered_reads
from lilbee.runtime.cancellation import CancelSignal

log = logging.getLogger(__name__)

_FIRST_USE_LOCK_S = 1800.0
_UNSERVED_GPU_VENDORS = {PCIVendorID.AMD: "AMD", PCIVendorID.INTEL: "Intel"}
_TOOL_SUFFIXES = ("", ".exe")
_VULKAN_LOADER_ADVICE = {
    HostOS.LINUX: "install the Vulkan loader (libvulkan1 or vulkan-loader)",
    HostOS.WINDOWS: "install or update the GPU driver, which provides the Vulkan loader",
}
_ROCM_LIBRARY_ADVICE = "install the NUMA library (libnuma1 or numactl-libs)"


class PackNotPublishedError(Exception):
    """No pack of the llama.cpp engine is listed for this computer."""


INSTALL_REFUSALS = (
    PackNotPublishedError,
    ChecksumError,
    EngineConflictError,
    EngineDiskError,
    FileLockUnavailableError,
    LicenceRefusedError,
    MissingPackError,
    PackageInstallError,
    PackUrlError,
    UnsafeWheelError,
    UnsupportedOsError,
    UnsupportedPlatformError,
    OSError,
)
"""The errors that stop a pack install for a reason the user can act on."""


class RefusedInstalls:
    """The refusal of each first-use install this process made, so a lookup does not retry it."""

    def __init__(self) -> None:
        self._by_pack: dict[tuple[str, str, str], Exception] = {}

    def remember(self, key: tuple[str, str, str], refusal: Exception) -> None:
        self._by_pack[key] = refusal

    def forget(self, key: tuple[str, str, str]) -> None:
        self._by_pack.pop(key, None)

    def raise_if_refused(self, key: tuple[str, str, str]) -> None:
        if key in self._by_pack:
            raise self._by_pack[key]


_refusals = RefusedInstalls()


def _store() -> EngineStore:
    return EngineStore(engines_root(cfg.models_dir))


def pack_lock() -> tuple[PinLock, str]:
    """The lock of the llama.cpp pack for this computer, with its cell key.

    Raises ``PackNotPublishedError`` when the digest table has no row for the cell.
    """
    pin = LLAMA.pack_pin()
    if not pin["platforms"]:
        cell = cell_key(session_host(), pin["backend"]) or "this computer"
        raise PackNotPublishedError(
            f"The llama.cpp pack {pin['version']} for {cell} is not published yet."
        )
    lock = pin_lock(pin)
    cell = host_cell(lock, session_host())
    return lock.for_cell(cell), cell


def installed_tool(tool: str) -> Path | None:
    """The path of *tool* in the installed llama.cpp pack, or ``None``; never installs."""
    pin = LLAMA.pack_pin()
    pinned = (LLAMA.name.value, pin["version"], pin["backend"])
    for build in _store().installed():
        if (build.engine, build.version, build.backend) != pinned or build.binary is None:
            continue
        for suffix in _TOOL_SUFFIXES:
            found = build.binary.parent / f"{tool}{suffix}"
            if found.is_file():
                return found
    return None


def warn_if_gpu_unserved() -> None:
    """Warn once when the CPU pack installs on a computer with a GPU it cannot use.

    An NVIDIA card or driver below the CUDA pack gets its own warning; otherwise an AMD or
    Intel GPU does.
    """
    host = session_host()
    if pack_backend(host) != "cpu":
        return
    if gap := cuda_pack_gap(host):
        log.warning("%s", gap)
        return
    for vendor, name in _UNSERVED_GPU_VENDORS.items():
        if vendor in host.display_vendors:
            log.warning(
                "This computer has an %s GPU, and no llama.cpp pack can use it here. "
                "lilbee installs the CPU build. %sTo use your own build, set "
                "LILBEE_LLAMA_SERVER_PATH to your llama-server.",
                name,
                _pack_advice(host),
            )
            return


def _pack_advice(host: HostCaps) -> str:
    """The sentences that tell *host* how to get a GPU pack; empty when none exists."""
    advice = ""
    if rocm_needs_libnuma(host):
        advice = f"For the ROCm build, {_ROCM_LIBRARY_ADVICE}. "
    if host.arch == HostArch.X86_64 and host.os in _VULKAN_LOADER_ADVICE:
        advice += f"For the Vulkan build, {_VULKAN_LOADER_ADVICE[host.os]}. "
    return advice


def install_pack(
    wheel: Path | None = None,
    *,
    on_progress: ProgressCallback | None = None,
    on_retry: RetryCallback | None = None,
    cancel: CancelSignal | None = None,
) -> PinLock:
    """Install the llama.cpp pack once; a second caller waits and finds it installed.

    The download runs under one lock per engine, so two first uses fetch the wheel once.
    A pack copied into the packs folder by hand is used with no download. The warning for a
    GPU the CPU pack cannot use (NVIDIA below the CUDA pack, AMD or Intel) comes only with a
    download. *on_progress*
    hears the bytes, *on_retry* each failed attempt, and a set *cancel* ends the download.
    """
    lock, cell = pack_lock()
    store = _store()

    def download(pin: PinLock, scratch: Path) -> Path:
        warn_if_gpu_unserved()
        return fetch_frozen(
            pin,
            cell,
            dest=scratch,
            on_progress=on_progress,
            on_retry=on_retry,
            cancel=cancel,
        )

    with file_lock(store.root / ".locks" / f"first-use-{lock.engine}", _FIRST_USE_LOCK_S):
        if wheel is not None:
            store.install_from_wheel(lock, wheel)
        else:
            store.ensure(lock, download)
    forget_unanswered_reads()
    _refusals.forget((str(store.root), lock.version, cell))
    return lock


def first_use_tool(tool: str) -> Path | None:
    """The path of *tool* from the pack, installing the pack when it is absent.

    Raises ``PackNotPublishedError`` or a store refusal when the pack cannot be installed.
    A refused install is remembered for the rest of the process: each later lookup raises
    the same refusal and fetches nothing. ``install_pack`` always tries again.
    """
    found = installed_tool(tool)
    if found is not None:
        return found
    lock, cell = pack_lock()
    key = (str(_store().root), lock.version, cell)
    _refusals.raise_if_refused(key)
    try:
        install_pack()
    except INSTALL_REFUSALS as exc:
        _refusals.remember(key, exc)
        raise
    return installed_tool(tool)
