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
from lilbee.providers.fleet.engines.llama import LLAMA, pack_backend
from lilbee.providers.fleet.gpu_hardware import installed_gpu_vendor_ids
from lilbee.providers.fleet.gpu_select import PCIVendorID
from lilbee.providers.gguf_meta import forget_unanswered_reads
from lilbee.runtime.cancellation import CancelSignal

log = logging.getLogger(__name__)

_FIRST_USE_LOCK_S = 1800.0
_UNSERVED_GPU_VENDORS = {PCIVendorID.AMD: "AMD", PCIVendorID.INTEL: "Intel"}
_TOOL_SUFFIXES = ("", ".exe")


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
    """Warn when the CPU pack installs on a computer with an AMD or Intel GPU."""
    if pack_backend(session_host()) != "cpu":
        return
    present = installed_gpu_vendor_ids()
    for vendor, name in _UNSERVED_GPU_VENDORS.items():
        if vendor in present:
            log.warning(
                "This computer has an %s GPU, and no llama.cpp pack uses it. "
                "lilbee installs the CPU build. For a GPU build, use a bundled build "
                "or set LILBEE_LLAMA_SERVER_PATH to your own llama-server.",
                name,
            )
            return


def install_pack(
    wheel: Path | None = None,
    *,
    on_progress: ProgressCallback | None = None,
    on_retry: RetryCallback | None = None,
    cancel: CancelSignal | None = None,
) -> PinLock:
    """Install the llama.cpp pack once; a second caller waits and finds it installed.

    The download runs under one lock per engine, so two first uses fetch the wheel once.
    A pack copied into the packs folder by hand is used with no download. *on_progress*
    hears the bytes, *on_retry* each failed attempt, and a set *cancel* ends the download.
    """
    lock, cell = pack_lock()
    warn_if_gpu_unserved()
    store = _store()
    with file_lock(store.root / ".locks" / f"first-use-{lock.engine}", _FIRST_USE_LOCK_S):
        if wheel is not None:
            store.install_from_wheel(lock, wheel)
        else:
            store.ensure(
                lock,
                lambda pin, scratch: fetch_frozen(
                    pin,
                    cell,
                    dest=scratch,
                    on_progress=on_progress,
                    on_retry=on_retry,
                    cancel=cancel,
                ),
            )
    forget_unanswered_reads()
    return lock


def first_use_tool(tool: str) -> Path | None:
    """The path of *tool* from the pack, installing the pack when it is absent.

    Raises ``PackNotPublishedError`` or a store refusal when the pack cannot be installed.
    """
    found = installed_tool(tool)
    if found is not None:
        return found
    install_pack()
    return installed_tool(tool)
