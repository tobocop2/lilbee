"""Surface-agnostic use-cases for choosing the engine of a model."""

from __future__ import annotations

import json
import logging
import os
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from lilbee.app.engine_choice import (
    CHOICES_KEY,
    engine_for_format,
    resolve_engines,
    stored_choices,
)
from lilbee.app.services import Services, _new_store, peek_services
from lilbee.catalog import runnable
from lilbee.catalog.download import retry_reporter
from lilbee.catalog.download_progress import make_download_callback
from lilbee.catalog.models import DownloadProgress
from lilbee.catalog.refs import canonical_ref, model_format
from lilbee.catalog.runnable import refresh_session_host
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.binary import llama_server_ready
from lilbee.providers.fleet.engine_manifest import UnsupportedOsError, cell_key, host_cell
from lilbee.providers.fleet.engine_store import (
    BundledEngineError,
    ChecksumError,
    EngineConflictError,
    EngineDiskError,
    EngineFolderError,
    EngineInUseError,
    EngineNotInstalledError,
    EngineStore,
    LicenceRefusedError,
    MissingPackError,
    PackUrlError,
    StorePathError,
    UnsafeWheelError,
    UnsupportedPlatformError,
    engines_root,
    fetch_frozen,
    launch_uses_engine,
)
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineAdapter, EngineName, UserInstalled
from lilbee.providers.fleet.engines.choice import (
    AUTO,
    LIST_HINT,
    InstalledEngines,
    engines_loading,
    installed_engines,
)
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.engines.llama import LLAMA, pack_backend
from lilbee.providers.fleet.engines.proof import experimental_clause, experimental_devices
from lilbee.providers.fleet.router_install import ROUTER_REFUSALS, install_router
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole, unserved_role_message
from lilbee.retrieval.canary_probe import embedder_trigger
from lilbee.runtime.cancellation import CancelSignal

log = logging.getLogger(__name__)
_T = TypeVar("_T")

STORE_REFUSALS = (
    BundledEngineError,
    EngineFolderError,
    EngineInUseError,
    EngineNotInstalledError,
    StorePathError,
)
NO_BUILDS = "No engine builds are installed."
DIFFERS_FROM_PIN_TEXT = "differs from the pin"
"""The words for a build on disk that is not the build the pin names; every surface shows them."""
FOLDER_LINE = (
    "Engines folder: {folder} (shared by every data directory; LILBEE_MODELS_DIR moves it)"
)
REMOVE_DONE = "Removed engine {build}."
PREFERRED_KEY = "preferred_engine"
NOT_APPLIED_CODE = "engine_not_applied"


class EngineRefusedError(ValueError):
    """The engine service refuses a write; the text names the problem and the fix."""


class EngineInstallRefusedError(EngineRefusedError):
    """The install of a named engine is refused; every surface sends the same code."""


class EngineNotAppliedError(Exception):
    """The choice is saved but the running server could not apply it; a fault, not a refusal."""


class EngineNotSavedError(Exception):
    """The choice could not be written to the settings file; the text holds no path."""


class EmbedConfirmRequiredError(Exception):
    """The change can make vectors the index cannot use; the caller confirms or drops it."""

    def __init__(self, library: str, documents: int) -> None:
        self.library = library
        self.documents = documents
        super().__init__(
            f"This change can make the vectors of library {library!r} ({documents} documents) "
            "unusable by the index. Confirm it, or keep the current engine."
        )


WRITE_FAILURES = (
    EngineRefusedError,
    EngineNotSavedError,
    EngineNotAppliedError,
    EmbedConfirmRequiredError,
)
CONFIRM_CODE = "embed_engine_confirm_required"


@dataclass(frozen=True)
class RoleView:
    """What a surface shows of one role: its model, the engine it runs on and why."""

    role: WorkerRole
    model: str
    format: str
    engine: str | None
    notice: str | None


@dataclass(frozen=True)
class InstalledBuild:
    """The engine build an install wrote."""

    engine: str
    version: str
    backend: str
    note: str = ""

    @property
    def not_run_on(self) -> tuple[str, ...]:
        """The devices of this build's cell that have no record."""
        return BuildView(self.engine, self.version, self.backend).not_run_on

    @property
    def plain_label(self) -> str:
        """The build as ``name version (backend)``, the form that JSON output carries."""
        return BuildView(self.engine, self.version, self.backend).plain_label

    @property
    def label(self) -> str:
        """The build as one line, with the devices it has not run on."""
        return BuildView(self.engine, self.version, self.backend).label


_STORE_REFUSALS = (
    ChecksumError,
    EngineConflictError,
    EngineDiskError,
    EngineInUseError,
    LicenceRefusedError,
    MissingPackError,
    PackUrlError,
    UnsafeWheelError,
    UnsupportedOsError,
    UnsupportedPlatformError,
)


# What the zip reader raises for a pinned file the store cannot read: a bad archive,
# METADATA that is not UTF-8, an encrypted member, a compression method it lacks.
_UNREADABLE_WHEEL = (
    OSError,
    zipfile.BadZipFile,
    UnicodeDecodeError,
    RuntimeError,
    NotImplementedError,
)


def _unknown(name: str) -> EngineRefusedError:
    known = ", ".join([AUTO, *EngineName])
    return EngineRefusedError(f"No engine is named {name!r}. Choose one of: {known}.")


def _not_installable(name: str) -> EngineRefusedError:
    """The refusal for a name install cannot take; the list leaves out auto."""
    known = ", ".join(EngineName)
    if name == AUTO:
        return EngineRefusedError(
            f"{AUTO!r} picks an engine for you and cannot be installed. Choose one of: {known}."
        )
    return EngineRefusedError(f"No engine is named {name!r}. Choose one of: {known}.")


def _engine_name(name: str) -> EngineName:
    try:
        return EngineName(name)
    except ValueError:
        raise _unknown(name) from None


def _require_installed(name: EngineName, installed: InstalledEngines) -> None:
    """Refuse a name that is not among the *installed* engines the host can run.

    An engine the host gate refuses gets the gate's reason, with a build or without.
    The engine list hides such an engine, so the list is no help to the user.
    """
    if name in installed:
        return
    has_build = name in installed_engines()
    refusal = choice.host_answer(name, runnable.session_host) if has_build else _install_gate(name)
    if not refusal.ok:
        state = "installed, but" if has_build else "not installed, and"
        raise EngineRefusedError(
            f"Engine {name} is {state} this computer cannot run it. {refusal.reason}".rstrip()
        )
    raise EngineRefusedError(f"Engine {name} is not installed. {LIST_HINT}")


def _store(
    key: str,
    apply: Callable[[object], tuple[object, _T]],
    commit: Callable[[_T], None],
) -> None:
    """Write one setting; the live value changes only after the file is written."""
    try:
        settings.mutate_value(cfg.data_root, key, apply, commit)
    except OSError as exc:
        log.warning("The engine choice was not saved: %s", exc)
        raise EngineNotSavedError(
            "The engine choice was not saved. Check that the data folder is writable "
            "and that no other lilbee process holds the settings file."
        ) from exc


def _apply_to_running() -> None:
    """Reload the roles of a running server whose engine changed; no services means nothing runs."""
    services = peek_services()
    if services is None:
        return
    try:
        services.provider.replan_fleet(wait=True)
    except (ProviderError, OSError) as exc:
        log.warning("The running server did not apply the engine choice: %s", exc)
        raise EngineNotAppliedError(
            "The engine choice is saved, but the running server could not apply it: "
            f"{str(exc).rstrip('.')}. Restart lilbee to use the choice."
        ) from exc


def _check_embed_identity(candidate: Config, confirmed: bool) -> None:
    """Refuse unless *confirmed* when the write changes the trigger of the embedder.

    A running server counts its own documents. With no server, the count is the saved library's.
    """
    if confirmed:
        return
    services = peek_services()
    documents = _document_count(services)
    if documents == 0:
        return
    registry = services.registry if services is not None else ModelRegistry(cfg.models_dir)
    if embedder_trigger(cfg, registry) != embedder_trigger(candidate, registry):
        raise EmbedConfirmRequiredError(cfg.data_root.name, documents)


@dataclass(frozen=True)
class BuildView:
    """What a surface shows of one installed engine build.

    ``differs_from_pin`` is True for a build that is not the build the pin names for this
    host: lilbee runs the pin build only, so this build stays on disk and runs nothing.
    """

    engine: str
    version: str
    backend: str
    differs_from_pin: bool = False

    @property
    def not_run_on(self) -> tuple[str, ...]:
        """The devices of the cell this host runs the build on that have no record."""
        return experimental_devices(self.engine, cell_key(runnable.session_host(), self.backend))

    @property
    def plain_label(self) -> str:
        """The build as ``name version (backend)``: JSON output carries this form only."""
        return f"{self.engine} {self.version} ({self.backend})"

    @property
    def label(self) -> str:
        """The line a person reads: the plain label, the pin clause, the experimental clause."""
        pin = f", {DIFFERS_FROM_PIN_TEXT}" if self.differs_from_pin else ""
        return f"{self.plain_label}{pin}{experimental_clause(self.not_run_on)}"


def engines_folder(config: Config = cfg) -> Path:
    """The machine-wide folder that holds the installed engine builds."""
    return engines_root(config.models_dir)


def _pin_build(name: EngineName, adapter: EngineAdapter) -> tuple[str, str] | None:
    """The (version, backend) of the build the pin names for this host; None when it names none."""
    pin = adapter.pin_record()
    if name is EngineName.LLAMA:
        return pin["version"], _bundled_backend(name)
    try:
        lock = choice.pin_lock(pin)
    except choice.PinError:
        return None
    return lock.version, lock.backend


def builds(config: Config = cfg) -> list[BuildView]:
    """Every engine build installed beside the models that the host gate lets this host run."""
    store = EngineStore(engines_root(config.models_dir))
    pins = {str(name): _pin_build(name, adapter) for name, adapter in choice.ADAPTERS.items()}
    return [
        BuildView(
            b.engine,
            b.version,
            b.backend,
            differs_from_pin=b.engine in pins and pins[b.engine] != (b.version, b.backend),
        )
        for b in store.installed()
        if choice.host_answer(b.engine, runnable.session_host).ok
    ]


def damaged_folders(config: Config = cfg) -> list[EngineFolderError]:
    """The error of each engine folder, or build folder in one, that lilbee cannot read."""
    return EngineStore(engines_root(config.models_dir)).damaged()


@dataclass(frozen=True)
class EngineListView:
    """What every surface returns for the list: builds, damaged and missing engines, the folder."""

    builds: list[BuildView]
    damaged: list[EngineFolderError]
    not_installed: list[str]
    folder: Path

    def experimental(self) -> dict[str, list[str]]:
        """Each listed engine with devices that have no record: a build's cell, else every cell."""
        found: dict[str, dict[str, None]] = {}
        for b in self.builds:
            found.setdefault(b.engine, {}).update(dict.fromkeys(b.not_run_on))
        for name in self.not_installed:
            found.setdefault(name, {}).update(dict.fromkeys(experimental_devices(name)))
        return {engine: list(devices) for engine, devices in found.items() if devices}

    def payload(self) -> dict[str, Any]:
        """The list as plain data, the same on the CLI, REST and MCP."""
        return {
            "builds": [
                {
                    "engine": b.engine,
                    "version": b.version,
                    "backend": b.backend,
                    "not_run_on": list(b.not_run_on),
                    "differs_from_pin": b.differs_from_pin,
                }
                for b in self.builds
            ],
            "damaged": [{"folder": str(e.folder), "error": str(e)} for e in self.damaged],
            "not_installed": self.not_installed,
            "experimental": self.experimental(),
            "engines_folder": str(self.folder),
        }


def engine_list(config: Config = cfg) -> EngineListView:
    """The installed builds, the damaged folders and the engines with no installed build.

    Every build of an engine with no installed build differs from the pin, because the
    pin build is the one that is missing.
    """
    absent = not_installed()
    listed = [
        replace(b, differs_from_pin=True) if b.engine in absent else b for b in builds(config)
    ]
    return EngineListView(listed, damaged_folders(config), absent, engines_folder(config))


def _removal_changes_embedder(
    store: EngineStore, registry: ModelRegistry, engine: str, version: str, backend: str
) -> bool:
    """True when the embedder trigger names this build, or this engine with no other build left."""
    trigger = embedder_trigger(cfg, registry)
    if trigger is None:
        return False
    if f"{store.root / engine / f'{version}-{backend}'}{os.sep}" in trigger:
        return True
    others = [
        b
        for b in store.installed()
        if b.engine == engine and (b.version, b.backend) != (version, backend)
    ]
    return f"|engine={engine}" in trigger and not others


def _check_removal_keeps_embedder(
    store: EngineStore, engine: str, version: str, backend: str, confirmed: bool
) -> None:
    """Refuse unless *confirmed* when removing the build changes the engine that embeds."""
    if confirmed:
        return
    if (engine, version, backend) not in [
        (b.engine, b.version, b.backend) for b in store.installed()
    ]:
        return
    services = peek_services()
    registry = services.registry if services is not None else ModelRegistry(cfg.models_dir)
    if not _removal_changes_embedder(store, registry, engine, version, backend):
        return
    documents = _document_count(services)
    if documents:
        raise EmbedConfirmRequiredError(cfg.data_root.name, documents)


def _document_count(services: Services | None) -> int:
    """The documents in the library: the running store's count, else the saved library's count."""
    if services is not None:
        return services.store.source_summary(0).document_count
    if not cfg.lancedb_dir.exists():
        return 0
    # The open registry is the one the exit waits on; a bare Store would escape it.
    store = _new_store(cfg)
    try:
        return store.source_summary(0).document_count
    finally:
        store.close()


def remove_build(
    engine: str, version: str, backend: str, config: Config = cfg, confirmed: bool = False
) -> str:
    """Remove one installed build and return its label; refuses a bundled, missing or busy one.

    A build that the embedder runs on needs *confirmed* when the library holds documents:
    the next start finds a changed engine and runs the canary check on the index.
    """
    store = EngineStore(engines_root(config.models_dir))
    _check_removal_keeps_embedder(store, engine, version, backend, confirmed)
    try:
        store.remove(engine, version, backend, in_use=partial(launch_uses_engine, store.root))
    except STORE_REFUSALS as exc:
        raise EngineRefusedError(str(exc)) from exc
    return BuildView(engine, version, backend).plain_label


def roles(config: Config = cfg) -> list[RoleView]:
    """The engine of every role, with the notice for a role that fell back or has none."""
    assignment = resolve_engines(config)
    views = []
    for role, info in ROLE_REGISTRY.items():
        ref = str(getattr(config, info.config_field))
        choice = assignment.choices[role]
        views.append(
            RoleView(
                role, ref, str(model_format(ref)), choice.name and str(choice.name), choice.notice
            )
        )
    return views


def _install_gate(name: EngineName) -> HostGate:
    """What the host says about installing the engine *name*; a bundled engine reads nothing."""
    adapter = choice.ADAPTERS.get(name)
    if adapter is None or adapter.bundled:
        return PASS
    return choice.install_offer(adapter, runnable.session_host())


def offered_engines() -> choice.InstalledEngines:
    """The installed engines that the host gate lets this host run: the ones a surface may offer."""
    return {
        name: formats
        for name, formats in installed_engines().items()
        if choice.host_answer(name, runnable.session_host).ok
    }


def not_installed() -> list[str]:
    """The names of the engines with no installed build that the host gate lets this host run."""
    installed = installed_engines()
    missing = [
        name
        for name, adapter in choice.ADAPTERS.items()
        if not (llama_server_ready() if adapter.bundled else name in installed)
    ]
    offered = [str(name) for name in missing if _install_gate(name).ok]
    hidden = [str(name) for name in missing if str(name) not in offered]
    host = runnable.session_host()
    if hidden and (host.os is None or host.arch is None):
        log.warning(
            "The engine list hides %s: the OS or CPU of this computer could not be read.",
            ", ".join(hidden),
        )
    return offered


class InstallRoute(StrEnum):
    """How a surface gets an engine that has no installed build."""

    DOWNLOAD = "download"
    """lilbee downloads the pack of the engine."""
    NO_PACK = "no_pack"
    """No pack of the engine is published yet, so nothing can be downloaded."""
    BY_HAND = "by_hand"
    """The user installs the engine; lilbee finds it."""


@dataclass(frozen=True)
class EngineOption:
    """One row of the engine choice for a model: an engine it can run on, or ``auto``.

    ``download_bytes`` is set only for an engine that is not installed and has a pack for
    this computer. ``route`` says how an engine with no build is installed; ``docs_url``
    is the install page of an engine the user installs by hand, or empty. The ``auto``
    row carries the engine that ``auto`` picks now in ``auto_now``, or None when no
    engine loads the model.
    """

    name: str
    installed: bool = True
    bundled: bool = False
    backend: str = ""
    download_bytes: int | None = None
    auto_now: str | None = None
    route: InstallRoute = InstallRoute.DOWNLOAD
    docs_url: str = ""

    @property
    def offers_download(self) -> bool:
        """True when the row is an engine with no build that lilbee can download."""
        return not self.installed and self.route is InstallRoute.DOWNLOAD


def _auto_now(model: str) -> str | None:
    """The engine that ``auto`` picks for *model* now, or None when no engine loads it."""
    try:
        return engine_for_format(model_format(model), cfg).name
    except choice.NoEngineError:
        return None


def _missing_option(name: EngineName, adapter: EngineAdapter) -> EngineOption:
    """The row of an engine with no installed build: how to get it, its backend, its pack size.

    llama.cpp reads the pin of its pack, as its install does: its own record names no backend.
    """
    if isinstance(adapter, UserInstalled):
        return EngineOption(
            str(name),
            installed=False,
            route=InstallRoute.BY_HAND,
            docs_url=adapter.install_docs_url,
        )
    host = runnable.session_host()
    pin = LLAMA.pack_pin() if name is EngineName.LLAMA else adapter.pin_record()
    try:
        lock = choice.pin_lock(pin)
    except choice.NoPublishedPackError:
        return EngineOption(str(name), installed=False, route=InstallRoute.NO_PACK)
    except (choice.PinError, OSError):
        return EngineOption(str(name), installed=False)
    key = cell_key(host, lock.backend)
    asset = lock.platforms.get(key) if key else None
    size = asset.size if asset is not None else None
    return EngineOption(str(name), installed=False, backend=lock.backend, download_bytes=size)


def _bundled_backend(name: EngineName) -> str:
    """The backend of the llama.cpp pack this host runs; empty for any other engine."""
    return pack_backend(runnable.session_host()) if name is EngineName.LLAMA else ""


def engine_options(model: str) -> list[EngineOption]:
    """The rows a surface shows to choose the engine of *model*: ``auto``, then each engine.

    An engine row appears for an engine that loads the format of *model* and that the host
    gate lets this host run, whether or not a build of it is installed.
    """
    fmt = model_format(model)
    present = offered_engines()
    backends = {b.engine: b.backend for b in builds()}
    rows = [EngineOption(AUTO, auto_now=_auto_now(model))]
    for name, adapter in choice.ADAPTERS.items():
        if fmt not in adapter.formats:
            continue
        if name in present:
            backend = backends.get(str(name)) or _bundled_backend(name)
            rows.append(EngineOption(str(name), bundled=adapter.bundled, backend=backend))
        elif _install_gate(name).ok:
            rows.append(_missing_option(name, adapter))
    return rows


def saved_engine(model: str) -> str | None:
    """The engine the user saved for *model*, or None when the user saved none."""
    return stored_choices(cfg.engine_choices).get(model)


class EngineState(StrEnum):
    """Whether an engine row is a build on disk, the bundled engine, a user install, or a download.

    ``USER_INSTALLED`` is an engine the user installed outside lilbee, such as a binary on PATH.
    ``DIFFERS_FROM_PIN`` is a build on disk of an engine that is not installed: another
    version, another backend, or another wheel than the pin names.
    """

    INSTALLED = "installed"
    BUNDLED = "bundled"
    USER_INSTALLED = "user_installed"
    NOT_INSTALLED = "not_installed"
    DIFFERS_FROM_PIN = "differs_from_pin"


@dataclass(frozen=True)
class EngineRow:
    """One row of the engine table: a build, the bundled engine, a user install, or a missing one.

    ``download_bytes`` is set only for a missing engine that has a pack for this computer.
    ``size_bytes`` is the size on disk of a build, None when it cannot be read.
    ``roles`` names the roles that use the engine now.
    """

    engine: str
    state: EngineState
    version: str = ""
    backend: str = ""
    download_bytes: int | None = None
    roles: tuple[str, ...] = ()
    size_bytes: int | None = None


def _build_row(
    store: EngineStore, build: BuildView, state: EngineState, users: tuple[str, ...] = ()
) -> EngineRow:
    """The row of one build on disk, with its size."""
    return EngineRow(
        build.engine,
        state,
        build.version,
        build.backend,
        roles=users,
        size_bytes=store.build_size(build.engine, build.version, build.backend),
    )


def engine_rows(config: Config = cfg, listed: EngineListView | None = None) -> list[EngineRow]:
    """The engine list as table rows: one row per build, then each missing engine.

    An engine counts as installed or missing by the rule of ``engine_list``, so a user
    install is an installed one. A missing engine is followed by a row for each of its
    builds on disk, as ``lilbee engine list`` names them, so the user can remove them.
    """
    used: dict[str, list[str]] = {}
    for view in roles(config):
        if view.engine and view.model.strip():
            used.setdefault(view.engine, []).append(view.role.value)
    listed = listed or engine_list(config)
    missing = set(listed.not_installed)
    present = offered_engines()
    store = EngineStore(listed.folder)
    rows: list[EngineRow] = []
    for name, adapter in choice.ADAPTERS.items():
        key = str(name)
        mine = [b for b in listed.builds if b.engine == key]
        if key in missing or name not in present:
            if key in missing:
                option = _missing_option(name, adapter)
                rows.append(
                    EngineRow(
                        key,
                        EngineState.NOT_INSTALLED,
                        backend=option.backend,
                        download_bytes=option.download_bytes,
                    )
                )
            rows.extend(_build_row(store, b, EngineState.DIFFERS_FROM_PIN) for b in mine)
            continue
        users = tuple(used.get(key, ()))
        rows.extend(
            _build_row(
                store,
                b,
                EngineState.DIFFERS_FROM_PIN if b.differs_from_pin else EngineState.INSTALLED,
                users,
            )
            for b in mine
        )
        if adapter.bundled:
            rows.append(
                EngineRow(key, EngineState.BUNDLED, backend=_bundled_backend(name), roles=users)
            )
        elif not mine:
            rows.append(EngineRow(key, EngineState.USER_INSTALLED, roles=users))
    return rows


def prefer(name: str, confirmed: bool = False) -> str:
    """Store the global engine preference; refuses an unknown or uninstalled engine."""
    try:
        setting = EngineSetting(name)
    except ValueError:
        raise _unknown(name) from None
    if setting is not EngineSetting.AUTO:
        _require_installed(EngineName(setting.value), offered_engines())
    _check_embed_identity(cfg.model_copy(update={"preferred_engine": setting}), confirmed)

    def apply(_raw: object) -> tuple[str, str]:
        return setting.value, setting.value

    def commit(_chosen: str) -> None:
        cfg.preferred_engine = setting

    _store(PREFERRED_KEY, apply, commit)
    _apply_to_running()
    return setting.value


def _installable(name: str) -> EngineAdapter:
    """The adapter of the engine named *name*; refuses a name with no adapter."""
    try:
        engine = EngineName(name)
    except ValueError:
        raise _not_installable(name) from None
    adapter = choice.ADAPTERS.get(engine)
    if adapter is None:
        raise EngineRefusedError(
            f"Engine {name} has no build to install in this version of lilbee."
        )
    return adapter


def install(
    name: str,
    wheel: Path | None = None,
    on_update: Callable[[DownloadProgress], None] | None = None,
    on_retry: Callable[[str], None] | None = None,
    cancel: CancelSignal | None = None,
) -> InstalledBuild:
    """Install the pinned build of engine *name* from *wheel*, a pack, or its release URL.

    *on_update* hears the download bytes. *on_retry* hears each failed attempt with its
    cause and attempt number. A set *cancel* ends the download, also while it waits to try again.

    The host gate and the cell for this computer run first. The store checks the digest,
    the licence and the files of the wheel and writes the record.
    """
    adapter = _installable(name)
    try:
        return _install_adapter(adapter, name, wheel, on_update, on_retry, cancel)
    except EngineRefusedError as exc:
        raise EngineInstallRefusedError(str(exc)) from exc


def _install_adapter(
    adapter: EngineAdapter,
    name: str,
    wheel: Path | None,
    on_update: Callable[[DownloadProgress], None] | None,
    on_retry: Callable[[str], None] | None,
    cancel: CancelSignal | None,
) -> InstalledBuild:
    """Install the build of *adapter*; every refusal is an ``EngineRefusedError``."""
    if adapter.bundled:
        if not _pack_installable():
            raise EngineRefusedError(
                f"Engine {name} ships with lilbee. There is nothing to install."
            )
        return _install_llama_pack(wheel, on_update, on_retry, cancel)
    host = refresh_session_host()
    gate = choice.install_answer(adapter, host)
    if not gate.ok:
        if not gate.known:
            raise EngineRefusedError(gate.reason)
        raise EngineRefusedError(
            f"Engine {name} cannot run on this computer. {gate.reason} {LLAMA.name} stays in use."
        )
    try:
        lock = choice.pin_lock(adapter.pin_record())
    except choice.NoPublishedPackError as exc:
        hint = f" {adapter.install_hint}" if isinstance(adapter, UserInstalled) else ""
        raise EngineRefusedError(f"{exc}{hint}") from exc
    except choice.PinError as exc:
        raise EngineRefusedError(f"Engine {name} cannot be installed: {exc}.") from exc
    store = EngineStore(engines_root(cfg.models_dir))
    try:
        cell = host_cell(lock, host)
        lock = lock.for_cell(cell)
        if wheel is None:
            on_bytes = make_download_callback(on_update) if on_update is not None else None
            store.ensure(
                lock,
                lambda pin, scratch: fetch_frozen(
                    pin,
                    cell,
                    dest=scratch,
                    on_progress=on_bytes,
                    on_retry=retry_reporter(on_retry),
                    cancel=cancel,
                ),
            )
        else:
            store.install_from_wheel(lock, wheel)
    except _STORE_REFUSALS as exc:
        raise EngineRefusedError(str(exc)) from exc
    except _UNREADABLE_WHEEL as exc:
        raise EngineRefusedError(f"Engine {name} was not installed: {exc}") from exc
    _install_router(name, on_update, on_retry, cancel)
    return InstalledBuild(lock.engine, lock.version, lock.backend, gate.reason)


def _install_router(
    name: str,
    on_update: Callable[[DownloadProgress], None] | None,
    on_retry: Callable[[str], None] | None,
    cancel: CancelSignal | None,
) -> None:
    """Install the router that starts every engine; a router already installed is kept."""
    try:
        install_router(
            on_progress=make_download_callback(on_update) if on_update is not None else None,
            on_retry=retry_reporter(on_retry),
            cancel=cancel,
        )
    except ROUTER_REFUSALS as exc:
        raise EngineRefusedError(
            f"Engine {name} is installed, but the router that starts it is not. {exc}"
        ) from exc


def _pack_installable() -> bool:
    """True on a computer where no wheel, path or PATH entry supplies llama.cpp."""
    from lilbee.providers.fleet.binary import EngineTool, resolve_engine_tool

    try:
        resolve_engine_tool(EngineTool.LLAMA_SERVER, install=False)
    except ProviderError:
        return True
    return False


def _install_llama_pack(
    wheel: Path | None,
    on_update: Callable[[DownloadProgress], None] | None,
    on_retry: Callable[[str], None] | None,
    cancel: CancelSignal | None,
) -> InstalledBuild:
    """Install the llama.cpp pack through the store, as a first use does."""
    from lilbee.providers.fleet import first_use

    refresh_session_host()
    try:
        lock = first_use.install_pack(
            wheel,
            on_progress=make_download_callback(on_update) if on_update is not None else None,
            on_retry=retry_reporter(on_retry),
            cancel=cancel,
        )
    except first_use.INSTALL_REFUSALS as exc:
        raise EngineRefusedError(str(exc)) from exc
    except _UNREADABLE_WHEEL as exc:
        raise EngineRefusedError(f"Engine {EngineName.LLAMA} was not installed: {exc}") from exc
    return InstalledBuild(lock.engine, lock.version, lock.backend, "")


def http_refused_message() -> str:
    """The refusal text for an engine choice on the HTTP server while allow_http_engines is off."""
    return (
        "Changing the engine on the HTTP server is unavailable: the choice applies to every "
        "connected client. Enable allow_http_engines (LILBEE_ALLOW_HTTP_ENGINES) on a "
        "deployment you alone use, or change the engine from the CLI or TUI."
    )


def parse_role(text: str) -> WorkerRole:
    """The role named by *text*; a surface that reads free text calls this once."""
    try:
        return WorkerRole(text)
    except ValueError:
        known = ", ".join(WorkerRole)
        raise EngineRefusedError(f"No role is named {text!r}. Choose one of: {known}.") from None


def use_for_role(
    role: WorkerRole, model: str, name: str, config: Config = cfg, confirmed: bool = False
) -> str:
    """Store the engine for the model of *role*; refuses a model that is not the role's model."""
    model = canonical_ref(model)
    current = str(getattr(config, ROLE_REGISTRY[role].config_field))
    if not current.strip():
        raise EngineRefusedError(
            f"No {role} model is set. Choose a {role} model before you choose its engine."
        )
    if model != current:
        raise EngineRefusedError(
            f"The {role} model is {current}, not {model}. Choose the engine for {current}."
        )
    if name != AUTO:
        _require_serves(_engine_name(name), role, model)
    return use(model, name, confirmed)


def _require_serves(name: EngineName, role: WorkerRole, model: str) -> None:
    adapter = choice.ADAPTERS.get(name)
    if adapter is not None and role not in adapter.served_roles:
        raise EngineRefusedError(unserved_role_message(name, role, adapter.served_roles))
    fmt = model_format(model)
    if choice.refuses_format_for_role(name, role, fmt):
        raise EngineRefusedError(
            f"Engine {name} cannot load the {fmt} format of {model} for {role}. "
            f"Choose an engine that loads {fmt} for {role}."
        )


def use(model: str, name: str, confirmed: bool = False) -> str:
    """Store the engine for *model*; ``auto`` removes it. Refuses an engine that cannot run it."""
    if not model.strip():
        raise EngineRefusedError("Give the model to choose an engine for.")
    if name != AUTO:
        engine = _engine_name(name)
        installed = offered_engines()
        _require_installed(engine, installed)
        fmt = model_format(model)
        if engine not in engines_loading(fmt, installed):
            raise EngineRefusedError(
                f"Engine {name} cannot load the {fmt} format of {model}. "
                f"Choose an engine that loads {fmt}."
            )

    def merged(raw: str | None) -> str | None:
        chosen = stored_choices(raw)
        if name == AUTO:
            chosen.pop(model, None)
        else:
            chosen[model] = name
        return json.dumps(chosen, sort_keys=True) if chosen else None

    candidate = cfg.model_copy(update={"engine_choices": merged(cfg.engine_choices)})
    _check_embed_identity(candidate, confirmed)

    def apply(raw: object) -> tuple[str | None, str | None]:
        encoded = merged(raw if isinstance(raw, str) else None)
        return encoded, encoded

    def commit(encoded: str | None) -> None:
        cfg.engine_choices = encoded

    _store(CHOICES_KEY, apply, commit)
    _apply_to_running()
    return name
