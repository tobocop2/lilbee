"""Surface-agnostic use-cases for choosing the engine of a model."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from lilbee.app.engine_choice import CHOICES_KEY, resolve_engines, stored_choices
from lilbee.app.services import peek_services
from lilbee.catalog.download import RetryCallback
from lilbee.catalog.download_progress import make_download_callback
from lilbee.catalog.models import DownloadProgress
from lilbee.catalog.refs import model_format
from lilbee.catalog.runnable import refresh_session_host, session_host
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.engine_manifest import UnsupportedOsError, host_cell
from lilbee.providers.fleet.engine_store import (
    BundledEngineError,
    ChecksumError,
    EngineConflictError,
    EngineDiskError,
    EngineInUseError,
    EngineNotInstalledError,
    EngineStore,
    LicenceRefusedError,
    MissingPackError,
    PackageInstallError,
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
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.engines.proof import not_run_clause, not_run_on
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole, unserved_role_message
from lilbee.retrieval.canary_probe import embedder_trigger
from lilbee.runtime.cancellation import CancelSignal

log = logging.getLogger(__name__)
_T = TypeVar("_T")

STORE_REFUSALS = (BundledEngineError, EngineInUseError, EngineNotInstalledError, StorePathError)
NO_BUILDS = "No engine builds are installed."
REMOVE_DONE = "Removed engine {build}."
PREFERRED_KEY = "preferred_engine"
NOT_APPLIED_CODE = "engine_not_applied"


class EngineRefusedError(ValueError):
    """The engine service refuses a write; the text names the problem and the fix."""


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
    not_run_on: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """The build as one line, with the devices it has not run on."""
        return BuildView(self.engine, self.version, self.backend, self.not_run_on).label


_STORE_REFUSALS = (
    ChecksumError,
    EngineConflictError,
    EngineDiskError,
    LicenceRefusedError,
    MissingPackError,
    PackageInstallError,
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

WHEEL_CLI_ONLY = "The wheel option works only with the lilbee engine install command."


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
    if name not in installed:
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
        services.provider.reload_placement(wait=True)
    except (ProviderError, OSError) as exc:
        log.warning("The running server did not apply the engine choice: %s", exc)
        raise EngineNotAppliedError(
            "The engine choice is saved, but the running server could not apply it: "
            f"{str(exc).rstrip('.')}. Restart lilbee to use the choice."
        ) from exc


def _check_embed_identity(candidate: Config, confirmed: bool) -> None:
    """Refuse unless *confirmed* when the write changes the trigger of the embedder."""
    services = peek_services()
    if services is None or confirmed:
        return
    documents = services.store.source_summary(0).document_count
    if documents == 0:
        return
    registry = services.registry
    if embedder_trigger(cfg, registry) != embedder_trigger(candidate, registry):
        raise EmbedConfirmRequiredError(cfg.data_root.name, documents)


@dataclass(frozen=True)
class BuildView:
    """What a surface shows of one installed engine build."""

    engine: str
    version: str
    backend: str
    not_run_on: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """The build as one line: ``name version (backend)``, then the devices not run on."""
        return f"{self.engine} {self.version} ({self.backend}){not_run_clause(self.not_run_on)}"


def builds(config: Config = cfg) -> list[BuildView]:
    """Every engine build installed beside the models that the host gate lets this host run."""
    store = EngineStore(engines_root(config.models_dir))
    return [
        BuildView(b.engine, b.version, b.backend, not_run_on(b.engine))
        for b in store.installed()
        if choice.host_answer(b.engine, session_host).ok
    ]


@dataclass(frozen=True)
class EngineListView:
    """What every surface returns for the engine list: the builds and the missing engines."""

    builds: list[BuildView]
    not_installed: list[str]

    def payload(self) -> dict[str, list[dict[str, Any]] | list[str]]:
        """The list as plain data, the same on the CLI, REST and MCP."""
        return {
            "builds": [
                {
                    "engine": b.engine,
                    "version": b.version,
                    "backend": b.backend,
                    "not_run_on": list(b.not_run_on),
                }
                for b in self.builds
            ],
            "not_installed": self.not_installed,
        }


def engine_list(config: Config = cfg) -> EngineListView:
    """The installed builds and the engines with no installed build."""
    return EngineListView(builds(config), not_installed())


def remove_build(engine: str, version: str, backend: str, config: Config = cfg) -> str:
    """Remove one installed build and return its label; refuses a bundled, missing or busy one."""
    store = EngineStore(engines_root(config.models_dir))
    try:
        store.remove(engine, version, backend, in_use=launch_uses_engine)
    except STORE_REFUSALS as exc:
        raise EngineRefusedError(str(exc)) from exc
    return BuildView(engine, version, backend).label


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


def _offered(adapter: EngineAdapter) -> bool:
    """True when an install of *adapter* passes the host gate; a bundled engine reads nothing."""
    return adapter.bundled or choice.install_answer(adapter, session_host()).ok


def not_installed() -> list[str]:
    """The names of the engines with no installed build that the host gate lets this host run."""
    installed = installed_engines()
    missing = [
        (name, adapter) for name, adapter in choice.ADAPTERS.items() if name not in installed
    ]
    offered = [str(name) for name, adapter in missing if _offered(adapter)]
    hidden = [str(name) for name, _ in missing if str(name) not in offered]
    host = session_host()
    if hidden and (host.os is None or host.arch is None):
        log.warning(
            "The engine list hides %s: the OS or CPU of this host could not be read.",
            ", ".join(hidden),
        )
    return offered


def engine_names(model: str) -> list[str]:
    """The names a surface may offer for *model*: ``auto`` and each engine that loads it."""
    return [AUTO, *engines_loading(model_format(model), installed_engines())]


def prefer(name: str, confirmed: bool = False) -> str:
    """Store the global engine preference; refuses an unknown or uninstalled engine."""
    try:
        setting = EngineSetting(name)
    except ValueError:
        raise _unknown(name) from None
    if setting is not EngineSetting.AUTO:
        _require_installed(EngineName(setting.value), installed_engines())
    _check_embed_identity(cfg.model_copy(update={"preferred_engine": setting}), confirmed)

    def apply(_raw: object) -> tuple[str, str]:
        return setting.value, setting.value

    def commit(_chosen: str) -> None:
        cfg.preferred_engine = setting

    _store(PREFERRED_KEY, apply, commit)
    _apply_to_running()
    return setting.value


_RETRY_TEXT = "{cause}. Attempt {attempt} of {attempts} failed; trying again."


def _retry_reporter(on_retry: Callable[[str], None] | None) -> RetryCallback | None:
    """Turn each failed attempt into the sentence *on_retry* shows; None when nobody listens."""
    if on_retry is None:
        return None

    def report(attempt: int, attempts: int, cause: str) -> None:
        on_retry(_RETRY_TEXT.format(cause=cause, attempt=attempt, attempts=attempts))

    return report


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
            f"Engine {name} cannot run on this host. {gate.reason} {LLAMA.name} stays in use."
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
                    on_retry=_retry_reporter(on_retry),
                    cancel=cancel,
                ),
            )
        else:
            store.install_from_wheel(lock, wheel)
    except _STORE_REFUSALS as exc:
        raise EngineRefusedError(str(exc)) from exc
    except _UNREADABLE_WHEEL as exc:
        raise EngineRefusedError(f"Engine {name} was not installed: {exc}") from exc
    return InstalledBuild(
        lock.engine, lock.version, lock.backend, gate.reason, not_run_on(lock.engine)
    )


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
            on_retry=_retry_reporter(on_retry),
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
        _require_serves(_engine_name(name), role)
    return use(model, name, confirmed)


def _require_serves(name: EngineName, role: WorkerRole) -> None:
    adapter = choice.ADAPTERS.get(name)
    if adapter is not None and role not in adapter.served_roles:
        raise EngineRefusedError(unserved_role_message(name, role, adapter.served_roles))


def use(model: str, name: str, confirmed: bool = False) -> str:
    """Store the engine for *model*; ``auto`` removes it. Refuses an engine that cannot run it."""
    if not model.strip():
        raise EngineRefusedError("Give the model to choose an engine for.")
    if name != AUTO:
        engine = _engine_name(name)
        installed = installed_engines()
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
