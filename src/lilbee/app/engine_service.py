"""Surface-agnostic use-cases for choosing the engine of a model."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from lilbee.app.engine_choice import CHOICES_KEY, resolve_engines, stored_choices
from lilbee.catalog.refs import model_format
from lilbee.core import settings
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engine_store import (
    ChecksumError,
    EngineConflictError,
    EngineDiskError,
    EngineStore,
    LicenceRefusedError,
    MissingPackError,
    PackageInstallError,
    PinLock,
    UnsafeWheelError,
    UnsupportedPlatformError,
    engines_root,
)
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import (
    AUTO,
    LIST_HINT,
    InstalledEngines,
    engines_loading,
    installed_engines,
)
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole

log = logging.getLogger(__name__)
_T = TypeVar("_T")

PREFERRED_KEY = "preferred_engine"
RESTART_NOTE = "A running server uses the choice after a restart."


class EngineRefusedError(ValueError):
    """The engine service refuses a write; the text names the problem and the fix."""


class EngineNotSavedError(Exception):
    """The choice could not be written to the settings file; the text holds no path."""


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


_STORE_REFUSALS = (
    ChecksumError,
    EngineConflictError,
    EngineDiskError,
    LicenceRefusedError,
    MissingPackError,
    PackageInstallError,
    UnsafeWheelError,
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


def engine_names(model: str) -> list[str]:
    """The names a surface may offer for *model*: ``auto`` and each engine that loads it."""
    return [AUTO, *engines_loading(model_format(model), installed_engines())]


def prefer(name: str) -> str:
    """Store the global engine preference; refuses an unknown or uninstalled engine."""
    try:
        setting = EngineSetting(name)
    except ValueError:
        raise _unknown(name) from None
    if setting is not EngineSetting.AUTO:
        _require_installed(EngineName(setting.value), installed_engines())

    def apply(_raw: object) -> tuple[str, str]:
        return setting.value, setting.value

    def commit(_chosen: str) -> None:
        cfg.preferred_engine = setting

    _store(PREFERRED_KEY, apply, commit)
    return setting.value


def _no_download(_lock: PinLock, _scratch: Path) -> Path:
    """The fetch for an install with no wheel: engines have no download route."""
    raise FileNotFoundError("Engines have no download route.")


def install(name: str, wheel: Path | None = None) -> InstalledBuild:
    """Install the pinned build of engine *name* from *wheel*, else from its pack.

    The host gate of the engine runs first. The store checks the digest, the licence
    and the files of the wheel and writes the record.
    """
    try:
        engine = EngineName(name)
    except ValueError:
        raise _not_installable(name) from None
    adapter = choice.ADAPTERS.get(engine)
    if adapter is None:
        raise EngineRefusedError(
            f"Engine {name} has no build to install in this version of lilbee."
        )
    if adapter.bundled:
        raise EngineRefusedError(f"Engine {name} ships with lilbee. There is nothing to install.")
    gate = adapter.host_gate(host_caps.detect())
    if not gate.ok:
        raise EngineRefusedError(f"Engine {name} cannot run on this host. {gate.reason}")
    try:
        lock = choice.pin_lock(adapter.pin_record())
    except choice.PinError as exc:
        raise EngineRefusedError(f"Engine {name} cannot be installed: {exc}.") from exc
    store = EngineStore(engines_root(cfg.models_dir))
    try:
        if wheel is None:
            store.ensure(lock, _no_download)
        else:
            store.install_from_wheel(lock, wheel)
    except _STORE_REFUSALS as exc:
        raise EngineRefusedError(str(exc)) from exc
    except _UNREADABLE_WHEEL as exc:
        raise EngineRefusedError(f"Engine {name} was not installed: {exc}") from exc
    return InstalledBuild(lock.engine, lock.version, lock.backend)


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


def use_for_role(role: WorkerRole, model: str, name: str, config: Config = cfg) -> str:
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
    return use(model, name)


def use(model: str, name: str) -> str:
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

    def apply(raw: object) -> tuple[str | None, str | None]:
        chosen = stored_choices(raw if isinstance(raw, str) else None)
        if name == AUTO:
            chosen.pop(model, None)
        else:
            chosen[model] = name
        encoded = json.dumps(chosen, sort_keys=True) if chosen else None
        return encoded, encoded

    def commit(encoded: str | None) -> None:
        cfg.engine_choices = encoded

    _store(CHOICES_KEY, apply, commit)
    return name
