"""The pure rule that picks the engine for one model format on one host."""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.engine_store import (
    EngineConflictError,
    EngineStore,
    PinLock,
    engines_root,
)
from lilbee.providers.fleet.engines.base import EngineAdapter, EngineName
from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

AUTO = "auto"
_DARWIN = "darwin"
_ARM64 = "arm64"
# The command that shows the installed engines.
LIST_HINT = "Run lilbee engine list to see the installed engines."

# The adapters this build ships, by engine name.
ADAPTERS: Mapping[EngineName, EngineAdapter] = {EngineName(a.name): a for a in (LLAMA, IK_LLAMA)}

# What each installed engine loads: the ``formats`` of its adapter.
InstalledEngines = Mapping[EngineName, Collection[str]]


class Accel(StrEnum):
    """Accelerators the default rule looks at."""

    INTEL_NPU = "intel_npu"


class NoticeKind(StrEnum):
    """Why the rule did not use the engine that was asked for."""

    OVERRIDE_UNUSABLE = "override-unusable"
    PREFERRED_UNUSABLE = "preferred-unusable"
    NONE = "none"


class NoEngineError(ProviderError):
    """No installed engine loads the model format."""


@dataclass(frozen=True)
class HostFacts:
    """The host properties the default rule reads; ``system`` and ``machine`` are lower case."""

    system: str
    machine: str
    accel: tuple[Accel, ...] = ()


@dataclass(frozen=True)
class EngineChoice:
    """The engine picked for one model, or no engine and the notice that says why."""

    name: EngineName | None
    notice_kind: NoticeKind | None
    notice: str | None

    def require(self) -> EngineName:
        """The engine name, or ``NoEngineError`` when no engine can load the format."""
        if self.name is None:
            raise NoEngineError(self.notice or "")
        return self.name


def _default(fmt: ModelFormat, host: HostFacts, installed: InstalledEngines) -> EngineName | None:
    """The first engine in the host's order that is installed and loads *fmt*."""
    order: list[EngineName] = []
    if Accel.INTEL_NPU in host.accel:
        order.append(EngineName.OPENVINO)
    if host.system == _DARWIN and host.machine == _ARM64:
        order.append(EngineName.MLX)
    order.append(EngineName.LLAMA)
    return next((e for e in order if e in installed and fmt in installed[e]), None)


def _request(preferred: str, override: str | None) -> tuple[str, NoticeKind] | None:
    for value, kind in (
        (override, NoticeKind.OVERRIDE_UNUSABLE),
        (preferred, NoticeKind.PREFERRED_UNUSABLE),
    ):
        if value is not None and not value.strip():
            raise ValueError(f"The engine name must not be empty (got {value!r}).")
        if value is not None and value != AUTO:
            return value, kind
    return None


def engines_loading(fmt: str, installed: InstalledEngines) -> list[EngineName]:
    """The installed engines whose adapter loads *fmt*."""
    return [name for name, formats in installed.items() if fmt in formats]


def choose_engine(
    *,
    fmt: ModelFormat,
    host: HostFacts,
    installed: InstalledEngines,
    preferred: str,
    override: str | None,
    role: WorkerRole | None = None,
    model: str | None = None,
) -> EngineChoice:
    """Pick the engine for *fmt*: the override, else the preference, else the host default.

    The result never names an engine that cannot load *fmt*. A preference that
    is unusable gives the host default, with a notice. An override that is
    installed but cannot load *fmt* gives no engine. A notice names *role* and
    *model* when given.
    """
    request = _request(preferred, override)
    of_model = f" of {model}" if model else ""
    if request is None:
        default = _default(fmt, host, installed)
        if default is not None:
            return EngineChoice(default, None, None)
        text = f"No installed engine loads the {fmt} format{of_model}. {LIST_HINT}"
        return EngineChoice(None, NoticeKind.NONE, f"For {role.value}: {text}" if role else text)
    name, kind = request
    if name in engines_loading(fmt, installed):
        return EngineChoice(EngineName(name), None, None)
    subject = f"Engine {name} for {role.value}" if role else f"Engine {name}"
    if name not in installed:
        reason = f"{subject} is not installed."
        hint = f" {LIST_HINT}"
    else:
        reason = f"{subject} cannot load the {fmt} format{of_model}."
        hint = ""
        if kind is NoticeKind.OVERRIDE_UNUSABLE:
            return EngineChoice(None, kind, f"{reason} Choose an engine that loads {fmt}.")
    default = _default(fmt, host, installed)
    if default is None:
        return EngineChoice(None, kind, f"{reason} No installed engine loads the {fmt} format.")
    return EngineChoice(default, kind, f"{reason} Using {default}.{hint}")


def installed_engines() -> InstalledEngines:
    """The formats of each adapter that ships in the package or has a complete store build."""
    store = EngineStore(engines_root(cfg.models_dir))
    return {
        name: adapter.formats
        for name, adapter in ADAPTERS.items()
        if _is_installed(name, adapter, store)
    }


class PinError(Exception):
    """An adapter's pin record cannot name a build."""


def pin_lock(pin: Mapping[str, Any]) -> PinLock:
    """The lock for *pin*; raises ``PinError`` for a missing field or an unusable name."""
    try:
        licences = tuple(pin["licenses"])
        return PinLock(pin["engine"], pin["version"], pin["backend"], {}, licences, pin["sha256"])
    except (KeyError, ValueError, TypeError) as exc:
        raise PinError(f"the pin record is not usable ({exc!r})") from exc


def _is_installed(name: EngineName, adapter: EngineAdapter, store: EngineStore) -> bool:
    """True for the engine in the package, or an adapter whose pin has a complete build.

    A read that fails (unreadable folder, a build of another wheel) leaves that
    engine not installed and is logged; it never touches the engine in the package.
    """
    if adapter.bundled:
        return True
    try:
        pin = adapter.pin_record()
        lock = pin_lock(pin)
        return store.complete_build(lock) is not None
    except (OSError, EngineConflictError, PinError) as exc:
        log.warning("Engine %s counts as not installed: %s", name, exc)
        return False


@dataclass(frozen=True)
class RoleEngine:
    """The engine one role runs on, with the notice that says why it is not the one asked for."""

    name: EngineName
    notice: str | None

    @property
    def adapter(self) -> EngineAdapter:
        """The adapter of the engine."""
        return ADAPTERS[self.name]


@dataclass(frozen=True)
class EngineAssignment:
    """The choice for each role, so two roles can run on two engines."""

    choices: Mapping[WorkerRole, EngineChoice]

    @classmethod
    def for_test(cls, engines: Mapping[str, str]) -> EngineAssignment:
        """An assignment with no notices, from role and engine names."""
        return cls(
            {
                WorkerRole(role): EngineChoice(EngineName(name), None, None)
                for role, name in engines.items()
            }
        )

    def for_role(self, role: WorkerRole) -> RoleEngine:
        """The engine for *role*; raises ``NoEngineError`` when none loads its model."""
        choice = self.choices[role]
        return RoleEngine(choice.require(), choice.notice)
