"""The pure rule that picks the engine for one model format on one host."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.engine_store import (
    EngineConflictError,
    EngineStore,
    PinLock,
    engines_root,
)
from lilbee.providers.fleet.engines.base import EngineAdapter, EngineName
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

AUTO = "auto"
# The command that shows the installed engines.
LIST_HINT = "Run lilbee engine list to see the installed engines."

# The adapters this build ships, by engine name.
ADAPTERS: Mapping[EngineName, EngineAdapter] = {
    EngineName(a.name): a for a in (LLAMA, IK_LLAMA, TEI, MISTRAL)
}

# What each installed engine loads: the ``formats`` of its adapter.
InstalledEngines = Mapping[EngineName, Collection[str]]

# The reader of the host, called only when a gate needs its answer.
HostSource = Callable[[], HostCaps]


class NoticeKind(StrEnum):
    """Why the rule did not use the engine that was asked for."""

    OVERRIDE_UNUSABLE = "override-unusable"
    PREFERRED_UNUSABLE = "preferred-unusable"
    NONE = "none"


class NoEngineError(ProviderError):
    """No installed engine loads the model format."""


@dataclass(frozen=True)
class EngineChoice:
    """The engine picked for one model, or no engine and the notice that says why."""

    name: EngineName | None
    notice_kind: NoticeKind | None
    notice: str | None

    def require(self) -> EngineName:
        """The engine name, or ``NoEngineError`` when no engine can load the format."""
        if self.name is None:
            raise NoEngineError(self.notice or "", kind=ProviderErrorKind.BAD_REQUEST)
        return self.name


def _adapter(name: str) -> EngineAdapter | None:
    return next((a for n, a in ADAPTERS.items() if n == name), None)


def _serves(name: str, role: WorkerRole | None) -> bool:
    """False only for a shipped adapter that does not serve *role*."""
    adapter = _adapter(name)
    return role is None or adapter is None or role in adapter.served_roles


def _host_answer(name: str, host: HostSource) -> HostGate:
    """What the host says about *name*; a bundled engine reads nothing."""
    adapter = _adapter(name)
    return PASS if adapter is None or adapter.bundled else adapter.host_gate(host())


def _is_apple_silicon(caps: HostCaps) -> bool:
    return caps.os is HostOS.MACOS and caps.arch is HostArch.ARM64


def _default(
    fmt: ModelFormat, host: HostSource, installed: InstalledEngines, role: WorkerRole | None
) -> EngineName | None:
    """The first engine in the host's order that is installed, loads *fmt* and serves *role*."""
    order: list[EngineName] = []
    if EngineName.MLX in installed and _is_apple_silicon(host()):
        order.append(EngineName.MLX)
    order.append(EngineName.LLAMA)
    return next(
        (
            e
            for e in order
            if e in installed
            and fmt in installed[e]
            and _serves(e, role)
            and _host_answer(e, host).auto_selectable
        ),
        None,
    )


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


def _candidates(
    fmt: str, installed: InstalledEngines, role: WorkerRole | None, host: HostSource
) -> list[EngineName]:
    """The installed engines that load *fmt*, serve *role* and can run on the host."""
    return [
        name
        for name in engines_loading(fmt, installed)
        if _serves(name, role) and _host_answer(name, host).ok
    ]


def _choose_hint(names: list[EngineName]) -> str:
    return f"Choose {' or '.join(names)} with lilbee engine use or lilbee engine prefer."


def _by_default(
    fmt: ModelFormat,
    host: HostSource,
    installed: InstalledEngines,
    role: WorkerRole | None,
    of_model: str,
) -> EngineChoice:
    """The host default with no notice, or no engine and the notice that says why."""
    default = _default(fmt, host, installed, role)
    if default is not None:
        return EngineChoice(default, None, None)
    candidates = _candidates(fmt, installed, role, host)
    if candidates:
        text = f"No engine is chosen by default for the {fmt} format{of_model}. "
        text += _choose_hint(candidates)
    else:
        text = f"No installed engine loads the {fmt} format{of_model}. {LIST_HINT}"
    return EngineChoice(None, NoticeKind.NONE, f"For {role.value}: {text}" if role else text)


def choose_engine(
    *,
    fmt: ModelFormat,
    host: HostSource,
    installed: InstalledEngines,
    preferred: str,
    override: str | None,
    role: WorkerRole | None = None,
    model: str | None = None,
) -> EngineChoice:
    """Pick the engine for *fmt*: the override, else the preference, else the host default.

    The result never names an engine that cannot load *fmt* or that the host
    refuses. A request that is unusable gives the host default, with a notice
    that holds the reason. An override that is installed but cannot load *fmt*
    gives no engine. A notice names *role* and *model* when given.
    """
    request = _request(preferred, override)
    of_model = f" of {model}" if model else ""
    if request is None:
        return _by_default(fmt, host, installed, role, of_model)
    name, kind = request
    loads = name in engines_loading(fmt, installed)
    serves = _serves(name, role)
    answer = _host_answer(name, host) if loads and serves else PASS
    if loads and serves and answer.ok:
        return EngineChoice(EngineName(name), None, None)
    subject = f"Engine {name} for {role.value}" if role else f"Engine {name}"
    hint = ""
    left = f"loads the {fmt} format"
    if name not in installed:
        reason = f"{subject} is not installed."
        hint = f" {LIST_HINT}"
    elif not loads:
        reason = f"{subject} cannot load the {fmt} format{of_model}."
        if kind is NoticeKind.OVERRIDE_UNUSABLE:
            return EngineChoice(None, kind, f"{reason} Choose an engine that loads {fmt}.")
    elif not serves:
        reason = f"{subject} does not serve this role."
        left = "serves this role"
    else:
        reason = f"{subject} cannot run on this host. {answer.reason}"
        left = "that loads this model can run on this host"
    default = _default(fmt, host, installed, role)
    if default is None:
        candidates = _candidates(fmt, installed, role, host)
        tail = (
            f"No engine is chosen by default. {_choose_hint(candidates)}"
            if candidates
            else f"No installed engine {left}."
        )
        return EngineChoice(None, kind, f"{reason} {tail}")
    if loads and not serves and kind is NoticeKind.PREFERRED_UNUSABLE:
        # A preference that skips a role leaves it on the next engine without a notice.
        return EngineChoice(default, None, None)
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


class NoPublishedPackError(PinError):
    """The adapter records no pack digest: no pack of the engine is published yet."""


def pin_lock(pin: Mapping[str, Any]) -> PinLock:
    """The lock for *pin*; raises ``PinError`` for a missing field or an unusable name.

    A pin whose ``sha256`` is ``None`` raises ``NoPublishedPackError``; an adapter
    with no published pack sets its pack digest to ``None`` and holds no made-up digest.
    """
    if "sha256" in pin and pin["sha256"] is None:
        raise NoPublishedPackError(
            f"No pack of {pin.get('engine')} is published yet. {LLAMA.name} stays in use."
        )
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
        if not (store.root / name.value).is_dir():
            return False
        lock = pin_lock(adapter.pin_record())
        return store.complete_build(lock) is not None
    except NoPublishedPackError:
        return False
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
