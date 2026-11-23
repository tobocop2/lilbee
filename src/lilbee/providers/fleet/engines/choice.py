"""The pure rule that picks the engine for one model format on one host."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from lilbee.catalog.refs import model_format
from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import engines as engines_package
from lilbee.providers.fleet.engine_manifest import (
    UnsupportedOsError,
    host_cell,
    unread_host_refusal,
)
from lilbee.providers.fleet.engine_store import (
    EngineConflictError,
    EngineStore,
    PinLock,
    PlatformAsset,
    UnsupportedPlatformError,
    engines_root,
)
from lilbee.providers.fleet.engines.base import (
    EngineAdapter,
    EngineName,
    RoleFormatChecked,
    UserInstalled,
)
from lilbee.providers.fleet.engines.host_gate import PASS, HostGate
from lilbee.providers.fleet.engines.proof import is_proven
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.roles import WorkerRole

log = logging.getLogger(__name__)

AUTO = "auto"
# The command that shows the installed engines.
LIST_HINT = "Run lilbee engine list to see the installed engines."

# The module attribute through which a module of the engines package exports its adapter.
_ADAPTER_ATTR = "ADAPTER"


def discover_adapters() -> dict[EngineName, EngineAdapter]:
    """The adapter that each module of the engines package exports, by engine name.

    The order is the order of ``EngineName``; it fixes how engines list in a hint.
    """
    found: dict[EngineName, EngineAdapter] = {}
    for info in pkgutil.iter_modules(engines_package.__path__):
        module = importlib.import_module(f"{engines_package.__name__}.{info.name}")
        adapter = vars(module).get(_ADAPTER_ATTR)
        if adapter is None:
            continue
        name = EngineName(adapter.name)
        if name in found:
            raise ValueError(f"The engine {name} is exported by more than one module.")
        found[name] = adapter
    return {name: found[name] for name in EngineName if name in found}


# The adapters this build ships, by engine name.
ADAPTERS: Mapping[EngineName, EngineAdapter] = discover_adapters()

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


def shipped_adapter(name: str) -> EngineAdapter | None:
    """The shipped adapter called *name*, or ``None`` when this build ships none."""
    return next((a for n, a in ADAPTERS.items() if n == name), None)


def refuses_format_for_role(name: str, role: WorkerRole, fmt: str) -> bool:
    """True when a shipped adapter serves *role* but cannot load *fmt* for it."""
    adapter = shipped_adapter(name)
    return isinstance(adapter, RoleFormatChecked) and fmt in adapter.refused_role_formats.get(
        role, ()
    )


def _serves(name: str, role: WorkerRole | None) -> bool:
    """False only for a shipped adapter that does not serve *role*."""
    adapter = shipped_adapter(name)
    return role is None or adapter is None or role in adapter.served_roles


def _accepts(name: str, role: WorkerRole | None, fmt: str) -> bool:
    """``_serves``, and no refusal of *fmt* for *role* by the adapter."""
    return _serves(name, role) and (role is None or not refuses_format_for_role(name, role, fmt))


def host_answer(name: str, host: HostSource) -> HostGate:
    """What the host says about *name*; a bundled engine reads nothing."""
    adapter = shipped_adapter(name)
    return PASS if adapter is None or adapter.bundled else adapter.host_gate(host())


def install_answer(adapter: EngineAdapter, host: HostCaps) -> HostGate:
    """What the host says about installing *adapter*.

    A pin that needs a cell to find its wheel cannot install on a host whose OS or CPU is
    unknown. That refusal is unknown, not known, and its reason is the whole sentence.
    """
    gate = adapter.host_gate(host)
    if not gate.ok or (host.os is not None and host.arch is not None):
        return gate
    pin = adapter.pin_record()
    if pin["platforms"] or pin["sha256"] is None:
        return HostGate(False, unread_host_refusal(adapter.name, host), False)
    return gate


def _is_apple_silicon(caps: HostCaps) -> bool:
    return caps.os is HostOS.MACOS and caps.arch is HostArch.ARM64


def _default(
    fmt: ModelFormat, host: HostSource, installed: InstalledEngines, role: WorkerRole | None
) -> EngineName | None:
    """The first engine in the host's order that is installed, loads *fmt* and serves *role*.

    An engine with a device it has not run on is never chosen here; ``is_proven`` says so.
    """
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
            and is_proven(e)
            and host_answer(e, host).auto_selectable
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


def loadable_refs(refs: Iterable[str], role: WorkerRole | None = None) -> list[str]:
    """The refs in *refs* whose format an installed engine loads for *role*, in order.

    With no *role*, an engine that loads the format counts whatever it serves.
    """
    installed = installed_engines()

    def loadable(ref: str) -> bool:
        fmt = model_format(ref)
        return any(_accepts(name, role, fmt) for name in engines_loading(fmt, installed))

    return [ref for ref in refs if loadable(ref)]


def serving_engines(fmt: str, role: WorkerRole) -> list[EngineName]:
    """The shipped engines whose adapter loads *fmt* and serves *role*, installed or not."""
    return [
        name
        for name, adapter in ADAPTERS.items()
        if fmt in adapter.formats and _accepts(name, role, fmt)
    ]


def _candidates(
    fmt: str, installed: InstalledEngines, role: WorkerRole | None, host: HostSource
) -> list[EngineName]:
    """The installed engines that load *fmt*, serve *role* and can run on the host."""
    return [
        name
        for name in engines_loading(fmt, installed)
        if _accepts(name, role, fmt) and host_answer(name, host).ok
    ]


def _choose_hint(names: list[EngineName], role: WorkerRole | None) -> str:
    """The commands that fix the choice; ``use`` works only once a role holds the model."""
    which = " or ".join(names)
    if role is None:
        return (
            f"Choose {which} with lilbee engine prefer. "
            "It applies to every model without its own choice."
        )
    return (
        f"Choose {which} with lilbee engine use for this model, "
        "or with lilbee engine prefer for every model without its own choice."
    )


def _servers_to_install(
    fmt: ModelFormat, installed: InstalledEngines, role: WorkerRole
) -> list[EngineName]:
    """The shipped engines to install so that *role* has one for *fmt*.

    Empty when no installed engine loads *fmt* (the message then says no engine loads it),
    and empty when an installed engine already serves *role* (the host refuses it, not the role).
    """
    if not engines_loading(fmt, installed):
        return []
    servers = serving_engines(fmt, role)
    return [] if any(name in installed for name in servers) else servers


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
        text += _choose_hint(candidates, role)
    elif role is not None and (missing := _servers_to_install(fmt, installed, role)):
        text = (
            f"No installed engine serves the {fmt} format{of_model} for {role.value}. "
            f"{' or '.join(missing)} serves it. {install_advice(ADAPTERS[missing[0]])}"
        )
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
    gives no engine. A notice names *model* when given. It names *role* only when the
    cause is specific to that role.
    """
    request = _request(preferred, override)
    of_model = f" of {model}" if model else ""
    if request is None:
        return _by_default(fmt, host, installed, role, of_model)
    name, kind = request
    loads = name in engines_loading(fmt, installed)
    serves = _accepts(name, role, fmt)
    answer = host_answer(name, host) if loads and serves else PASS
    if loads and serves and answer.ok:
        return EngineChoice(EngineName(name), None, None)
    # A cause that holds for every role names no role, so one engine gives one notice text.
    subject = f"Engine {name}"
    role_subject = f"{subject} for {role.value}" if role else subject
    hint = ""
    left = f"loads the {fmt} format"
    if name not in installed:
        reason = f"{subject} is not installed."
        hint = f" {LIST_HINT}"
    elif not loads:
        reason = f"{role_subject} cannot load the {fmt} format{of_model}."
        if kind is NoticeKind.OVERRIDE_UNUSABLE:
            return EngineChoice(None, kind, f"{reason} Choose an engine that loads {fmt}.")
    elif role is not None and refuses_format_for_role(name, role, fmt):
        reason = f"{role_subject} cannot load the {fmt} format{of_model}."
        left = f"loads the {fmt} format for {role}"
    elif not serves:
        reason = f"{role_subject} does not serve this role."
        left = "serves this role"
    else:
        reason = f"{subject} cannot run on this host. {answer.reason}"
        left = "that loads this model can run on this host"
    default = _default(fmt, host, installed, role)
    if default is None:
        candidates = _candidates(fmt, installed, role, host)
        tail = (
            f"No engine is chosen by default. {_choose_hint(candidates, role)}"
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
            f"No pack of {pin.get('engine')} is published yet. {EngineName.LLAMA} stays in use."
        )
    try:
        licences = tuple(pin["licenses"])
        cells = {
            key: PlatformAsset(row["url"], row["sha256"], row["size"], row["min_os"])
            for key, row in pin["platforms"].items()
        }
        return PinLock(
            pin["engine"],
            pin["version"],
            pin["backend"],
            cells,
            licences,
            pin["sha256"],
            pin["tool"],
            tuple(pin.get("libraries", ())),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise PinError(f"the pin record is not usable ({exc!r})") from exc


def install_advice(adapter: EngineAdapter) -> str:
    """The advice to get *adapter*: the install command, or the reason install refuses."""
    try:
        pin_lock(adapter.pin_record())
    except NoPublishedPackError:
        return f"No pack of {adapter.name} is published yet, so there is nothing to install."
    except PinError as exc:
        return f"Engine {adapter.name} cannot be installed: {exc}."
    return f"Run lilbee engine install {adapter.name}."


def install_any_advice(host: HostCaps) -> str:
    """The install command of every shipped engine that serves *host* and has a pack to install.

    Falls back to the list command when no engine qualifies.
    """
    commands = []
    for name, adapter in ADAPTERS.items():
        if not install_answer(adapter, host).ok:
            continue
        try:
            pin_lock(adapter.pin_record())
        except PinError:
            continue
        commands.append(f"lilbee engine install {name}")
    return f"Run {' or '.join(commands)}." if commands else LIST_HINT


BUILD_UNREADABLE = (
    OSError,
    EngineConflictError,
    PinError,
    UnsupportedPlatformError,
    UnsupportedOsError,
)
"""The errors that leave an engine's installed build unreadable on this host."""


def host_lock(pin: Mapping[str, Any], host: HostCaps) -> PinLock:
    """The lock for *pin* with the digest of the cell of *host*, as an install records it."""
    lock = pin_lock(pin)
    return lock.for_cell(host_cell(lock, host))


def store_tool(pin: Mapping[str, Any], tool: str, host: HostCaps) -> Path | None:
    """The executable *tool* of the complete build that *pin* names, or ``None`` when absent.

    Raises ``ProviderError`` when the store holds two builds for the pin.
    """
    store = EngineStore(engines_root(cfg.models_dir))
    try:
        build = store.complete_build(host_lock(pin, host))
    except EngineConflictError as exc:
        raise ProviderError(str(exc)) from exc
    except BUILD_UNREADABLE:
        return None
    return next(iter(build.glob(f"*/bin/{tool}*")), None) if build else None


def release_pins(builds: Sequence[tuple[str, str, str, Path]]) -> set[tuple[str, str, str]]:
    """The builds to keep, as (engine, version, backend), for the installed *builds*.

    A build stays when its engine ships in the package, when it has the version
    of the current pin in any backend, or when the pin names no backend. The pin
    itself is always in the set, so a store with only older builds still prunes.
    """
    keep: set[tuple[str, str, str]] = set()
    for adapter in ADAPTERS.values():
        if adapter.bundled:
            keep.update((e, v, b) for e, v, b, _ in builds if e == adapter.name)
            continue
        pin = adapter.pin_record()
        named = isinstance(pin["backend"], str)
        keep.add((pin["engine"], pin["version"], pin["backend"] if named else ""))
        keep.update(
            (e, v, b)
            for e, v, b, _ in builds
            if e == pin["engine"] and (not named or v == pin["version"])
        )
    return keep


def _is_installed(name: EngineName, adapter: EngineAdapter, store: EngineStore) -> bool:
    """True for the engine in the package, or an adapter with a build or a user install.

    A read that fails (unreadable folder, a build of another wheel) leaves that
    engine not installed and is logged; it never touches the engine in the package.
    """
    if adapter.bundled:
        return True
    if isinstance(adapter, UserInstalled) and adapter.user_binary() is not None:
        return True
    from lilbee.catalog.runnable import session_host

    try:
        if not (store.root / name.value).is_dir():
            return False
        lock = host_lock(adapter.pin_record(), session_host())
        return store.complete_build(lock) is not None
    except NoPublishedPackError:
        return False
    except BUILD_UNREADABLE as exc:
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
