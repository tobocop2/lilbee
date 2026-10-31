"""The pure rule that picks the engine for one model format on one host."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum

from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.engines.base import EngineName

AUTO = "auto"
_DARWIN = "darwin"
_ARM64 = "arm64"

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


class NoEngineError(Exception):
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
            raise NoEngineError(self.notice)
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


def choose_engine(
    *,
    fmt: ModelFormat,
    host: HostFacts,
    installed: InstalledEngines,
    preferred: str,
    override: str | None,
) -> EngineChoice:
    """Pick the engine for *fmt*: the override, else the preference, else the host default.

    The result never names an engine that cannot load *fmt*. A preference that
    is unusable gives the host default, with a notice. An override that is
    installed but cannot load *fmt* gives no engine.
    """
    request = _request(preferred, override)
    if request is None:
        default = _default(fmt, host, installed)
        if default is not None:
            return EngineChoice(default, None, None)
        text = f"No installed engine loads the {fmt} format. Install an engine that loads {fmt}."
        return EngineChoice(None, NoticeKind.NONE, text)
    name, kind = request
    if name in installed and fmt in installed[EngineName(name)]:
        return EngineChoice(EngineName(name), None, None)
    if name not in installed:
        reason = f"Engine {name} is not installed. Install {name} to use it."
    else:
        reason = f"Engine {name} cannot load the {fmt} format."
        if kind is NoticeKind.OVERRIDE_UNUSABLE:
            return EngineChoice(None, kind, f"{reason} Choose an engine that loads {fmt}.")
    default = _default(fmt, host, installed)
    if default is None:
        return EngineChoice(None, kind, f"{reason} No installed engine loads the {fmt} format.")
    return EngineChoice(default, kind, f"{reason} Using {default}.")
