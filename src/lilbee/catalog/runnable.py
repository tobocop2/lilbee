"""Which catalog rows this host can run, and why a row is hidden."""

from __future__ import annotations

import functools
import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.host_caps import HostCaps, collect, detect

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lilbee.providers.fleet.engines.base import EngineAdapter


class EngineState(StrEnum):
    """Whether an engine is ready, can be fetched, or cannot be had on this host."""

    INSTALLED = "installed"
    INSTALLABLE = "installable"
    UNAVAILABLE = "unavailable"


class HiddenReason(StrEnum):
    """Why a catalog row is hidden."""

    NO_ENGINE = "no_engine"
    HOST_UNSUPPORTED = "host_unsupported"


@dataclass(frozen=True)
class EngineOffer:
    """One engine and whether it can run here."""

    adapter: EngineAdapter
    state: EngineState


@dataclass(frozen=True)
class Hidden:
    """A hidden row's reason and the text every surface shows for it."""

    reason: HiddenReason
    text: str


def default_offers() -> list[EngineOffer]:
    """The shipped engines, read from the engine table."""
    from lilbee.providers.fleet.engines import LLAMA

    return [EngineOffer(LLAMA, EngineState.INSTALLED)]


@functools.cache
def session_host() -> HostCaps:
    """The host answer, read once for the session; a read that fails leaves every field unknown.

    Detection reads ``platform`` (``OSError`` when the OS refuses the query) and parses the
    answers (``ValueError``). Neither may close the catalog: the gates treat an unknown field
    as a pass, so every row stays listed.
    """
    try:
        return detect()
    except (OSError, ValueError) as exc:
        log.warning("Host detection failed, the catalog lists every row: %s", exc)
        unreadable = {"driver": None, "cuda_driver": None, "gpus": None}
        return collect(
            system="", machine="", mac_ver="", cpu_info={}, nvml=unreadable, npu_devices=()
        )


def refresh_session_host() -> HostCaps:
    """Read the host now and make it the session host; an engine install calls this."""
    session_host.cache_clear()
    return session_host()


def _names(adapter: EngineAdapter, formats: Collection[str]) -> bool:
    return bool(adapter.formats & set(formats))


def runnable_formats(
    formats: Collection[str], host: HostCaps, offers: Sequence[EngineOffer]
) -> frozenset[ModelFormat]:
    """The formats of a model that an engine can run on *host*."""
    return frozenset(
        ModelFormat(fmt)
        for fmt in formats
        if any(
            o.state is not EngineState.UNAVAILABLE
            and fmt in o.adapter.formats
            and o.adapter.host_gate(host).ok
            for o in offers
        )
    )


def explain(
    formats: Collection[str], host: HostCaps, offers: Sequence[EngineOffer]
) -> Hidden | None:
    """Why a model with *formats* is hidden on *host*, or ``None`` when it shows."""
    naming = [
        o for o in offers if o.state is not EngineState.UNAVAILABLE and _names(o.adapter, formats)
    ]
    if not naming:
        wanted = ", ".join(sorted(formats))
        return Hidden(HiddenReason.NO_ENGINE, f"no engine for {wanted}")
    gates = [o.adapter.host_gate(host) for o in naming]
    if any(g.ok for g in gates):
        return None
    reasons = "; ".join(dict.fromkeys(g.reason for g in gates if g.reason))
    return Hidden(HiddenReason.HOST_UNSUPPORTED, f"this host cannot run it: {reasons}")


def hidden_reason(
    formats: Collection[str], host: HostCaps, offers: Sequence[EngineOffer]
) -> HiddenReason | None:
    """The reason a model with *formats* is hidden, or ``None`` when it shows."""
    hidden = explain(formats, host, offers)
    return None if hidden is None else hidden.reason


def hidden_summary(texts: Collection[str]) -> str:
    """One line for the hidden rows: the count and each distinct reason; empty when none."""
    if not texts:
        return ""
    return f"{len(texts)} hidden ({'; '.join(sorted(set(texts)))})"
