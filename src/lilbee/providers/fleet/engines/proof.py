"""Which devices an engine has run on, and the one sentence that says what is unproven.

An engine with any ``NOT_RUN`` device is never chosen by ``auto``. A row flips to
``RECORDED`` only with a log from a real run on that device. Every surface builds its
text from ``not_run_clause``, so no two surfaces word the claim differently.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum


class HardwareProof(StrEnum):
    """Whether lilbee has a recorded run of an engine on one kind of device."""

    RECORDED = "recorded"
    NOT_RUN = "not_run"


# Device words are what a user reads: "Intel NPU", not a vendor id. No engine here has a
# NOT_RUN device yet, because no engine for an NPU ships. The OpenVINO and Lemonade
# adapters add their rows with the adapter.
HARDWARE_PROOF: Mapping[str, Mapping[str, HardwareProof]] = {}

_DEVICE_JOINER = " and "


def not_run_on(engine: str) -> tuple[str, ...]:
    """The devices *engine* has not run on, in table order; empty when none."""
    rows = HARDWARE_PROOF.get(engine, {})
    return tuple(device for device, proof in rows.items() if proof is HardwareProof.NOT_RUN)


def is_proven(engine: str) -> bool:
    """False when *engine* has any ``NOT_RUN`` device."""
    return not not_run_on(engine)


def not_run_clause(devices: Iterable[str]) -> str:
    """``", not run on X"`` for the devices, or the empty string for none."""
    named = tuple(devices)
    return f", not run on {_DEVICE_JOINER.join(named)}" if named else ""
