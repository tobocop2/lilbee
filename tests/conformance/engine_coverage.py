"""Every engine cell lilbee ships, as typed rows read from the coverage table the package ships."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from lilbee.providers.fleet.engines import proof
from tests.conformance.compare_record import Step
from tests.conformance.compare_verdicts import Oracle


class Coverage(StrEnum):
    """What the conformance suite holds for one engine cell and device."""

    COMPARED = "compared"
    RECORDED = "recorded"
    """A hardware recording under fixtures/engines; no comparison record yet."""
    DEFERRED = "deferred"
    EXCLUDED = "excluded"


class Route(StrEnum):
    """How the engine of a cell reaches a user, read from the adapter and the install path."""

    PACK = "pack"
    """The pack table has a pack; `engine install` fetches it."""
    USER_INSTALLED = "user-installed"
    """The user installs the engine; lilbee finds its binary. `engine install` refuses."""
    UNPUBLISHED = "unpublished"
    """The adapter records no pack digest; `engine install` refuses."""
    PRESET = "preset"
    """A preset serves it; the fleet has no such engine."""


class Device(StrEnum):
    """The hardware a cell runs its work on."""

    CPU = "cpu"
    CUDA = "cuda"
    METAL = "metal"
    VULKAN = "vulkan"
    ROCM = "rocm"
    INTEL_IGPU = "intel-igpu"
    INTEL_ARC = "intel-arc"
    NPU = "npu"


# An engine served by a preset, not by the fleet; it has no EngineName member and no pack.
INFINITY = "infinity"
PRESET_CELL = "preset"

LINUX_CPU = "linux-x86_64-cpu"


@dataclass(frozen=True)
class CoverageRow:
    """One engine cell and device with its status and the reason for it."""

    engine: str
    cell: str
    device: Device
    status: Coverage
    route: Route
    reason: str
    host: str = ""
    owner_decision: str = ""
    record: str = ""
    oracle: Oracle = Oracle.LLAMA
    steps: tuple[Step, ...] = ()
    """The steps a COMPARED row claims; its record must hold each one at the verdict it needs."""
    device_label: str = ""
    """The device as a user reads it; the experimental label names it."""


def _typed(row: Mapping[str, Any]) -> CoverageRow:
    return CoverageRow(
        row["engine"],
        row["cell"],
        Device(row["device"]),
        Coverage(row["status"]),
        Route(row["route"]),
        row["reason"],
        host=row["host"],
        owner_decision=row["owner_decision"],
        record=row["record"],
        oracle=Oracle(row["oracle"]),
        steps=tuple(Step(s) for s in row["steps"]),
        device_label=row["device_label"],
    )


COVERAGE_ROWS: tuple[CoverageRow, ...] = tuple(_typed(row) for row in proof.coverage())
