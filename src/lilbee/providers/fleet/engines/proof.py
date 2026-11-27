"""The engine coverage table, read from package data, and the experimental label built from it."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any

COVERAGE_FILE: Traversable = resources.files("lilbee.providers.fleet.engines").joinpath(
    "coverage.json"
)
HAS_RECORD = frozenset({"compared", "recorded"})
EXPERIMENTAL_NOTE = (
    "Experimental: built and tested without the real device. If you run it, "
    "report the result at https://github.com/tobocop2/lilbee/issues."
)

_DEVICE_JOINER = " and "


@cache
def coverage() -> tuple[Mapping[str, Any], ...]:
    """Every row of the coverage table: one engine, cell and device with its status."""
    rows: list[Mapping[str, Any]] = json.loads(COVERAGE_FILE.read_text(encoding="utf-8"))["rows"]
    return tuple(rows)


def experimental_devices(engine: str, cell: str | None = None) -> tuple[str, ...]:
    """The device words of *engine* on *cell* (every cell for ``None``) that have no record.

    A cell with no row for *engine* has no record either, so it names itself.
    """
    rows = [r for r in coverage() if r["engine"] == engine and cell in (None, r["cell"])]
    if cell is not None and not rows:
        return (cell,)
    words = (r["device_label"] for r in rows if r["status"] not in HAS_RECORD)
    return tuple(dict.fromkeys(words))


def is_proven(engine: str) -> bool:
    """False when *engine* has a device with no record on any cell."""
    return not experimental_devices(engine)


def experimental_clause(devices: Iterable[str]) -> str:
    """``", experimental: not run on X"`` for the devices, or the empty string for none."""
    named = tuple(devices)
    return f", experimental: not run on {_DEVICE_JOINER.join(named)}" if named else ""
