"""Refuse an engine pack that ``lilbee engine install`` would refuse, with the checks of the install.

Usage:
  check_pack_licences.py terms PLAN   screen the licence that each planned cell declares
  check_pack_licences.py wheels DIR   run the wheel check of the install on each wheel in DIR

The build workflow of the engine-packs repository runs this file from the lilbee tag that it builds.
PLAN is the JSON that its scripts/plan.sh prints: the cells of each engine. A cell that declares
``license`` passes when ``licence_permitted`` accepts the terms with every notice present, which
is the screen that the install runs on a pin. A cell with no ``license`` gets its terms from its
build, and the wheel check reads them.

The wheel check reads each ``lilbee_engine_*.whl`` file in DIR with the lock that the shipped
adapter of its engine gives: the engine, the version, the tool and the libraries. It refuses a
wheel of an engine that lilbee ships no adapter for, and a version that the adapter does not pin.

Each refusal prints one line that names the cell or the wheel and the reason. The exit is 1 when
anything is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lilbee.providers.fleet.engine_record import PackRecordError, wheel_cell
from lilbee.providers.fleet.engine_store import (
    NOTICE_RULES,
    LicenceRefusedError,
    PinLock,
    UnsafeWheelError,
    check_wheel,
    licence_permitted,
)
from lilbee.providers.fleet.engines.choice import shipped_adapter

EVERY_NOTICE = frozenset(rule.file for rule in NOTICE_RULES.values())
WHEEL_GLOB = "lilbee_engine_*.whl"


class PackRefusedError(Exception):
    """lilbee cannot read a wheel as a pack of a shipped engine."""


def refused_terms(plan: Mapping[str, list[Mapping[str, Any]]]) -> list[str]:
    """One line for each planned cell whose declared licence the install screen refuses."""
    return [
        f"{engine} cell {cell['cell']}: lilbee refuses the licence {cell['license']!r}, "
        "because a term is not on its permissive list."
        for engine, cells in plan.items()
        for cell in cells
        if "license" in cell and not licence_permitted(str(cell["license"]), EVERY_NOTICE)
    ]


def wheel_lock(wheel: Path) -> PinLock:
    """The lock that the shipped adapter gives for the engine and version of the file name of *wheel*."""
    try:
        cell = wheel_cell(wheel.name, wheel.name)
    except PackRecordError as exc:
        raise PackRefusedError(str(exc)) from exc
    adapter = shipped_adapter(cell.engine)
    if adapter is None:
        raise PackRefusedError(f"lilbee ships no adapter for the engine {cell.engine}.")
    pin = adapter.pin_record()
    if pin["version"] != cell.version:
        raise PackRefusedError(
            f"the adapter of {cell.engine} pins version {pin['version']}, not {cell.version}."
        )
    backend = cell.key.split("-", 2)[2]
    return PinLock(
        cell.engine, cell.version, backend, {}, (), "", pin["tool"], tuple(pin.get("libraries", ()))
    )


def refused_wheels(folder: Path) -> list[str]:
    """One line for each wheel in *folder* that the install would refuse; one line when it has none."""
    wheels = sorted(folder.glob(WHEEL_GLOB))
    if not wheels:
        return [f"{folder}: no {WHEEL_GLOB} file to check."]
    refused = []
    for wheel in wheels:
        try:
            check_wheel(wheel_lock(wheel), wheel)
        except (PackRefusedError, LicenceRefusedError, UnsafeWheelError) as exc:
            refused.append(f"{wheel.name}: {exc}")
    return refused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="check", required=True)
    sub.add_parser("terms", help="screen the licence of each planned cell").add_argument(
        "plan", type=Path
    )
    sub.add_parser("wheels", help="run the wheel check of the install").add_argument(
        "folder", type=Path
    )
    args = parser.parse_args(argv)
    if args.check == "terms":
        refused = refused_terms(json.loads(args.plan.read_text(encoding="utf-8")))
    else:
        refused = refused_wheels(args.folder)
    for line in refused:
        print(f"::error::{line}", file=sys.stderr)
    if not refused:
        print(f"check_pack_licences {args.check}: lilbee accepts every pack.")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
