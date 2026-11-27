"""The build workflow of the engine-packs repository, for the tests that read it."""

from __future__ import annotations

import json
import os
import re
from importlib import resources
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

WORKFLOW_ENV = "ENGINE_PACKS_WORKFLOW"
CELLS_ENV = "ENGINE_PACKS_CELLS"
Cell = tuple[str, str, str, int]  # name, backend, platform, build number
_WHEEL = re.compile(
    r"^(?P<dist>[^-]+)-(?P<version>[^-]+)-(?P<build>\d+)\.(?P<backend>\w+)"
    r"-py3-none-(?P<platform>.+)\.whl$"
)


def shipped_rows() -> list[dict[str, Any]]:
    """The rows of the shipped record of engine packs."""
    record = resources.files("lilbee.providers.fleet").joinpath("engine_packs.json")
    return cast(list[dict[str, Any]], json.loads(record.read_text(encoding="utf-8")))


def _wheel_key(file: str) -> tuple[str, str, str, str] | None:
    """The distribution, version, backend and platform of a wheel file name, or None."""
    m = _WHEEL.match(file)
    return (m["dist"], m["version"], m["backend"], m["platform"]) if m else None


def published_builds(
    dist: str,
    version: str,
    backend: str,
    platform: str,
    rows: list[dict[str, Any]] | None = None,
) -> list[int]:
    """The build numbers that the rows (the shipped record if None) hold for wheels of this name."""
    key = (dist, version, backend, platform)
    return [
        int(cast(re.Match[str], _WHEEL.match(row["file"]))["build"])
        for row in (shipped_rows() if rows is None else rows)
        if _wheel_key(row["file"]) == key
    ]


def cells_below_the_record(
    dist: str, version: str, cells: list[Cell], rows: list[dict[str, Any]] | None = None
) -> list[str]:
    """The names of the cells that build a number under a row of the same wheel name.

    A cell that equals its row is allowed: the publish step refuses an existing file name, so one
    name never holds two wheels, and the rows are rewritten to the cell numbers after a publish.
    """
    below = []
    for name, backend, platform, build in cells:
        published = published_builds(dist, version, backend, platform, rows)
        assert published, name
        if build < max(published):
            below.append(name)
    return below


def record_after_publish(
    dist: str, version: str, cells: list[Cell], rows: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """A copy of the rows in which the wheel of each cell replaces the row of that cell."""
    after = list(shipped_rows() if rows is None else rows)
    for _name, backend, platform, build in cells:
        key = (dist, version, backend, platform)
        after = [row for row in after if _wheel_key(row["file"]) != key]
        after.append({"file": f"{dist}-{version}-{build}.{backend}-py3-none-{platform}.whl"})
    return after


def pack_workflow() -> Path:
    """The path in ENGINE_PACKS_WORKFLOW; fail when CI is set and it is empty, else skip."""
    path = os.environ.get(WORKFLOW_ENV)
    if path:
        return Path(path)
    message = f"{WORKFLOW_ENV} names the build workflow of the engine-packs repository"
    if os.environ.get("CI"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


def _pack_engine(engine: str) -> dict[str, Any]:
    """The entry of one engine in ENGINE_PACKS_CELLS; fail in CI if unset, else skip."""
    path = os.environ.get(CELLS_ENV)
    if not path:
        message = f"{CELLS_ENV} names the cells file of the engine-packs repository"
        if os.environ.get("CI"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    engines = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["engines"]
    return cast(dict[str, Any], engines[engine])


def pack_cells(engine: str) -> list[dict[str, Any]]:
    """The cells of one engine in ENGINE_PACKS_CELLS; fail in CI if unset, else skip."""
    return cast(list[dict[str, Any]], _pack_engine(engine)["cells"])


def pack_pins(engine: str) -> dict[str, str]:
    """The pins of one engine in ENGINE_PACKS_CELLS; fail in CI if unset, else skip."""
    return cast(dict[str, str], _pack_engine(engine)["pins"])
