"""The build workflow of the engine-packs repository, for the tests that read it."""

from __future__ import annotations

import json
import os
import re
from importlib import resources
from pathlib import Path

import pytest

WORKFLOW_ENV = "ENGINE_PACKS_WORKFLOW"
_WHEEL = re.compile(
    r"^(?P<dist>[^-]+)-(?P<version>[^-]+)-(?P<build>\d+)\.(?P<backend>\w+)"
    r"-py3-none-(?P<platform>.+)\.whl$"
)


def published_builds(dist: str, version: str, backend: str, platform: str) -> list[int]:
    """The build numbers that the shipped record holds for wheels of this name."""
    text = (
        resources.files("lilbee.providers.fleet").joinpath("engine_packs.json").read_text("utf-8")
    )
    key = (dist, version, backend, platform)
    found = (_WHEEL.match(row["file"]) for row in json.loads(text))
    return [
        int(m["build"])
        for m in found
        if m and (m["dist"], m["version"], m["backend"], m["platform"]) == key
    ]


def pack_workflow() -> Path:
    """The path in ENGINE_PACKS_WORKFLOW; fail when CI is set and it is empty, else skip."""
    path = os.environ.get(WORKFLOW_ENV)
    if path:
        return Path(path)
    message = f"{WORKFLOW_ENV} names the build workflow of the engine-packs repository"
    if os.environ.get("CI"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message)
