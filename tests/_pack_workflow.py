"""The build workflow of the engine-packs repository, for the tests that read it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

WORKFLOW_ENV = "ENGINE_PACKS_WORKFLOW"


def pack_workflow() -> Path:
    """The path in ENGINE_PACKS_WORKFLOW; fail when CI is set and it is empty, else skip."""
    path = os.environ.get(WORKFLOW_ENV)
    if path:
        return Path(path)
    message = f"{WORKFLOW_ENV} names the build workflow of the engine-packs repository"
    if os.environ.get("CI"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message)
