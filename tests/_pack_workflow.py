"""The build workflow of the engine-packs repository, for the tests that read it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

WORKFLOW_ENV = "ENGINE_PACKS_WORKFLOW"


def pack_workflow() -> Path:
    """The path in ENGINE_PACKS_WORKFLOW; skip the test when it is unset."""
    path = os.environ.get(WORKFLOW_ENV)
    if not path:
        pytest.skip(f"{WORKFLOW_ENV} names the build workflow of the engine-packs repository")
    return Path(path)
