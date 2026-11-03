"""Publish a fixture pack digest for the cell the session host runs."""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.engine_manifest import SCHEMA, cell_key
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS


def publish(
    monkeypatch: pytest.MonkeyPatch, name: EngineName, sha: str, key: str | None = None
) -> None:
    """Make the manifest of *name* list *sha* for cell *key*, else the cell of the session host."""
    if key is None:
        key = cell_key(runnable.session_host(), ADAPTERS[name].pin_record()["backend"])
    assert key is not None
    manifest: dict[str, Any] = {"schema": SCHEMA, "cells": {key: {"sha256": sha}}}
    monkeypatch.setattr(
        engine_manifest, "MANIFESTS", {**engine_manifest.MANIFESTS, name.value: manifest}
    )
