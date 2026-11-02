"""An installed record holds the digest of the host cell; every reader compares that digest."""

from __future__ import annotations

import hashlib
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog import runnable
from lilbee.core.config import cfg
from lilbee.providers.fleet.engine_store import (
    EngineConflictError,
    EngineStore,
    engines_root,
)
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import (
    ADAPTERS,
    host_lock,
    installed_engines,
    pin_lock,
)
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.host_caps import collect
from tests.conformance.fake_adapter import FakeAdapter

CELL = "macos-arm64-cpu"
TOP = "7" * 64


class _CellEngine(FakeAdapter):
    """A registered store engine whose pin holds one macOS arm64 cell."""

    name = "mistral.rs"
    formats = frozenset({"gguf"})
    cell_sha = ""

    def pin_record(self) -> Mapping[str, Any]:
        return _pin(self.cell_sha)


def _pin(cell_sha: str) -> dict[str, Any]:
    row = {"url": "https://example.test/x.whl", "sha256": cell_sha, "size": 1, "min_os": None}
    return {
        "engine": "mistral.rs",
        "version": "1.2.3",
        "backend": "cpu",
        "sha256": TOP,
        "platforms": {CELL: row},
        "licenses": ["MIT"],
        "tool": "mistralrs",
        "launch": ["mistralrs"],
        "memory": {"sizing": None, "rule": None},
    }


@pytest.fixture
def mac(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A macOS arm64 host and a build installed from the wheel of its cell."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(cfg, "models_dir", models_dir)
    host = collect(system="Darwin", machine="arm64", cpu_info={}, nvml=None)
    monkeypatch.setattr(runnable, "session_host", lambda: host)
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        tool = zipfile.ZipInfo("pkg/bin/mistralrs")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("pkg-1.dist-info/METADATA", "License-Expression: MIT\n")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert sha != TOP
    adapter = _CellEngine()
    adapter.cell_sha = sha
    monkeypatch.setitem(ADAPTERS, EngineName.MISTRAL, adapter)  # type: ignore[arg-type]
    monkeypatch.setattr(MISTRAL, "pin_record", adapter.pin_record)
    lock = pin_lock(_pin(sha)).for_cell(CELL)
    EngineStore(engines_root(models_dir)).install_from_wheel(lock, wheel)
    return models_dir


def test_a_cell_install_stays_in_the_installed_list(mac: Path) -> None:
    assert installed_engines()[EngineName.MISTRAL] == frozenset({"gguf"})


def test_a_cell_install_completes_the_build_of_the_host_lock(mac: Path) -> None:
    lock = host_lock(ADAPTERS[EngineName.MISTRAL].pin_record(), runnable.session_host())
    assert lock.sha256 != TOP
    assert EngineStore(engines_root(mac)).complete_build(lock) is not None


def test_the_engine_binary_resolves_for_a_cell_install(mac: Path) -> None:
    assert MISTRAL.binary().name == "mistralrs"


def test_the_top_digest_still_conflicts_with_a_cell_record(mac: Path) -> None:
    with pytest.raises(EngineConflictError, match="different wheel"):
        EngineStore(engines_root(mac)).complete_build(pin_lock(_pin(TOP)))
