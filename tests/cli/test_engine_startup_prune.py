"""The startup prune keeps every build the shipped pack table lists and removes the rest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.providers.fleet.engine_manifest import MANIFESTS
from lilbee.providers.fleet.engine_store import RECORD_NAME, EngineStore, engines_root

runner = CliRunner()

_UNLISTED_VERSION = "0.0.1"
_ABSENT_ENGINE = "fx"
_DIGEST = "0" * 64
# llama.cpp ships inside the package, so the prune keeps every build of it, listed or not.
_PRUNABLE = ("ik_llama.cpp", "mistral.rs", "tei")

Build = tuple[str, str, str]


def listed_builds() -> set[Build]:
    """Every (engine, version, backend) that a row of the shipped pack table names."""
    return {
        (engine, version, key.split("-", 2)[2])
        for engine, versions in MANIFESTS.items()
        for version, cells in versions.items()
        for key in cells
    }


def place(root: Path, build: Build) -> None:
    """Lay one complete build out as the store does: a folder and its record."""
    engine, version, backend = build
    folder = root / engine / f"{version}-{backend}"
    folder.mkdir(parents=True)
    record = {
        "engine": engine,
        "version": version,
        "backend": backend,
        "sha256": _DIGEST,
        "licences": ["MIT"],
        "installed_at": "2026-11-17T00:00:00+00:00",
        "binary": None,
    }
    (folder / RECORD_NAME).write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    return tmp_path / "models"


def installed(models_dir: Path) -> set[Build]:
    store = EngineStore(engines_root(models_dir))
    return {(b.engine, b.version, b.backend) for b in store.installed()}


def test_the_shipped_table_lists_a_build_of_every_prunable_engine() -> None:
    assert {engine for engine, _, _ in listed_builds()} >= set(_PRUNABLE)


def test_a_start_keeps_every_listed_build_and_removes_the_unlisted_ones(models_dir: Path) -> None:
    listed = listed_builds()
    unlisted = {(engine, _UNLISTED_VERSION, "cpu") for engine in _PRUNABLE}
    unlisted.add((_ABSENT_ENGINE, "1", "cpu"))
    root = engines_root(models_dir)
    for build in sorted(listed | unlisted):
        place(root, build)
    assert installed(models_dir) == listed | unlisted

    result = runner.invoke(app, ["--json", "engine", "list"])

    assert result.exit_code == 0, result.output
    assert installed(models_dir) == listed
    assert json.loads(result.output)["builds"], "the command lists the kept builds"
