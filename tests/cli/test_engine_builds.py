"""lilbee engine list and remove, and the startup sweep of stale install folders."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.providers.fleet.engine_store import EngineStore, PinLock, engines_root

runner = CliRunner()


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setenv("LILBEE_ENGINE_DIR", str(tmp_path / "slot"))
    return tmp_path / "models"


def install(models_dir: Path) -> None:
    wheel = models_dir.parent / "fx.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        tool = zipfile.ZipInfo("lilbee_engine_fx/bin/server")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("lilbee_engine_fx-1.dist-info/METADATA", "License-Expression: MIT\n")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    EngineStore(engines_root(models_dir)).install_from_wheel(
        PinLock("fx", "1", "cpu", {}, ("MIT",), sha), wheel
    )


def test_list_shows_nothing_installed(models_dir: Path) -> None:
    result = runner.invoke(app, ["engine", "list"])
    assert result.exit_code == 0
    assert "No engine builds are installed." in result.output


def test_list_shows_each_build_as_json(models_dir: Path) -> None:
    install(models_dir)
    result = runner.invoke(app, ["--json", "engine", "list"])
    assert result.exit_code == 0
    assert json.loads(result.output)["builds"] == [
        {"engine": "fx", "version": "1", "backend": "cpu"}
    ]


def test_list_prints_a_line_per_build(models_dir: Path) -> None:
    install(models_dir)
    assert "fx 1 (cpu)" in runner.invoke(app, ["engine", "list"]).output


def test_remove_deletes_the_build(models_dir: Path) -> None:
    install(models_dir)
    result = runner.invoke(app, ["engine", "remove", "fx", "1", "cpu"])
    assert result.exit_code == 0
    assert "Removed engine fx 1 (cpu)." in result.output
    assert EngineStore(engines_root(models_dir)).installed() == []


def test_remove_json_names_the_build(models_dir: Path) -> None:
    install(models_dir)
    result = runner.invoke(app, ["--json", "engine", "remove", "fx", "1", "cpu"])
    assert json.loads(result.output)["removed"] == "fx 1 (cpu)"


def test_remove_refuses_while_a_launch_holds_the_engine(models_dir: Path, tmp_path: Path) -> None:
    from lilbee.runtime.engine_lock import hold_user_lock

    install(models_dir)
    hold = hold_user_lock(tmp_path / "slot")
    try:
        result = runner.invoke(app, ["engine", "remove", "fx", "1", "cpu"])
    finally:
        hold.release_and_check_last()
    assert result.exit_code == 1
    assert "is in use" in result.output
    assert len(EngineStore(engines_root(models_dir)).installed()) == 1


def test_remove_json_reports_a_missing_build(models_dir: Path) -> None:
    result = runner.invoke(app, ["--json", "engine", "remove", "fx", "9", "cpu"])
    assert result.exit_code == 1
    assert "is not installed" in json.loads(result.output)["error"]


def test_remove_refuses_the_bundled_engine(models_dir: Path) -> None:
    result = runner.invoke(app, ["engine", "remove", "llama.cpp", "bundled", "bundled"])
    assert result.exit_code == 1
    assert "inside the executable" in result.output


def test_a_command_sweeps_a_stale_stage_at_startup(models_dir: Path) -> None:
    stale = engines_root(models_dir) / "fx" / ".stage-old"
    stale.mkdir(parents=True)
    runner.invoke(app, ["engine", "list"])
    assert not stale.exists()


def test_a_command_with_nothing_installed_creates_no_engines_folder(models_dir: Path) -> None:
    runner.invoke(app, ["engine", "list"])
    assert not engines_root(models_dir).exists()


def test_remove_errors_go_to_stderr_not_stdout(models_dir: Path) -> None:
    result = runner.invoke(app, ["engine", "remove", "llama.cpp", "bundled", "bundled"])
    assert result.exit_code == 1
    assert "inside the executable" in result.stderr
    assert result.stdout == ""
