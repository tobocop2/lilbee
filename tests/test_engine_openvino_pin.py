"""The OpenVINO runtime pin file and the pack build it names."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lilbee.providers.fleet.engines.openvino import OPENVINO

ROOT = Path(__file__).parents[1]


def _pin() -> dict[str, str]:
    lines = (ROOT / "engines" / "openvino.env").read_text(encoding="utf-8").splitlines()
    pairs = (line.split("=", 1) for line in lines if line and not line.startswith("#"))
    return {key: value for key, value in pairs}


def test_the_pin_holds_the_runtime_and_the_release_cell_only() -> None:
    assert set(_pin()) == {
        "OPENVINO_VERSION",
        "OPENVINO_ARCHIVE_URL",
        "OPENVINO_ARCHIVE_SHA256",
        "OPENVINO_CELL",
        "OPENVINO_BACKEND",
        "OPENVINO_PLATFORM",
        "OPENVINO_BUILD_SCRIPT",
    }


def test_the_pin_names_the_archive_of_its_version_with_a_digest() -> None:
    pin = _pin()
    name = pin["OPENVINO_ARCHIVE_URL"].rsplit("/", 1)[1]
    version = re.escape(pin["OPENVINO_VERSION"])
    assert re.fullmatch(rf"openvino_toolkit_rhel8_{version}\.\S+\.tgz", name)
    assert f"/packages/{pin['OPENVINO_VERSION']}/linux/" in pin["OPENVINO_ARCHIVE_URL"]
    assert re.fullmatch(r"[0-9a-f]{64}", pin["OPENVINO_ARCHIVE_SHA256"])


def test_the_release_cell_is_the_cell_the_adapter_installs() -> None:
    pin = _pin()
    assert pin["OPENVINO_CELL"] == f"linux-x86_64-{pin['OPENVINO_BACKEND']}"
    assert OPENVINO.pin_record()["backend"] == pin["OPENVINO_BACKEND"]
    assert pin["OPENVINO_PLATFORM"] == "manylinux_2_34_x86_64"


def test_the_pin_names_a_build_script_that_builds_its_backend() -> None:
    pin = _pin()
    script = (ROOT / pin["OPENVINO_BUILD_SCRIPT"]).read_text(encoding="utf-8")
    assert f"  {pin['OPENVINO_BACKEND']})" in script
    assert "engines/openvino.env" in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="the script needs bash")
@pytest.mark.parametrize("ruling", [None, "", "yes"])
def test_the_build_stops_before_any_work_without_the_ruling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ruling: str | None
) -> None:
    """No install tool or network is here: a run past the ruling would fail on them, not exit 3."""
    monkeypatch.delenv("OPENVINO_LICENCE_RULING", raising=False)
    if ruling is not None:
        monkeypatch.setenv("OPENVINO_LICENCE_RULING", ruling)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    for key, value in {
        "BACKEND": "openvino",
        "UPSTREAM_REPO": "ggml-org/llama.cpp",
        "UPSTREAM_TAG": "b0",
        "BIN_DIR": str(tmp_path / "bin"),
        "OWNER": "0:0",
        "LICENSE_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    done = subprocess.run(
        ["bash", str(ROOT / _pin()["OPENVINO_BUILD_SCRIPT"])],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert done.returncode == 3
    assert "Intel OpenVINO Distribution License" in done.stderr
    assert not (tmp_path / "bin").exists()
