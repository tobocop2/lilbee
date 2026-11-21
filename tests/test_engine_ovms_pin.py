"""The OpenVINO Model Server pin file and the repackage script it names."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _pin() -> dict[str, str]:
    lines = (ROOT / "engines" / "ovms.env").read_text(encoding="utf-8").splitlines()
    pairs = (line.split("=", 1) for line in lines if line and not line.startswith("#"))
    return {key: value for key, value in pairs}


def test_the_pin_names_a_repackage_script_that_exists() -> None:
    assert (ROOT / _pin()["OVMS_REPACKAGE_SCRIPT"]).is_file()


def test_the_pin_names_the_archive_that_was_measured() -> None:
    pin = _pin()
    assert pin["OVMS_ARCHIVE_URL"].endswith("/ovms_ubuntu24_2026.4.1_python_off.tar.gz")
    assert re.fullmatch(r"[0-9a-f]{64}", pin["OVMS_ARCHIVE_SHA256"])
    assert pin["OVMS_PLATFORM"] == "manylinux_2_38_x86_64"
    assert pin["OVMS_VERSION"] in pin["OVMS_ARCHIVE_URL"]


def test_the_licence_expression_names_the_libgit2_exception() -> None:
    script = (ROOT / _pin()["OVMS_REPACKAGE_SCRIPT"]).read_text(encoding="utf-8")
    assert "GPL-2.0-only WITH GCC-exception-2.0" in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="the script needs bash")
@pytest.mark.parametrize("ruling", [None, "", "yes"])
def test_the_script_stops_before_any_work_without_the_ruling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ruling: str | None
) -> None:
    """The archive does not exist: a run that got past the ruling would fail on it, not exit 3."""
    monkeypatch.delenv("OVMS_LIBGIT2_RULING", raising=False)
    if ruling is not None:
        monkeypatch.setenv("OVMS_LIBGIT2_RULING", ruling)
    # A bypass keyed on a CI variable must not open the gate either.
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("WHEEL_VERSION", "2026.4.1")
    out = tmp_path / "out"
    done = subprocess.run(
        [
            "bash",
            str(ROOT / _pin()["OVMS_REPACKAGE_SCRIPT"]),
            str(tmp_path / "no-such-archive.tar.gz"),
            str(tmp_path),
            "manylinux_2_38_x86_64",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 3
    assert "libgit2" in done.stderr
    assert not out.exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="the script needs bash")
def test_the_script_passes_the_ruling_with_the_accepted_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the ruling set, the run goes on and fails later, on the archive that is not there."""
    monkeypatch.setenv("OVMS_LIBGIT2_RULING", "accepted")
    monkeypatch.setenv("WHEEL_VERSION", "2026.4.1")
    done = subprocess.run(
        [
            "bash",
            str(ROOT / _pin()["OVMS_REPACKAGE_SCRIPT"]),
            str(tmp_path / "no-such-archive.tar.gz"),
            str(tmp_path),
            "manylinux_2_38_x86_64",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode not in (0, 3)
    assert "libgit2" not in done.stderr


@pytest.mark.skipif(shutil.which("sh") is None, reason="the licence step needs sh")
def test_the_wheel_holds_no_gpl_or_lgpl_text_but_the_one_of_libgit2(tmp_path: Path) -> None:
    server = tmp_path / "ovms"
    texts = server / "thirdparty-licenses"
    texts.mkdir(parents=True)
    (server / "LICENSE").write_text("Apache License", encoding="utf-8")
    gnu = "GNU GENERAL PUBLIC LICENSE\nVersion 2, June 1991\n"
    lesser = "GNU LESSER GENERAL PUBLIC LICENSE\nVersion 2.1, February 1999\n"
    bodies = {
        "espeak-ng.LICENSE.txt": gnu,
        "ngsolve.LICENSE.txt": lesser,
        "ittnotify-GPL-2.0-only.txt": gnu,
        "ittnotify-BSD-3-Clause.txt": "Redistribution and use in source and binary forms",
        "libgt2.LICENSE.txt": gnu,
        "zlib.LICENSE.txt": "zlib licence",
    }
    for name, body in bodies.items():
        (texts / name).write_text(body, encoding="utf-8")
    project = tmp_path / "project"
    script = ROOT / "tools" / "wheel-build" / "ovms_licences.sh"
    subprocess.run(["sh", str(script), str(server), str(project)], check=True, timeout=60)
    shipped = {path.name: path.read_text(encoding="utf-8") for path in project.rglob("*.txt")}
    copyleft = {name for name, text in shipped.items() if text.startswith("GNU ")}
    assert copyleft == {"libgt2.LICENSE.txt"}
    assert "ittnotify-BSD-3-Clause.txt" in shipped
    assert (project / "LICENSE-ovms").is_file()


def test_the_script_puts_libtbb_where_every_library_finds_it() -> None:
    script = (ROOT / _pin()["OVMS_REPACKAGE_SCRIPT"]).read_text(encoding="utf-8")
    assert 'cp "${tbb}/libtbb.so.12" "${lib}/libtbb.so.12"' in script
    assert "patchelf --set-rpath '$ORIGIN'" in script
