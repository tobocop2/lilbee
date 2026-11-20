"""The OpenVINO Model Server pin file and the repackage script it names."""

from __future__ import annotations

import re
from pathlib import Path

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


def test_the_script_refuses_to_bundle_libgit2_without_a_ruling() -> None:
    script = (ROOT / _pin()["OVMS_REPACKAGE_SCRIPT"]).read_text(encoding="utf-8")
    assert 'test "${OVMS_LIBGIT2_RULING:-}" = "accepted"' in script
    assert "GPL-2.0-only WITH GCC-exception-2.0" in script


def test_the_script_puts_libtbb_where_every_library_finds_it() -> None:
    script = (ROOT / _pin()["OVMS_REPACKAGE_SCRIPT"]).read_text(encoding="utf-8")
    assert 'cp "${tbb}/libtbb.so.12" "${lib}/libtbb.so.12"' in script
    assert "patchelf --set-rpath '$ORIGIN'" in script
