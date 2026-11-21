"""The OpenVINO runtime pin file and the pack build it names."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lilbee.providers.fleet.engine_store import _is_copyleft_file
from lilbee.providers.fleet.engines.openvino import OPENVINO

ROOT = Path(__file__).parents[1]
_SCRIPT = "tools/wheel-build/llama_gpu_manylinux.sh"
_PACK_FOLDER = "lilbee_engine_llama_cpp-1.12032.1.dist-info/licenses/THIRD-PARTY/openvino"
_NOTICE = "onetbb_third-party-programs.txt"
# Item 4 of the oneTBB notice as OpenVINO ships it: the GPL-3.0 text under a GPL title.
_OPENVINO_NOTICE = (
    "4.  Mateusz Kwiatkowski Workaround for bug 62258 in libstdc++\n\n"
    "    GPL 3.0 with GCC Runtime Library Exception 3.1\n\n"
    "                    GNU GENERAL PUBLIC LICENSE\n"
    "                       Version 3, 29 June 2007\n"
)
# Item 4 of the notice in the oneTBB source tree: public domain, with a caution about the GPL.
_ONETBB_NOTICE = (
    "4.  Mateusz Kwiatkowski Workaround for bug 62258 in libstdc++\n\n"
    "    * I hereby renounce all copyright to this file and my rights resulting from    *\n"
    "    * it, to the broadest extent permitted by law. It may be treated as public     *\n"
    "    * domain.                                                                      *\n"
    "    * However, as this file interfaces with GCC internal ABI, it may be subject to *\n"
    "    * the terms and conditions of the GNU General Public License. Please consult   *\n"
    "    * the GCC licensing terms and/or a lawyer for details.                         *\n"
)


def _pin() -> dict[str, str]:
    lines = (ROOT / "engines" / "openvino.env").read_text(encoding="utf-8").splitlines()
    pairs = (line.split("=", 1) for line in lines if line and not line.startswith("#"))
    return {key: value for key, value in pairs}


def _script() -> str:
    return (ROOT / _SCRIPT).read_text(encoding="utf-8")


def test_the_pin_holds_the_runtime_the_notice_and_the_release_cell_only() -> None:
    assert set(_pin()) == {
        "OPENVINO_VERSION",
        "OPENVINO_WHEEL_URL",
        "OPENVINO_WHEEL_SHA256",
        "ONETBB_VERSION",
        "ONETBB_NOTICE_URL",
        "ONETBB_NOTICE_SHA256",
        "ONETBB_LICENSE_URL",
        "ONETBB_LICENSE_SHA256",
        "OPENVINO_CELL",
        "OPENVINO_BACKEND",
        "OPENVINO_PLATFORM",
        "OPENVINO_BUILD_SCRIPT",
    }


def test_the_pin_names_the_pypi_wheel_of_its_version_with_a_digest() -> None:
    pin = _pin()
    name = pin["OPENVINO_WHEEL_URL"].rsplit("/", 1)[1]
    version = re.escape(pin["OPENVINO_VERSION"])
    assert pin["OPENVINO_VERSION"] == "2026.4.1"
    assert re.fullmatch(rf"openvino-{version}-\d+-cp312-cp312-manylinux_2_28_x86_64\.whl", name)
    assert pin["OPENVINO_WHEEL_URL"].startswith("https://files.pythonhosted.org/packages/")
    assert pin["OPENVINO_WHEEL_SHA256"] == (
        "6b4375c17ddcac83a5180349e2e2bb811185c261066e2a920659892d58ef0e3b"
    )


def test_the_pin_names_the_onetbb_notice_and_licence_of_the_wheels_libtbb() -> None:
    pin = _pin()
    tag = f"https://raw.githubusercontent.com/uxlfoundation/oneTBB/v{pin['ONETBB_VERSION']}/"
    assert pin["ONETBB_VERSION"] == "2021.13.1"
    assert pin["ONETBB_NOTICE_URL"] == tag + "third-party-programs.txt"
    assert pin["ONETBB_LICENSE_URL"] == tag + "LICENSE.txt"
    assert pin["ONETBB_NOTICE_SHA256"] == (
        "1634857eb058dce5278c57a832b8974190785145f4be9acdbdb511f3a0c4091c"
    )
    assert pin["ONETBB_LICENSE_SHA256"] == (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    )


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


def test_the_build_checks_the_digest_of_every_file_it_downloads() -> None:
    script = _script()
    for name in ("OPENVINO_WHEEL", "ONETBB_NOTICE", "ONETBB_LICENSE"):
        assert f'"${{{name}_URL}}"' in script
        assert f"${{{name}_SHA256}}" in script
    assert "sha256sum -c" in script


def test_the_build_takes_the_runtime_and_tbb_from_the_wheel() -> None:
    script = _script()
    assert 'export OpenVINO_DIR="${ov_root}/cmake"' in script
    assert 'export TBB_DIR="${ov_root}/cmake"' in script
    assert "OPENVINO_ARCHIVE" not in script
    assert "runtime/3rdparty" not in script


def test_the_build_ships_the_onetbb_notice_of_onetbb_and_not_the_one_of_openvino() -> None:
    script = _script()
    assert f'"${{LICENSE_DIR}}/THIRD-PARTY/openvino/{_NOTICE}"' in script
    assert "for f in runtime-third-party-programs.txt onednn_third-party-programs.txt; do" in script
    assert f"{_NOTICE} onednn" not in script
    # The source of each copy: the oneTBB tag's files, never the wheel's GPL-titled notice.
    assert f'cp "${{ov_tbb_notice}}" "${{LICENSE_DIR}}/THIRD-PARTY/openvino/{_NOTICE}"' in script
    assert 'cp "${ov_tbb_licence}" "${LICENSE_DIR}/LICENSE-oneTBB"' in script


def test_the_build_names_no_eula_and_asks_for_no_ruling() -> None:
    script = _script()
    for text in ("OPENVINO_LICENCE_RULING", "EULA", "Distribution License", "exit 3"):
        assert text not in script


@pytest.mark.skipif(shutil.which("bash") is None, reason="the script needs bash")
def test_the_build_is_not_refused_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No install tool or network is here: the build fails on them, not on a refusal."""
    monkeypatch.delenv("OPENVINO_LICENCE_RULING", raising=False)
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
    assert done.returncode != 3
    assert "licence" not in done.stderr.lower()


def _read(text: str) -> bytes:
    return text.encode("utf-8")


def test_the_install_check_refuses_the_notice_that_openvino_ships() -> None:
    name = f"{_PACK_FOLDER}/{_NOTICE}"
    assert _is_copyleft_file(name, lambda: _read(_OPENVINO_NOTICE)) is True


def test_the_install_check_passes_the_notice_that_onetbb_ships() -> None:
    name = f"{_PACK_FOLDER}/{_NOTICE}"
    assert _is_copyleft_file(name, lambda: _read(_ONETBB_NOTICE)) is False
