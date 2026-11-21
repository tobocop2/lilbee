"""The mlx pack pin file, the build script it names and the launcher of the package."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lilbee.providers.fleet import engine_store
from lilbee.providers.fleet.engine_crates import MPL_PYPI_CHECKSUMS
from lilbee.providers.fleet.engines import mlx

ROOT = Path(__file__).parents[1]
PROJECT = ROOT / "packaging" / "engine-mlx"
LAUNCHER = PROJECT / "lilbee_engine_mlx" / "bin" / "mlx-lm-server"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="the script needs bash")


def _pin(root: Path = ROOT) -> dict[str, str]:
    lines = (root / "engines" / "mlx.env").read_text(encoding="utf-8").splitlines()
    pairs = (line.split("=", 1) for line in lines if line and not line.startswith("#"))
    return {key: value for key, value in pairs}


def _requirements() -> str:
    return (ROOT / _pin()["MLX_REQUIREMENTS"]).read_text(encoding="utf-8")


def test_the_pin_names_a_build_script_and_requirements_that_exist() -> None:
    pin = _pin()
    assert (ROOT / pin["MLX_BUILD_SCRIPT"]).is_file()
    assert (ROOT / pin["MLX_REQUIREMENTS"]).is_file()
    assert pin["MLX_CELL"] == "macos-arm64-metal"


def test_the_pin_builds_the_version_the_adapter_launches() -> None:
    assert _pin()["MLX_LM_VERSION"] == mlx._VERSION


def test_the_pin_floor_is_the_floor_the_adapter_gates_on() -> None:
    pin = _pin()
    assert pin["MLX_FLOOR"] == ".".join(map(str, mlx.MACOS_FLOOR))
    assert pin["MLX_PLATFORM"] == f"macosx_{mlx.MACOS_FLOOR[0]}_0_arm64"
    assert pin["MLX_BACKEND"] == "metal"


def test_the_requirements_pin_mlx_lm_and_the_mlx_files_of_the_floor() -> None:
    pin = _pin()
    text = _requirements()
    assert f"mlx-lm=={pin['MLX_LM_VERSION']} \\\n" in text
    cp = "cp" + pin["MLX_PYTHON"].replace(".", "")
    version, platform = pin["MLX_VERSION"], pin["MLX_PLATFORM"]
    assert f"/mlx-{version}-{cp}-{cp}-{platform}.whl \\\n" in text
    assert f"/mlx_metal-{version}-py3-none-{platform}.whl \\\n" in text


def test_every_requirement_is_pinned_with_a_hash() -> None:
    blocks = re.split(r"\n(?=\S)", _requirements().split("\n", 4)[-1].strip())
    assert len(blocks) > 30
    for block in blocks:
        assert "--hash=sha256:" in block, block.splitlines()[0]


def test_the_pack_source_pointer_is_one_the_store_accepts() -> None:
    text = (PROJECT / "SOURCE-MPL-2.0").read_text(encoding="utf-8")
    assert engine_store._verified_notices([("SOURCE-MPL-2.0", text)]) == {"SOURCE-MPL-2.0"}


def test_each_listed_mpl_package_is_a_requirement_at_its_version() -> None:
    text = _requirements()
    pointer = (PROJECT / "SOURCE-MPL-2.0").read_text(encoding="utf-8")
    for name, version in MPL_PYPI_CHECKSUMS:
        assert f"\n{name}=={version} \\\n" in text
        assert f"/{name}-{version}.tar.gz " in pointer


def _tree(tmp_path: Path, **changes: str) -> Path:
    """A copy of the script, the pin and the requirements; *changes* replace pin values."""
    top = tmp_path / "tree"
    for part in ("tools/wheel-build/mlx_wheel.sh", "engines/mlx.env", _pin()["MLX_REQUIREMENTS"]):
        (top / part).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / part, top / part)
    env = top / "engines" / "mlx.env"
    text = env.read_text(encoding="utf-8")
    for key, value in changes.items():
        text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.MULTILINE)
    env.write_text(text, encoding="utf-8")
    return top


def _stub(folder: Path, name: str, body: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _build(top: Path, tmp_path: Path, host: tuple[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the script with ``uname`` answering *host*; a stub uv records that the run got that far."""
    stubs = tmp_path / "stubs"
    _stub(stubs, "uname", f'[ "$1" = -s ] && echo {host[0]} || echo {host[1]}')
    _stub(stubs, "uv", f'echo "$@" >> {tmp_path / "uv-calls"}; echo "uv 0.0.0"')
    return subprocess.run(
        ["bash", str(top / "tools" / "wheel-build" / "mlx_wheel.sh"), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
        env={**os.environ, "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}"},
    )


@needs_bash
@pytest.mark.parametrize("host", [("Linux", "x86_64"), ("Darwin", "x86_64"), ("Linux", "aarch64")])
def test_the_script_refuses_a_host_that_is_not_apple_silicon_macos(
    tmp_path: Path, host: tuple[str, str]
) -> None:
    done = _build(_tree(tmp_path), tmp_path, host)
    assert done.returncode == 1
    assert f"This host is {host[0]}-{host[1]}." in done.stderr
    assert not (tmp_path / "uv-calls").exists()


@needs_bash
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"MLX_FLOOR": "14.2"}, "MLX_FLOOR 14.2 is not a whole macOS version"),
        ({"MLX_PLATFORM": "macosx_15_0_arm64"}, "is not the tag of MLX_FLOOR 14.0"),
        ({"MLX_LM_VERSION": "0.33.0"}, "mlx-lm 0.33.0 is not the pin of"),
        ({"MLX_VERSION": "0.33.1"}, "pins no mlx 0.33.1 file"),
        (
            {"MLX_FLOOR": "15.0", "MLX_PLATFORM": "macosx_15_0_arm64"},
            "mlx-0.32.4-cp312-cp312-macosx_15_0_arm64.whl, the file of macOS 15.0",
        ),
    ],
)
def test_the_script_refuses_pins_that_disagree_before_it_installs(
    tmp_path: Path, changes: dict[str, str], message: str
) -> None:
    done = _build(_tree(tmp_path, **changes), tmp_path, ("Darwin", "arm64"))
    assert done.returncode == 1
    assert message in done.stderr
    assert not (tmp_path / "uv-calls").exists()


@needs_bash
def test_the_script_refuses_a_uv_other_than_the_pin(tmp_path: Path) -> None:
    done = _build(_tree(tmp_path), tmp_path, ("Darwin", "arm64"))
    assert done.returncode == 1
    assert f"uv {_pin()['MLX_UV_VERSION']} is pinned in engines/mlx.env" in done.stderr
    assert (tmp_path / "uv-calls").read_text(encoding="utf-8") == "--version\n"


@needs_bash
def test_the_python_install_writes_nothing_outside_the_scratch_directory(tmp_path: Path) -> None:
    """uv links an installed Python into ~/.local/bin unless its bin folder is moved."""
    top = _tree(tmp_path)
    shutil.copytree(PROJECT / "lilbee_engine_mlx", top / "packaging/engine-mlx/lilbee_engine_mlx")
    pin = _pin()
    stubs = tmp_path / "stubs"
    calls = tmp_path / "uv-calls"
    _stub(
        stubs,
        "uv",
        f'[ "$1" = --version ] && {{ echo "uv {pin["MLX_UV_VERSION"]}"; exit 0; }}\n'
        f'echo "$* install=$UV_PYTHON_INSTALL_DIR bin=$UV_PYTHON_BIN_DIR" >> {calls}; exit 1',
    )
    _stub(stubs, "delocate-wheel", f"echo {pin['MLX_DELOCATE_VERSION']}")
    _stub(stubs, "uname", '[ "$1" = -s ] && echo Darwin || echo arm64')
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    done = subprocess.run(
        ["bash", str(top / "tools" / "wheel-build" / "mlx_wheel.sh"), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
        env={**os.environ, "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}", "TMPDIR": str(scratch)},
    )
    assert done.returncode == 1
    line = calls.read_text(encoding="utf-8").strip()
    found = re.fullmatch(r"python install 3\.12 install=(\S+) bin=(\S+)", line)
    assert found is not None, line
    for folder in found.groups():
        assert Path(folder).is_relative_to(scratch)


_FAKE_UV = """
case "$1 $2" in
  "--version "*) echo "uv {uv}" ;;
  "python install")
    mkdir -p "$UV_PYTHON_INSTALL_DIR/cp/bin" "$UV_PYTHON_INSTALL_DIR/cp/lib/python3.12"
    printf '#!/bin/sh\\n' > "$UV_PYTHON_INSTALL_DIR/cp/bin/python3.12"
    chmod 755 "$UV_PYTHON_INSTALL_DIR/cp/bin/python3.12"
    cp "$UV_PYTHON_INSTALL_DIR/cp/bin/python3.12" "$UV_PYTHON_INSTALL_DIR/cp/bin/python3" ;;
  "python find") echo "$UV_PYTHON_INSTALL_DIR/cp/bin/python3.12" ;;
  "pip install")
    while [ "$1" != --target ]; do shift; done
    mkdir -p "$2/mlx_metal-{mlx}.dist-info" "$2/numpy/.dylibs"
    echo "Tag: py3-none-macosx_14_0_arm64" > "$2/mlx_metal-{mlx}.dist-info/WHEEL" ;;
esac
"""


@needs_bash
def test_the_script_stops_when_numpy_bundles_libraries_again(tmp_path: Path) -> None:
    """The numpy licence is replaced only because the wheel holds no numpy/.dylibs."""
    top = _tree(tmp_path)
    shutil.copytree(PROJECT / "lilbee_engine_mlx", top / "packaging/engine-mlx/lilbee_engine_mlx")
    pin = _pin()
    stubs = tmp_path / "stubs"
    _stub(stubs, "uv", _FAKE_UV.format(uv=pin["MLX_UV_VERSION"], mlx=pin["MLX_VERSION"]))
    _stub(stubs, "delocate-wheel", f"echo {pin['MLX_DELOCATE_VERSION']}")
    _stub(stubs, "uname", '[ "$1" = -s ] && echo Darwin || echo arm64')
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    done = subprocess.run(
        ["bash", str(top / "tools" / "wheel-build" / "mlx_wheel.sh"), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
        env={**os.environ, "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}", "TMPDIR": str(scratch)},
    )
    assert done.returncode == 1
    assert "numpy bundles libraries in numpy/.dylibs" in done.stderr


@pytest.mark.skipif(shutil.which("sh") is None, reason="the launcher is a sh script")
def test_the_launcher_runs_mlx_lm_server_on_the_bundled_interpreter(tmp_path: Path) -> None:
    package = tmp_path / "engines" / "lilbee_engine_mlx"
    (package / "bin").mkdir(parents=True)
    launcher = package / "bin" / "mlx-lm-server"
    shutil.copy(LAUNCHER, launcher)
    _stub(package / "python" / "bin", "python3", 'echo "$0" "$@"')
    done = subprocess.run(
        [str(launcher), "--model", "/m", "--port", "5"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
        cwd=tmp_path,
    )
    interpreter, *args = done.stdout.split()
    assert Path(interpreter).resolve() == (package / "python" / "bin" / "python3").resolve()
    assert args == ["-I", "-m", "mlx_lm.server", "--model", "/m", "--port", "5"]


def test_the_launcher_is_executable_and_names_the_adapter_tool() -> None:
    assert os.access(LAUNCHER, os.X_OK)
    assert LAUNCHER.name == mlx.MLX.server_process.name
