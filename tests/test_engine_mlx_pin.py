"""The mlx pack pin file, the build script it names and the launcher of the package."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from lilbee.providers.fleet import engine_store
from lilbee.providers.fleet.engine_crates import MPL_PYPI_CHECKSUMS
from lilbee.providers.fleet.engines import mlx
from lilbee.providers.roles import WorkerRole

ROOT = Path(__file__).parents[1]
PROJECT = ROOT / "packaging" / "engine-mlx"
LAUNCHER = PROJECT / "lilbee_engine_mlx" / "bin" / "mlx-lm-server"
EMBED_LAUNCHER = PROJECT / "lilbee_engine_mlx" / "bin" / "mlx-embed-server"
SERVER = PROJECT / "lilbee_engine_mlx" / "mlx_embed_server.py"

runs_on_macos = pytest.mark.skipif(
    sys.platform == "win32", reason="the mlx pack script and launchers run only on macOS"
)
needs_bash = pytest.mark.skipif(
    shutil.which("bash") is None or sys.platform == "win32",
    reason="the script needs bash and runs only on macOS",
)


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
    parts = ("tools/wheel-build/mlx_wheel.sh", "engines/mlx.env", _pin()["MLX_REQUIREMENTS"])
    for part in (*parts, "LICENSE"):
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
    """Run the script with ``uname`` answering *host*; a stub uv records each call."""
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
        env={
            **os.environ,
            "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
            "TMPDIR": str(scratch),
        },
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
        env={
            **os.environ,
            "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
            "TMPDIR": str(scratch),
        },
    )
    assert done.returncode == 1
    assert "numpy bundles libraries in numpy/.dylibs" in done.stderr


def test_the_pin_names_the_archive_that_holds_the_interpreter_licences() -> None:
    pin = _pin()
    assert re.fullmatch(rf"{re.escape(pin['MLX_PYTHON'])}\.\d+", pin["MLX_PBS_VERSION"])
    assert re.fullmatch(r"\d{8}", pin["MLX_PBS_TAG"])
    assert re.fullmatch(r"[0-9a-f]{64}", pin["MLX_PBS_SHA256"])


_PSF = "PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2\n"
_TQDM_LINE = (
    "https://files.pythonhosted.org/packages/source/t/tqdm/tqdm-4.70.1.tar.gz "
    "cefd0eca11b2a37a3aee776544d4f4ae913f02688135b5556b8788dfa474afc4"
)
_FAKE_PACK_UV = """
case "$1 $2" in
  "--version "*) echo "uv @UV@" ;;
  "python install")
    mkdir -p "$UV_PYTHON_INSTALL_DIR"
    cp -R "@TEMPLATE@" "$UV_PYTHON_INSTALL_DIR/cp" ;;
  "python find") echo "$UV_PYTHON_INSTALL_DIR/cp/bin/python3.12" ;;
  "pip install")
    while [ "$1" != --target ]; do shift; done
    site="$2"
    package() {
      mkdir -p "$site/$1.dist-info/licenses"
      printf 'Name: x\\nLicense-Expression: %s\\n' "$2" > "$site/$1.dist-info/METADATA"
      echo "licence text of $1" > "$site/$1.dist-info/licenses/LICENSE"
    }
    package mlx_lm-@LM@ MIT
    package mlx-@MLX@ MIT
    package mlx_metal-@MLX@ MIT
    package tqdm-4.70.1 "MPL-2.0 AND MIT"
    [ -z "$FAKE_PIP_REQUIREMENT" ] || mkdir -p "$site/pip"
    [ -z "$FAKE_GPL_EMBED" ] || mkdir -p "$site/$FAKE_GPL_EMBED"
    echo "Tag: py3-none-macosx_14_0_arm64" > "$site/mlx_metal-@MLX@.dist-info/WHEEL" ;;
esac
"""


def _pack_tree(
    tmp_path: Path, *, sources: list[str], version: str | None = None, **changes: str
) -> tuple[Path, dict[str, str]]:
    """A build tree and the environment of a build that stops at `wheel pack`.

    The stubs stand in for uv (it installs a small tree), curl (it serves a tar of the metadata
    folder), zstd and python3 (it fails, so the wheel is never packed).
    """
    pin = _pin()
    version = version or pin["MLX_PBS_VERSION"]
    template = tmp_path / "template" / "cp"
    (template / "bin").mkdir(parents=True)
    (template / "bin" / "python3.12").write_text("", encoding="utf-8")
    (template / "bin" / "python3").symlink_to(sys.executable)
    (template / "lib" / "python3.12").mkdir(parents=True)
    (template / "lib" / "python3.12" / "LICENSE.txt").write_text(_PSF, encoding="utf-8")
    packages = template / "lib" / "python3.12" / "site-packages"
    (packages / "pip" / "_vendor" / "certifi").mkdir(parents=True)
    (packages / "pip" / "_vendor" / "certifi" / "cacert.pem").write_text("pem", encoding="utf-8")
    (packages / "pip-26.3.dist-info" / "licenses").mkdir(parents=True)
    (packages / "pip-26.3.dist-info" / "METADATA").write_text(
        "Name: pip\nLicense-Expression: MIT\n", encoding="utf-8"
    )
    (packages / "pip-26.3.dist-info" / "licenses" / "LICENSE").write_text("MIT", encoding="utf-8")
    (template / "lib" / "python3.12" / "ensurepip" / "_bundled").mkdir(parents=True)
    (
        template / "lib" / "python3.12" / "ensurepip" / "_bundled" / "pip-26.3-py3-none-any.whl"
    ).write_text("wheel", encoding="utf-8")
    (template / "include" / "python3.12").mkdir(parents=True)
    (template / "include" / "python3.12" / "patchlevel.h").write_text(
        f'#define PY_VERSION "{pin["MLX_PBS_VERSION"]}"\n', encoding="utf-8"
    )
    folder = tmp_path / "archive" / "python"
    (folder / "licenses").mkdir(parents=True)
    (folder / "licenses" / "LICENSE.bzip2.txt").write_text("bzip2 licence", encoding="utf-8")
    extension = {
        "licenses": ["bzip2-1.0.6"],
        "license_paths": ["licenses/LICENSE.bzip2.txt"],
        "links": [{"name": "bz2", "path_static": "build/lib/libbz2.a"}],
    }
    (folder / "PYTHON.json").write_text(
        json.dumps(
            {
                "python_version": version,
                "target_triple": "aarch64-apple-darwin",
                "build_info": {"extensions": {"_bz2": [extension]}},
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "full.tar"
    with tarfile.open(archive, "w") as bundle:
        bundle.add(folder, arcname="python")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    top = _tree(tmp_path, **{"MLX_PBS_SHA256": digest, **changes})
    project = top / "packaging" / "engine-mlx"
    shutil.copytree(PROJECT / "lilbee_engine_mlx", project / "lilbee_engine_mlx")
    (project / "supplied-licences.txt").write_text("# nothing supplied\n", encoding="utf-8")
    (project / "SOURCE-MPL-2.0").write_text("\n".join(sources) + "\n", encoding="utf-8")
    for tool in ("collect_python_licences.py", "mpl_pypi_notice.py"):
        shutil.copy(ROOT / "tools" / "wheel-build" / tool, top / "tools" / "wheel-build" / tool)
    stubs = tmp_path / "stubs"
    uv = _FAKE_PACK_UV.replace("@UV@", pin["MLX_UV_VERSION"]).replace("@TEMPLATE@", str(template))
    _stub(
        stubs, "uv", uv.replace("@LM@", pin["MLX_LM_VERSION"]).replace("@MLX@", pin["MLX_VERSION"])
    )
    _stub(stubs, "delocate-wheel", f"echo {pin['MLX_DELOCATE_VERSION']}")
    _stub(stubs, "uname", '[ "$1" = -s ] && echo Darwin || echo arm64')
    calls = tmp_path / "curl-calls"
    _stub(
        stubs,
        "curl",
        f'echo "$@" >> {calls}; while [ "$1" != -o ]; do shift; done; cp {archive} "$2"',
    )
    _stub(stubs, "zstd", 'cat "$2"')
    _stub(stubs, "python3", "exit 1")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {**os.environ, "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}", "TMPDIR": str(scratch)}
    return top, env


def _run_pack(top: Path, env: dict[str, str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(top / "tools" / "wheel-build" / "mlx_wheel.sh"), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
        env=env,
    )


def _dist(tmp_path: Path) -> Path:
    (root,) = (tmp_path / "scratch").glob("mlx-wheel.*/pack/lilbee_engine_mlx-*")
    return next(root.glob("*.dist-info"))


needs_shasum = pytest.mark.skipif(shutil.which("shasum") is None, reason="the script needs shasum")


@needs_bash
@needs_shasum
def test_the_wheel_carries_the_licences_and_terms_of_the_interpreter_libraries(
    tmp_path: Path,
) -> None:
    top, env = _pack_tree(tmp_path, sources=[_TQDM_LINE])
    done = _run_pack(top, env, tmp_path)
    assert done.returncode == 1, done.stderr
    dist = _dist(tmp_path)
    licences = dist / "licenses" / "THIRD-PARTY" / "mlx" / "python"
    assert (licences / "LICENSE.bzip2.txt").read_text(encoding="utf-8") == "bzip2 licence"
    metadata = (dist / "METADATA").read_text(encoding="utf-8")
    assert "License-Expression: MIT AND MPL-2.0 AND PSF-2.0 AND bzip2-1.0.6\n" in metadata
    assert "License-File: THIRD-PARTY/mlx/python/LICENSE.bzip2.txt\n" in metadata
    url = (tmp_path / "curl-calls").read_text(encoding="utf-8")
    pin = _pin()
    assert (
        f"{pin['MLX_PBS_TAG']}/cpython-{pin['MLX_PBS_VERSION']}%2B{pin['MLX_PBS_TAG']}"
        "-aarch64-apple-darwin-pgo%2Blto-full.tar.zst"
    ) in url


def _interpreter_tree(tmp_path: Path) -> Path:
    (root,) = (tmp_path / "scratch").glob("mlx-wheel.*/pack/lilbee_engine_mlx-*")
    return root / "lilbee_engine_mlx" / "python" / "lib" / "python3.12"


@needs_bash
@needs_shasum
def test_the_build_drops_pip_and_its_bundled_wheel_from_the_interpreter_tree(
    tmp_path: Path,
) -> None:
    top, env = _pack_tree(tmp_path, sources=[_TQDM_LINE])
    assert _run_pack(top, env, tmp_path).returncode == 1
    tree = _interpreter_tree(tmp_path)
    assert not (tree / "site-packages" / "pip").exists()
    assert not list((tree / "site-packages").glob("pip-*.dist-info"))
    assert not (tree / "ensurepip" / "_bundled").exists()
    assert (tree / "LICENSE.txt").exists()


@needs_bash
@needs_shasum
def test_the_script_stops_when_a_pip_folder_is_in_the_tree_at_the_collector(
    tmp_path: Path,
) -> None:
    top, env = _pack_tree(tmp_path, sources=[_TQDM_LINE])
    done = _run_pack(top, {**env, "FAKE_PIP_REQUIREMENT": "1"}, tmp_path)
    assert done.returncode == 1
    assert "pip is in the interpreter tree" in done.stderr
    assert not (tmp_path / "out" / "terms.txt").exists()


@needs_bash
@needs_shasum
@pytest.mark.parametrize("folder", ["mlx_embeddings", "mlx_embeddings-0.1.0.dist-info"])
def test_the_script_stops_when_mlx_embeddings_is_in_the_tree_at_the_collector(
    tmp_path: Path, folder: str
) -> None:
    top, env = _pack_tree(tmp_path, sources=[_TQDM_LINE])
    done = _run_pack(top, {**env, "FAKE_GPL_EMBED": folder}, tmp_path)
    assert done.returncode == 1
    assert "mlx-embeddings is in the interpreter tree" in done.stderr
    assert f"site-packages/{folder})" in done.stderr
    assert not (tmp_path / "out" / "terms.txt").exists()


@needs_bash
@needs_shasum
def test_the_pack_holds_both_launchers_the_embed_server_and_the_lilbee_licence(
    tmp_path: Path,
) -> None:
    top, env = _pack_tree(tmp_path, sources=[_TQDM_LINE])
    for launcher in (top / "packaging/engine-mlx/lilbee_engine_mlx/bin").iterdir():
        launcher.chmod(0o644)
    done = _run_pack(top, env, tmp_path)
    assert done.returncode == 1, done.stderr
    (root,) = (tmp_path / "scratch").glob("mlx-wheel.*/pack/lilbee_engine_mlx-*")
    package = root / "lilbee_engine_mlx"
    for name in ("mlx-lm-server", "mlx-embed-server"):
        assert os.access(package / "bin" / name, os.X_OK), name
    assert (package / "mlx_embed_server.py").is_file()
    licence = _dist(tmp_path) / "licenses" / "LICENSE-lilbee"
    assert licence.read_text(encoding="utf-8") == (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "License-File: LICENSE-lilbee\n" in (_dist(tmp_path) / "METADATA").read_text(
        encoding="utf-8"
    )


def test_the_requirements_hold_no_mlx_embeddings() -> None:
    text = _requirements().lower().replace("_", "-")
    assert "mlx-embeddings" not in text
    assert "mlx-vlm" not in text


def test_the_embed_server_imports_only_mlx_mlx_lm_numpy_and_the_standard_library() -> None:
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
    outside = {name for name in roots if name not in sys.stdlib_module_names}
    assert outside == {"mlx", "mlx_lm", "numpy"}


@needs_bash
@needs_shasum
def test_the_script_stops_when_the_archive_is_not_the_pinned_one(tmp_path: Path) -> None:
    top, env = _pack_tree(tmp_path, sources=[_TQDM_LINE], MLX_PBS_SHA256="0" * 64)
    done = _run_pack(top, env, tmp_path)
    assert done.returncode == 1
    assert "does not match MLX_PBS_SHA256" in done.stderr
    assert not list((tmp_path / "scratch").glob("mlx-wheel.*/pack/*/*.dist-info/METADATA"))


@needs_bash
@needs_shasum
def test_the_script_stops_when_the_interpreter_is_not_the_archive_version(tmp_path: Path) -> None:
    top, env = _pack_tree(tmp_path, sources=[_TQDM_LINE], version="3.12.99")
    done = _run_pack(top, env, tmp_path)
    assert done.returncode == 1
    assert "is not the one that PYTHON.json describes" in done.stderr
    assert "the metadata is for 3.12.99" in done.stderr


@needs_bash
@needs_shasum
def test_the_notice_names_only_the_mpl_packages_of_the_tree(tmp_path: Path) -> None:
    extra = _TQDM_LINE.replace("tqdm", "gone").replace("4.70.1", "1.0")
    top, env = _pack_tree(tmp_path, sources=[extra, _TQDM_LINE])
    assert _run_pack(top, env, tmp_path).returncode == 1
    notice = _dist(tmp_path) / "licenses" / "SOURCE-MPL-2.0"
    assert notice.read_text(encoding="utf-8") == _TQDM_LINE + "\n"


@needs_bash
@needs_shasum
def test_the_script_stops_on_an_mpl_package_with_no_source_archive(tmp_path: Path) -> None:
    other = _TQDM_LINE.replace("tqdm", "gone").replace("4.70.1", "1.0")
    top, env = _pack_tree(tmp_path, sources=[other])
    done = _run_pack(top, env, tmp_path)
    assert done.returncode == 1
    assert "the list of source archives has no line for tqdm-4.70.1" in done.stderr
    assert "an MPL-2.0 package of the tree has no source archive" in done.stderr
    assert not (_dist(tmp_path) / "licenses" / "SOURCE-MPL-2.0").exists()


@needs_bash
@needs_shasum
def test_the_script_stops_when_zstd_is_not_installed(tmp_path: Path) -> None:
    top, env = _pack_tree(tmp_path, sources=[_TQDM_LINE])
    bare = tmp_path / "bare"
    bare.mkdir()
    tools = "bash sh grep mktemp mkdir cp chmod find dirname cat sed sort tar shasum awk head rm mv"
    for name in tools.split():
        if (found := shutil.which(name)) is not None:
            (bare / name).symlink_to(found)
    for stub in (tmp_path / "stubs").iterdir():
        if stub.name not in {"zstd", "python3"}:
            (bare / stub.name).symlink_to(stub)
    done = _run_pack(top, {**env, "PATH": str(bare)}, tmp_path)
    assert done.returncode == 1
    assert "zstd is not on the PATH" in done.stderr


@runs_on_macos
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


def _mlx_paragraph() -> str:
    text = (ROOT / "docs" / "usage.md").read_text(encoding="utf-8")
    (paragraph,) = (p for p in text.split("\n\n") if p.startswith("mlx (mlx-lm)"))
    return " ".join(paragraph.split())


def test_the_docs_paragraph_of_mlx_states_the_floor_the_roles_and_that_no_pack_exists() -> None:
    paragraph = _mlx_paragraph()
    assert f"macOS {mlx.MACOS_FLOOR[0]} or newer" in paragraph
    assert "needs Apple Silicon" in paragraph
    assert mlx.MLX.served_roles == {WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK}
    assert "serves chat, embed and rerank" in paragraph
    assert "Qwen3-Embedding and Qwen3-Reranker" in paragraph
    packs = (ROOT / "src" / "lilbee" / "providers" / "fleet" / "engine_packs.json").read_text(
        encoding="utf-8"
    )
    assert "mlx" not in packs
    assert "No pack of mlx is published yet" in paragraph


def test_the_launcher_is_executable_and_names_the_adapter_tool() -> None:
    assert os.access(LAUNCHER, os.X_OK)
    assert LAUNCHER.name == mlx.MLX.server_process.name


def test_the_embed_launcher_is_executable_and_is_the_tool_of_the_embed_argv() -> None:
    assert os.access(EMBED_LAUNCHER, os.X_OK)
    assert EMBED_LAUNCHER.name == mlx._EMBED_TOOL
    assert SERVER.name in mlx.MLX.server_process.scripts


@runs_on_macos
def test_the_embed_launcher_runs_the_server_script_on_the_bundled_interpreter(
    tmp_path: Path,
) -> None:
    package = tmp_path / "engines" / "lilbee_engine_mlx"
    (package / "bin").mkdir(parents=True)
    launcher = package / "bin" / "mlx-embed-server"
    shutil.copy(EMBED_LAUNCHER, launcher)
    _stub(package / "python" / "bin", "python3", 'echo "$0" "$@"')
    done = subprocess.run(
        [str(launcher), "--task", "embed", "--port", "5"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=True,
        cwd=tmp_path,
    )
    interpreter, *args = done.stdout.split()
    assert Path(interpreter).resolve() == (package / "python" / "bin" / "python3").resolve()
    assert args[0] == "-I"
    assert Path(args[1]).resolve() == (package / SERVER.name).resolve()
    assert args[2:] == ["--task", "embed", "--port", "5"]
