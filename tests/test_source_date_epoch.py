"""The llama.cpp and tei engine wheels take their entry times from their source commit."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._pack_workflow import pack_workflow
from tests._workflow_steps import commands, steps

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "wheel-build" / "source_date_epoch.sh"
COMMIT_TIME = 1_700_000_000
BASH = shutil.which("bash")
UV = shutil.which("uv")

pytestmark = pytest.mark.skipif(
    BASH is None or UV is None, reason="bash and uv are needed to build a wheel"
)

_CARGO = """#!/bin/sh
# Stand-in for cargo-about: write the report that the caller asks for.
json=no
while [ "$#" -gt 0 ]; do
  case "$1" in
    --format) json=yes ;;
    -o) out="$2"; shift ;;
  esac
  shift
done
if [ "$json" = yes ]; then
  echo '{"licenses": [{"id": "MIT", "used_by": []}]}' > "$out"
else
  echo 'licence text' > "$out"
fi
"""

_PYTHON = """#!/bin/sh
# Stand-in for python: the wheel build runs through uv, and the retag is skipped.
if [ "$1" = "-m" ] && [ "$2" = "build" ]; then
  while [ "$#" -gt 0 ]; do
    [ "$1" = "--outdir" ] && out="$2"
    shift
  done
  exec "$UV" build --wheel --out-dir "$out"
fi
if [ "$1" = "-m" ] && [ "$2" = "wheel" ]; then
  exit 0
fi
exec "$REAL_PYTHON" "$@"
"""


def _git(repo: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": f"{COMMIT_TIME} +0000",
        "GIT_COMMITTER_DATE": f"{COMMIT_TIME} +0000",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


def _source(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "LICENSE").write_text("licence\n", encoding="utf-8")
    (repo / "Cargo.lock").write_text("version = 4\n", encoding="utf-8")
    _git(repo, "add", "LICENSE", "Cargo.lock")
    _git(repo, "commit", "-q", "-m", "source")
    return repo


def _epoch(source: Path) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    return subprocess.run(
        [BASH, str(SCRIPT), str(source)], capture_output=True, text=True, encoding="utf-8"
    )


def _set_times(tree: Path, when: int) -> None:
    for path in (tree, *tree.rglob("*")):
        os.utime(path, (when, when))


def _sha256(wheels: Path) -> str:
    (wheel,) = wheels.glob("*.whl")
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_the_script_prints_the_commit_time_of_the_source(tmp_path: Path) -> None:
    done = _epoch(_source(tmp_path))
    assert (done.returncode, done.stdout) == (0, f"SOURCE_DATE_EPOCH={COMMIT_TIME}\n")


def test_the_script_refuses_a_folder_that_is_not_a_git_tree(tmp_path: Path) -> None:
    done = _epoch(tmp_path)
    assert done.returncode != 0
    assert done.stdout == ""


def test_building_the_llama_cpp_project_twice_gives_one_wheel(tmp_path: Path) -> None:
    """The build step of the cell is inline in the workflow, so this builds the project itself."""
    assert UV is not None
    env = dict([_epoch(_source(tmp_path)).stdout.strip().split("=", 1)])
    digests = []
    for run, checkout_time in (("first", 1_750_000_000), ("second", 1_760_000_000)):
        tree = tmp_path / run / "project"
        shutil.copytree(
            ROOT / "packaging" / "engine-wheel",
            tree,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
        _set_times(tree, checkout_time)
        subprocess.run(
            [UV, "build", "--wheel", "--out-dir", str(tmp_path / run / "dist")],
            cwd=tree,
            env={**os.environ, **env},
            check=True,
            capture_output=True,
        )
        digests.append(_sha256(tmp_path / run / "dist"))
    assert digests[0] == digests[1]


def _wrap_tei(tmp_path: Path, run: str, checkout_time: int) -> str:
    """Run tei_wheel.sh on a copy of the tools and the project, and hash the wheel."""
    assert BASH is not None and UV is not None
    tree = tmp_path / run / "tree"
    shutil.copytree(ROOT / "tools" / "wheel-build", tree / "tools" / "wheel-build")
    shutil.copytree(
        ROOT / "packaging" / "engine-wheel-tei",
        tree / "packaging" / "engine-wheel-tei",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    _set_times(tree, checkout_time)
    shims = tmp_path / run / "shims"
    shims.mkdir()
    for name, body in (("cargo", _CARGO), ("python", _PYTHON)):
        (shims / name).write_text(body, encoding="utf-8")
        (shims / name).chmod(0o755)
    router = tmp_path / "router"
    router.write_bytes(b"router")
    (tmp_path / "cargo-home" / "registry" / "src").mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PATH": f"{shims}{os.pathsep}{os.environ['PATH']}",
        "CARGO_HOME": str(tmp_path / "cargo-home"),
        "PYTHON": str(shims / "python"),
        "REAL_PYTHON": sys.executable,
        "UV": UV,
        "WHEEL_VERSION": "1.0.0",
    }
    subprocess.run(
        [
            BASH,
            str(tree / "tools" / "wheel-build" / "tei_wheel.sh"),
            str(tmp_path / "source"),
            str(router),
            "candle,http",
            "x86_64-unknown-linux-gnu",
            "cpu",
            "linux_x86_64",
            str(tmp_path / run / "out"),
        ],
        env=env,
        check=True,
        capture_output=True,
    )
    return _sha256(tmp_path / run / "out" / "dist")


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is not installed")
@pytest.mark.skipif(sys.platform == "win32", reason="tei_wheel.sh runs only on Linux and macOS")
def test_wrapping_one_tei_source_twice_gives_one_wheel(tmp_path: Path) -> None:
    _source(tmp_path)
    first = _wrap_tei(tmp_path, "first", 1_750_000_000)
    second = _wrap_tei(tmp_path, "second", 1_760_000_000)
    assert first == second


def test_the_llama_cpp_cell_hands_the_epoch_to_the_wheel_build() -> None:
    runs = [commands(step.get("run", "")) for step in steps(pack_workflow(), "cell")]
    (clone,) = (i for i, run in enumerate(runs) if "source_date_epoch.sh" in run)
    (build,) = (i for i, run in enumerate(runs) if "python -m build --wheel" in run)
    assert clone < build
    piped = "bash tools/wheel-build/source_date_epoch.sh ${work}/llama.cpp >> $GITHUB_ENV"
    assert piped in runs[clone]


def test_the_multigpu_wheel_build_reads_the_epoch_before_it_builds() -> None:
    (run,) = (
        commands(step["run"])
        for step in steps("build-multigpu.yml", "build-multigpu")
        if step.get("name") == "Build the lilbee-engine-llama-cpp wheel"
    )
    epoch = "export $(bash ${GITHUB_WORKSPACE}/tools/wheel-build/source_date_epoch.sh"
    assert run.index(epoch) < run.index("python -m build --wheel")
