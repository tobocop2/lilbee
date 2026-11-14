"""The CCCL source reader names the tag of the release job that builds one Windows CUDA asset."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/cccl_source.py"
_ASSET = "llama-b12032-bin-win-cuda-12.4-x64.zip"

# The shape of upstream's release.yml at b12032: two CUDA versions, the second also for arm64.
_RELEASE = """
jobs:
  windows-cuda:
    strategy:
      matrix:
        include:
          - cuda: '12.4'
            arch: x64
            defines: '-DGGML_CUDA_CCCL_VERSION=v3.4.3'
          - cuda: '13.4'
            arch: x64
            defines: '-DGGML_CUDA_CCCL_VERSION=v3.5.0'
          - cuda: '13.4'
            arch: arm64
            defines: '-DGGML_CUDA_CCCL_VERSION=v3.5.0 -DCMAKE_TOOLCHAIN_FILE=cmake/arm64.cmake'
"""
_NO_DEFINE = _RELEASE.replace("-DGGML_CUDA_CCCL_VERSION=v3.4.3", "")
_CMAKE = """
    if (GGML_CUDA_CCCL_VERSION)
        FetchContent_Declare(
            CCCL
            GIT_REPOSITORY https://github.com/nvidia/cccl.git
            GIT_TAG        "${GGML_CUDA_CCCL_VERSION}"
        )
"""


def _tree(root: pathlib.Path, release: str = _RELEASE, cmake: str = _CMAKE) -> pathlib.Path:
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/release.yml").write_text(release, encoding="utf-8")
    (root / "ggml/src/ggml-cuda").mkdir(parents=True)
    (root / "ggml/src/ggml-cuda/CMakeLists.txt").write_text(cmake, encoding="utf-8")
    return root


def _run(tree: pathlib.Path, asset: str = _ASSET) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), str(tree), asset, "b12032"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_prints_the_repository_and_the_tag_of_the_matching_job(tmp_path: pathlib.Path) -> None:
    result = _run(_tree(tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["https://github.com/nvidia/cccl.git", "v3.4.3"]


def test_reads_the_tag_of_the_entry_that_builds_the_asset(tmp_path: pathlib.Path) -> None:
    result = _run(_tree(tmp_path), "llama-b12032-bin-win-cuda-13.4-arm64.zip")
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[1] == "v3.5.0"


@pytest.mark.parametrize(
    ("release", "cmake", "asset", "message"),
    [
        (_RELEASE, _CMAKE, "llama-b12032-bin-win-cuda-12.9-x64.zip", "has 0 windows-cuda entries"),
        (_NO_DEFINE, _CMAKE, _ASSET, "passes no GGML_CUDA_CCCL_VERSION"),
        (_RELEASE, _CMAKE.replace("CCCL\n", "OTHER\n"), _ASSET, "fetches no CCCL repository"),
    ],
    ids=["no-entry-for-the-asset", "no-cccl-define", "no-cccl-fetch"],
)
def test_fails_and_names_what_is_missing(
    tmp_path: pathlib.Path, release: str, cmake: str, asset: str, message: str
) -> None:
    result = _run(_tree(tmp_path, release, cmake), asset)
    assert result.returncode == 1
    assert message in result.stderr
