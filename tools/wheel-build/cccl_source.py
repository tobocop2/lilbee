"""Print the CCCL repository and tag that upstream's release job compiled into one Windows CUDA asset.

Usage: cccl_source.py <llama.cpp checkout> <asset name> <upstream tag>
The tag comes from the GGML_CUDA_CCCL_VERSION define of the matching windows-cuda entry in the
checkout's release.yml, the repository from the CCCL FetchContent call in ggml-cuda's CMake file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


def cccl_source(tree: Path, asset: str, tag: str) -> tuple[str, str]:
    """The CCCL repository and tag of the release job that builds `asset`."""
    release = yaml.safe_load((tree / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    entries = [
        e
        for e in release["jobs"]["windows-cuda"]["strategy"]["matrix"]["include"]
        if asset == f"llama-{tag}-bin-win-cuda-{e['cuda']}-{e['arch']}.zip"
    ]
    if len(entries) != 1:
        raise SystemExit(f"release.yml has {len(entries)} windows-cuda entries that build {asset}")
    version = re.search(r"-DGGML_CUDA_CCCL_VERSION=(\S+)", entries[0].get("defines", ""))
    if not version:
        raise SystemExit(f"the release job of {asset} passes no GGML_CUDA_CCCL_VERSION")
    cmake = (tree / "ggml/src/ggml-cuda/CMakeLists.txt").read_text(encoding="utf-8")
    repo = re.search(r"FetchContent_Declare\(\s*CCCL\s+GIT_REPOSITORY\s+(\S+)", cmake)
    if not repo:
        raise SystemExit("ggml-cuda's CMake file fetches no CCCL repository")
    return repo.group(1), version.group(1)


if __name__ == "__main__":
    print(*cccl_source(Path(sys.argv[1]), sys.argv[2], sys.argv[3]))
