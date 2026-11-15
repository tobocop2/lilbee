#!/usr/bin/env python3
"""Repack an upstream llama.cpp OpenVINO archive into the tree of the engine wheel.

Usage: repack_openvino.py ARCHIVE WHEEL_DIR

The archive is a .tar.gz or .zip. The server and every shared library go to
``WHEEL_DIR/lilbee_engine_openvino/bin``. A zip cannot hold a symbolic link and the
engine store unpacks it as a plain file, so each library that a link names by its
SONAME (``libfoo.so.1``) is written once under that name. The licences go to
``WHEEL_DIR``: ``LICENSE-llama.cpp``, ``LICENSE-OpenVINO``, ``THIRD-PARTY/OpenVINO/``
and, when the archive holds libomp, ``LICENSE-LLVM-OpenMP``.
"""

from __future__ import annotations

import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

PACKAGE = "lilbee_engine_openvino"
LICENSING_DIR = "openvino-licensing"
_SERVERS = ("llama-server", "llama-server.exe")
_LIBRARY = re.compile(r"lib.+\.so(\.[\d.]+)?|.+\.dll")
_DEBUG_DLL = re.compile(r".+_debug\.dll")
_SONAME_LINK = re.compile(r".+\.so\.\d+")
_DATA_FILES = ("cache.json",)
_ARGUMENTS = 2


class RepackError(Exception):
    """The archive does not hold what the wheel needs."""


def _extract(archive: Path, scratch: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(scratch)  # noqa: S202 - the workflow checks the archive digest first
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(scratch, filter="data")


def _server_root(scratch: Path) -> Path:
    roots = {p.parent for p in scratch.rglob("*") if p.is_file() and p.name in _SERVERS}
    if len(roots) != 1:
        raise RepackError(f"expected one folder with llama-server, found {len(roots)}")
    return roots.pop()


def _install_names(root: Path) -> dict[Path, list[str]]:
    """Each real file to ship, with the names it installs under."""
    links: dict[Path, list[str]] = {}
    for entry in root.iterdir():
        if entry.is_symlink() and _SONAME_LINK.fullmatch(entry.name):
            links.setdefault(entry.resolve(), []).append(entry.name)
    names: dict[Path, list[str]] = {}
    for entry in sorted(root.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            continue
        wanted = entry.name in _SERVERS or entry.name in _DATA_FILES
        wanted = wanted or (_LIBRARY.fullmatch(entry.name) and not _DEBUG_DLL.fullmatch(entry.name))
        if wanted:
            names[entry] = links.get(entry.resolve()) or [entry.name]
    return names


def _copy_licences(root: Path, wheel_dir: Path) -> None:
    licensing = root / LICENSING_DIR
    for source, target in (
        (root / "LICENSE", wheel_dir / "LICENSE-llama.cpp"),
        (licensing / "LICENSE", wheel_dir / "LICENSE-OpenVINO"),
    ):
        if not source.is_file() or source.stat().st_size == 0:
            raise RepackError(f"the archive has no {source.relative_to(root)}")
        shutil.copy2(source, target)
    third = wheel_dir / "THIRD-PARTY" / "OpenVINO"
    shutil.copytree(licensing, third, dirs_exist_ok=True)
    omp = root / "LICENSE-LLVM-OpenMP"
    if omp.is_file():
        shutil.copy2(omp, wheel_dir / omp.name)
    elif (root / "libomp.dll").is_file():
        raise RepackError("libomp.dll without LICENSE-LLVM-OpenMP")


def repack(archive: Path, wheel_dir: Path) -> None:
    """Fill the wheel tree under *wheel_dir* from *archive*."""
    bin_dir = wheel_dir / PACKAGE / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        _extract(archive, Path(scratch))
        root = _server_root(Path(scratch))
        for source, installed in _install_names(root).items():
            for name in installed:
                shutil.copy2(source, bin_dir / name)
        _copy_licences(root, wheel_dir)


def main(argv: list[str]) -> int:
    if len(argv) != _ARGUMENTS:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        repack(Path(argv[0]), Path(argv[1]))
    except RepackError as error:
        print(f"{argv[0]}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
