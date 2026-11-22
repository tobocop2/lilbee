"""Fail when a Metal library in a wheel targets a newer macOS than the platform tag names.

Usage: metal_floor.py <wheel> <platform tag>

The layout of the library header is the one that Metal.jl reads and writes (read_metallib in
src/metallib/model.jl of https://github.com/JuliaGPU/Metal.jl).
"""

from __future__ import annotations

import re
import struct
import sys
import zipfile
from pathlib import Path

# Two letters after these four bytes are the name of a Metal class, MTLBuffer for example. The two
# bytes of a container version are 0x01 and 0x80 or 0x00.
LIBRARY = re.compile(rb"MTLB(?![A-Za-z]{2})")
# The container version (its high bit says macOS), the operating system, and the major, minor and
# patch numbers of the target.
HEADER = struct.Struct("<4xH5xBHBB")
MACOS = (0x8001, 0x81)

Version = tuple[int, int, int]


def text(version: Version) -> str:
    return ".".join(str(number) for number in version)


def target(data: bytes, start: int, place: str) -> Version:
    """The macOS version in the header of the Metal library at this byte."""
    if len(data) - start < HEADER.size:
        raise SystemExit(f"metal_floor: {place} has no complete header")
    container, system, major, minor, patch = HEADER.unpack_from(data, start)
    if (container, system) != MACOS:
        raise SystemExit(
            f"metal_floor: {place} names no macOS version: its container version is "
            f"{container:#06x} and its operating system is {system:#04x}"
        )
    return major, minor, patch


def targets(wheel: Path) -> list[Version]:
    """The macOS version of each Metal library in a wheel, also of one inside another file."""
    found = []
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            for match in LIBRARY.finditer(data):
                place = f"the Metal library at byte {match.start()} of {name}"
                found.append(target(data, match.start(), place))
    return found


def check(wheel: Path, platform: str) -> None:
    tag = re.fullmatch(r"macosx_([0-9]+)_([0-9]+)_arm64", platform)
    if tag is None:
        raise SystemExit(f"metal_floor: the platform {platform} names no macOS version for arm64")
    found = targets(wheel)
    if not found:
        raise SystemExit(
            f"metal_floor: {wheel.name} holds no Metal library, so its macOS floor is unknown"
        )
    named, needed = (int(tag[1]), int(tag[2]), 0), max(found)
    if needed > named:
        raise SystemExit(
            f"metal_floor: the Metal libraries of {wheel.name} need macOS {text(needed)}, "
            f"but its tag names macOS {text(named)}"
        )


if __name__ == "__main__":
    check(Path(sys.argv[1]), sys.argv[2])
