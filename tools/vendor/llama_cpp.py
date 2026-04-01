#!/usr/bin/env python3
"""Download a prebuilt llama-cpp-python wheel and vendor it into a lilbee wheel.

Usage:
    python -m tools.vendor.llama_cpp <lilbee-wheel> \\
        [--version VERSION] [--platform PLATFORM] [--wheel-platform WHEEL_PLAT]

This script:
1. Downloads the correct prebuilt llama-cpp-python wheel
2. Extracts llama_cpp/ (Python code + compiled shared libraries)
3. Injects it into the lilbee wheel as a top-level llama_cpp/ package
4. Removes the llama-cpp-python Requires-Dist from METADATA
5. Updates the WHEEL file with platform-specific tags
6. Renames the output wheel to match the new tags

The --platform flag specifies which prebuilt llama-cpp-python wheel to
download (e.g., manylinux2014_x86_64). The --wheel-platform flag specifies
the platform tag for the output lilbee wheel (e.g.,
manylinux_2_17_x86_64.manylinux2014_x86_64). Both auto-detect locally.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PREBUILT_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu/"
DEFAULT_VERSION = "0.3.18"


def _detect_tags(system: str, machine: str) -> tuple[str, str]:
    """Return (download_tag, wheel_tag) for a given OS and architecture.

    download_tag: platform tag used to fetch the prebuilt llama-cpp-python wheel.
    wheel_tag:    platform tag stamped on the output lilbee wheel.

    On Linux these differ (manylinux2014 vs manylinux_2_17 compat tag);
    on macOS and Windows they are identical.
    """
    machine = machine.lower()

    if system == "Linux":
        download = f"manylinux2014_{machine}"
        wheel = f"manylinux_2_17_{machine}.manylinux2014_{machine}"
        return download, wheel
    elif system == "Darwin":
        tag = "macosx_11_0_arm64" if machine == "arm64" else "macosx_10_15_x86_64"
        return tag, tag
    elif system == "Windows":
        tag = "win_amd64" if machine in ("amd64", "x86_64") else "win32"
        return tag, tag
    else:
        raise RuntimeError(f"Unsupported platform: {system} {machine}")


def detect_platform_tag() -> str:
    """Return the llama-cpp-python wheel platform tag for the current system."""
    return _detect_tags(platform.system(), platform.machine())[0]


def detect_wheel_platform() -> str:
    """Return the output wheel platform tag."""
    return _detect_tags(platform.system(), platform.machine())[1]


def python_tag() -> str:
    """Return the cpython tag, e.g. 'cp312'."""
    v = sys.version_info
    return f"cp{v.major}{v.minor}"


def download_wheel(version: str, plat: str, dest_dir: Path) -> Path:
    """Download the prebuilt llama-cpp-python wheel via pip download."""
    py = python_tag()
    print(f"Downloading llama-cpp-python {version} for {py}-{plat}...")

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            f"--platform={plat}",
            f"--python-version={py.removeprefix('cp')}",
            f"--index-url={PREBUILT_INDEX}",
            f"llama-cpp-python=={version}",
            f"--dest={dest_dir}",
        ],
    )

    wheels = list(dest_dir.glob("llama_cpp_python-*.whl"))
    if not wheels:
        raise RuntimeError("No wheel downloaded")
    return wheels[0]


def _sha256_urlsafe_b64(data: bytes) -> str:
    """Compute sha256 hash in the URL-safe base64 format used by RECORD."""
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def repack_wheel(
    lilbee_wheel: Path,
    llama_wheel: Path,
    wheel_platform: str,
) -> Path:
    """Inject llama_cpp into lilbee_wheel and retag as platform-specific."""
    py = python_tag()
    work_dir = lilbee_wheel.parent / "_repack_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    # Extract both wheels
    lilbee_dir = work_dir / "lilbee"
    llama_dir = work_dir / "llama"
    with zipfile.ZipFile(lilbee_wheel) as zf:
        zf.extractall(lilbee_dir)
    with zipfile.ZipFile(llama_wheel) as zf:
        zf.extractall(llama_dir)

    # Copy llama_cpp/ to wheel root so it installs as a top-level package
    src_llama = llama_dir / "llama_cpp"
    dst_vendor = lilbee_dir / "llama_cpp"
    if dst_vendor.exists():
        shutil.rmtree(dst_vendor)
    shutil.copytree(src_llama, dst_vendor)

    # Remove __pycache__ dirs
    for pycache in dst_vendor.rglob("__pycache__"):
        shutil.rmtree(pycache)

    print(f"Vendored llama_cpp ({_dir_size_mb(dst_vendor):.1f} MB)")

    dist_info = _find_dist_info(lilbee_dir)

    # Patch METADATA: remove llama-cpp-python Requires-Dist
    metadata_path = dist_info / "METADATA"
    metadata_lines = metadata_path.read_text().splitlines()
    metadata_lines = [
        line
        for line in metadata_lines
        if not (line.startswith("Requires-Dist:") and "llama-cpp-python" in line.lower())
    ]
    metadata_path.write_text("\n".join(metadata_lines) + "\n")

    # Patch WHEEL: update Tag from py3-none-any to platform-specific
    wheel_path = dist_info / "WHEEL"
    wheel_text = wheel_path.read_text()
    wheel_text = re.sub(r"Tag: py3-none-any", f"Tag: {py}-{py}-{wheel_platform}", wheel_text)
    wheel_path.write_text(wheel_text)

    # Regenerate RECORD with correct hashes
    record_path = dist_info / "RECORD"
    records = []
    for fpath in sorted(lilbee_dir.rglob("*")):
        if fpath.is_dir():
            continue
        rel = fpath.relative_to(lilbee_dir).as_posix()
        if rel == record_path.relative_to(lilbee_dir).as_posix():
            records.append(f"{rel},,")
            continue
        data = fpath.read_bytes()
        h = _sha256_urlsafe_b64(data)
        size = len(data)
        records.append(f"{rel},sha256={h},{size}")
    record_path.write_text("\n".join(records) + "\n")

    # Determine output filename
    # Original: lilbee-0.6.0-py3-none-any.whl
    # New:      lilbee-0.6.0-cp312-cp312-macosx_11_0_arm64.whl
    # PEP 427: wheel filenames are {name}-{ver}-{python}-{abi}-{platform}.whl
    # rsplit("-", 3) splits from the right into exactly 4 parts, so [0] is
    # always "{name}-{ver}" regardless of hyphens in the project name/version.
    name_version = lilbee_wheel.stem.rsplit("-", 3)[0]
    output_name = f"{name_version}-{py}-{py}-{wheel_platform}.whl"
    output = lilbee_wheel.parent / output_name

    # Remove old wheel
    lilbee_wheel.unlink()

    # Write new wheel
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(lilbee_dir.rglob("*")):
            if fpath.is_dir():
                continue
            arcname = fpath.relative_to(lilbee_dir).as_posix()
            zf.write(fpath, arcname)

    shutil.rmtree(work_dir)
    print(f"Wrote {output.name} ({output.stat().st_size / 1024 / 1024:.1f} MB)")
    return output


def _find_dist_info(wheel_dir: Path) -> Path:
    """Find the .dist-info directory inside an extracted wheel."""
    candidates = list(wheel_dir.glob("*.dist-info"))
    if not candidates:
        raise RuntimeError(f"No .dist-info found in {wheel_dir}")
    return candidates[0]


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory in MB."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor llama-cpp-python into a lilbee wheel")
    parser.add_argument("wheel", type=Path, help="Path to the lilbee .whl file to modify")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="llama-cpp-python version")
    parser.add_argument(
        "--platform", default=None, help="llama-cpp-python wheel platform (auto-detected)"
    )
    parser.add_argument(
        "--wheel-platform", default=None, help="Output wheel platform tag (auto-detected)"
    )
    args = parser.parse_args()

    if not args.wheel.exists():
        parser.error(f"Wheel not found: {args.wheel}")

    plat = args.platform or detect_platform_tag()
    wheel_plat = args.wheel_platform or detect_wheel_platform()

    with tempfile.TemporaryDirectory() as tmp:
        llama_whl = download_wheel(args.version, plat, Path(tmp))
        repack_wheel(args.wheel, llama_whl, wheel_plat)


if __name__ == "__main__":
    main()
