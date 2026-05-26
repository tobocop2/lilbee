#!/usr/bin/env python3
"""Emit PEP 503 simple-repository indexes for the per-backend engine wheels.

Reads the ``lilbee-llama-server`` wheels produced by build-multigpu.yml, groups
them by backend, and writes per-directory ``index.html`` files. Each wheel link
is an absolute URL into the GitHub release assets, so the static site need not
host multi-gigabyte wheels (GitHub Pages caps a deployment at 1 GB; releases do
not). A user picks a backend via ``pip install lilbee --extra-index-url
https://lilbee.sh/<backend>/`` -- the pure lilbee wheel still comes from PyPI;
this index supplies the matching engine for that backend.

Backend disambiguation. The cu121/cu124/cu125/rocm/cpu engine wheels share the
same PEP 427 filename (project + version + python + abi + platform) as the
default vulkan/metal wheels -- the backend lives inside the wheel, not its name.
GitHub releases use a flat filename namespace, so each non-default wheel gets a
PEP 427 build tag inserted (``1.cu125``, ``1.cpu``, ...). Defaults stay unchanged
because they also ship to PyPI, where a build tag would break the pin.

Input layout (artifact-dir mode):

    <input>/wheel-multigpu-<os>-<backend>/lilbee_llama_server-*.whl

Backend is the last dash-segment of the artifact dir name; the wheel's version
is parsed from its filename so the release URL needs no extra CLI flag.

Output layout:

    <site>/<backend>/lilbee-llama-server/index.html   # hrefs to <release>/<whl>
    <site>/<backend>/index.html                       # link to lilbee-llama-server/

Usage:
    python tools/build_pep503_indexes.py <artifacts-dir> <site-dir> \\
        [--release-base-url https://github.com/owner/repo/releases/download]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The per-backend index serves the engine wheel, lilbee-llama-server. Its
# artifacts are wheel-multigpu-<os>-<backend> (no per-CPython axis: the binary
# is Python-version-independent). The pure lilbee wheel is backend-agnostic and
# ships only to PyPI, so it has no per-backend index.
ARTIFACT_DIR_RE = re.compile(r"^wheel-multigpu-(?P<rest>.+)$")
WHEEL_FILENAME_RE = re.compile(r"^lilbee_llama_server-(?P<version>[^-]+)-")
_PROJECT = "lilbee-llama-server"
_DEFAULT_RELEASE_BASE_URL = "https://github.com/tobocop2/lilbee/releases/download"

# Backends whose wheels ship under the default filename (also on PyPI; renaming
# would break the PyPI pin). Everything else gets a PEP 427 build tag inserted
# so the GH release can hold every variant without filename collisions.
_DEFAULT_BACKENDS: frozenset[str] = frozenset({"vulkan", "metal"})

# Wheel filename layout (PEP 427): project-version[-buildtag]-python-abi-platform.whl
_WHEEL_FILENAME_PARTS_NO_BUILDTAG = 5


def backend_from_artifact_dir(name: str) -> str | None:
    """Extract backend tag from a wheel artifact directory name.

    Returns None if the directory name doesn't match the expected pattern.
    """
    m = ARTIFACT_DIR_RE.match(name)
    if not m:
        return None
    rest = m.group("rest")
    return rest.rsplit("-", 1)[-1]


def version_from_wheel_filename(name: str) -> str | None:
    """Extract the version from ``lilbee_llama_server-<version>-<rest>.whl``."""
    m = WHEEL_FILENAME_RE.match(name)
    return m.group("version") if m else None


def build_tag_for_backend(backend: str) -> str | None:
    """PEP 427 build tag used to disambiguate non-default backend variants."""
    if backend in _DEFAULT_BACKENDS:
        return None
    return f"1.{backend}"


def rename_for_release(wheel_name: str, backend: str) -> str:
    """Return the wheel filename as published on the GH release.

    Default backends (vulkan/metal) keep their original name. Extra backends
    get a build tag inserted after the version per PEP 427.
    """
    build_tag = build_tag_for_backend(backend)
    if build_tag is None:
        return wheel_name
    parts = wheel_name.removesuffix(".whl").split("-")
    if len(parts) != _WHEEL_FILENAME_PARTS_NO_BUILDTAG:
        raise ValueError(
            f"wheel filename {wheel_name!r} does not match the no-buildtag layout "
            f"(project-version-python-abi-platform.whl)"
        )
    project, version, python, abi, platform = parts
    return f"{project}-{version}-{build_tag}-{python}-{abi}-{platform}.whl"


def collect_wheels(input_dir: Path) -> dict[str, list[Path]]:
    """Group wheel files by backend tag."""
    by_backend: dict[str, list[Path]] = {}
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        backend = backend_from_artifact_dir(child.name)
        if backend is None:
            continue
        wheels = sorted(child.glob("lilbee_llama_server-*.whl"))
        if not wheels:
            continue
        by_backend.setdefault(backend, []).extend(wheels)
    return by_backend


def _wheel_href(release_base_url: str, version: str, asset_name: str) -> str:
    return f"{release_base_url.rstrip('/')}/v{version}/{asset_name}"


def write_backend_indexes(
    site: Path,
    by_backend: dict[str, list[Path]],
    release_base_url: str,
) -> None:
    """Write PEP 503 index pages under site/<backend>/, linking to release assets."""
    for backend, wheels in by_backend.items():
        pkg_dir = site / backend / _PROJECT
        pkg_dir.mkdir(parents=True, exist_ok=True)

        wheel_lines: list[str] = []
        for whl in wheels:
            version = version_from_wheel_filename(whl.name)
            if version is None:
                print(f"skipping unparseable wheel filename: {whl.name}", file=sys.stderr)
                continue
            asset_name = rename_for_release(whl.name, backend)
            href = _wheel_href(release_base_url, version, asset_name)
            wheel_lines.append(f'<a href="{href}">{asset_name}</a><br>')

        (pkg_dir / "index.html").write_text(
            "<!DOCTYPE html><html><body>\n" + "\n".join(wheel_lines) + "\n</body></html>\n"
        )

        (site / backend / "index.html").write_text(
            f'<!DOCTYPE html><html><body>\n<a href="{_PROJECT}/">{_PROJECT}</a><br>\n'
            "</body></html>\n"
        )

        print(f"backend={backend} wheels={len(wheels)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", type=Path, help="Directory containing wheel-multigpu-<os>-<backend> subdirs"
    )
    parser.add_argument(
        "site", type=Path, help="Output site root; backend subdirs are written under here"
    )
    parser.add_argument(
        "--release-base-url",
        default=_DEFAULT_RELEASE_BASE_URL,
        help="GitHub release-asset base URL (default: lilbee's release-download root).",
    )
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        print(f"input directory does not exist: {args.input}", file=sys.stderr)
        return 1

    args.site.mkdir(parents=True, exist_ok=True)
    by_backend = collect_wheels(args.input)
    if not by_backend:
        print("no wheel artifacts found; nothing to index", file=sys.stderr)
        return 0
    write_backend_indexes(args.site, by_backend, args.release_base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
