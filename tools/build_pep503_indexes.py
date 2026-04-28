#!/usr/bin/env python3
"""Emit PEP 503 simple-repository indexes for lilbee per-backend wheels.

Reads wheel artifacts produced by .github/workflows/build-wheels.yml,
groups by backend tag, copies wheels into <site>/<backend>/lilbee/, and
writes the per-directory index.html files pip's --extra-index-url
expects.

Input layout (artifact-dir mode, default):

    <input>/wheel-default-<os>-<backend>-py<ver>/lilbee-*.whl
    <input>/wheel-extra-<os>-<backend>-py<ver>/lilbee-*.whl

The os segment may itself contain dashes (e.g. ubuntu-22.04, ubuntu-latest,
windows-2022). Backend is parsed by stripping the wheel-{default,extra}-
prefix and the trailing -py<ver> suffix, then taking the last remaining
dash-segment.

Output layout:

    <site>/<backend>/lilbee/<wheel>.whl
    <site>/<backend>/lilbee/index.html      # links to each wheel
    <site>/<backend>/index.html             # links to lilbee/

Usage:
    python tools/build_pep503_indexes.py <artifacts-dir> <site-dir>
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ARTIFACT_DIR_RE = re.compile(r"^wheel-(?:default|extra)-(?P<rest>.+)-py\d+\.\d+$")


def backend_from_artifact_dir(name: str) -> str | None:
    """Extract backend tag from a wheel artifact directory name.

    Returns None if the directory name doesn't match the expected pattern.
    """
    m = ARTIFACT_DIR_RE.match(name)
    if not m:
        return None
    rest = m.group("rest")
    return rest.rsplit("-", 1)[-1]


def collect_wheels(input_dir: Path) -> dict[str, list[Path]]:
    """Group wheel files by backend tag."""
    by_backend: dict[str, list[Path]] = {}
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        backend = backend_from_artifact_dir(child.name)
        if backend is None:
            continue
        wheels = sorted(child.glob("lilbee-*.whl"))
        if not wheels:
            continue
        by_backend.setdefault(backend, []).extend(wheels)
    return by_backend


def write_backend_indexes(site: Path, by_backend: dict[str, list[Path]]) -> None:
    """Copy wheels and write PEP 503 index pages under site/<backend>/."""
    for backend, wheels in by_backend.items():
        pkg_dir = site / backend / "lilbee"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        for whl in wheels:
            shutil.copy2(whl, pkg_dir / whl.name)

        wheel_lines = [f'<a href="{whl.name}">{whl.name}</a><br>' for whl in wheels]
        (pkg_dir / "index.html").write_text(
            "<!DOCTYPE html><html><body>\n" + "\n".join(wheel_lines) + "\n</body></html>\n"
        )

        (site / backend / "index.html").write_text(
            '<!DOCTYPE html><html><body>\n<a href="lilbee/">lilbee</a><br>\n</body></html>\n'
        )

        print(f"backend={backend} wheels={len(wheels)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input", type=Path, help="Directory containing wheel-*-<backend>-py* artifact subdirs"
    )
    parser.add_argument(
        "site", type=Path, help="Output site root; backend subdirs are written under here"
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
    write_backend_indexes(args.site, by_backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
