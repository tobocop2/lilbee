#!/usr/bin/env python3
"""Copy wheel artifacts into a flat staging dir under their release-asset names.

Each ``wheel-{default,extra}-<os>-<backend>-py<ver>/`` artifact subdir holds
one wheel. The default-backend wheels (vulkan / metal) keep their original
name -- those also ship to PyPI and a build tag would break the PyPI pin.
Every other backend (cu121, cu124, cu125, cpu) gets a PEP 427 build tag
inserted into the wheel filename so multiple variants of the same
(project, version, python, abi, platform) tuple can coexist on a single
GitHub release. ``build_pep503_indexes.py`` uses the same rename so the
index hrefs match the release-asset names.

Usage:
    python tools/stage_release_assets.py <input-dir> <output-dir>
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tools.build_pep503_indexes import backend_from_artifact_dir, rename_for_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Directory containing wheel-*-<backend>-py* artifact subdirs"
    )
    parser.add_argument(
        "output", type=Path, help="Flat output directory to receive the renamed wheels"
    )
    args = parser.parse_args(argv)

    if not args.input.is_dir():
        print(f"input directory does not exist: {args.input}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    staged = 0
    for subdir in sorted(args.input.iterdir()):
        if not subdir.is_dir():
            continue
        backend = backend_from_artifact_dir(subdir.name)
        if backend is None:
            continue
        for whl in sorted(subdir.glob("lilbee-*.whl")):
            asset_name = rename_for_release(whl.name, backend)
            target = args.output / asset_name
            if target.exists():
                print(f"asset name collision: {asset_name} already staged", file=sys.stderr)
                return 1
            shutil.copy2(whl, target)
            staged += 1
            print(asset_name)
    print(f"staged {staged} wheels", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
