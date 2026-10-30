"""Fail when a licensecheck JSON report lists a GPL or AGPL entry in any package licence list."""

from __future__ import annotations

import json
import sys
import tomllib
from fnmatch import fnmatch
from pathlib import Path

from licensecheck.license_matrix import licenseType


def copyleft_packages(report: dict, ignore_packages: list[str]) -> list[str]:
    """Name each package whose licence list holds a GPL or AGPL entry."""
    ignored = [pattern.upper() for pattern in ignore_packages]
    found = []
    for package in report["packages"]:
        if any(fnmatch(package["name"].upper(), pattern) for pattern in ignored):
            continue
        kinds = licenseType(str(package["license"]), set())
        if any(kind.name.startswith(("GPL", "AGPL")) for kind in kinds):
            found.append(f"{package['name']}: {package['license']}")
    return found


def main(report_path: str, pyproject_path: str) -> int:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    pyproject = tomllib.loads(Path(pyproject_path).read_text(encoding="utf-8"))
    ignore = pyproject["tool"]["licensecheck"]["ignore_packages"]
    found = copyleft_packages(report, ignore)
    for line in found:
        print(f"copyleft licence in list: {line}")
    for package in report["packages"]:
        if not package["licenseCompat"]:
            print(f"incompatible licence: {package['name']}: {package['license']}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
