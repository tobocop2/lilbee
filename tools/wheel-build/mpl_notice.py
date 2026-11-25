"""Write SOURCE-MPL-2.0: one line for each MPL-2.0 crate, with its .crate URL and sha256.

Usage: mpl_notice.py <cargo-about-json> <Cargo.lock> <output>

MPL-2.0 section 3.2 needs a way to get the source of each file under the licence. A crate on
crates.io keeps every published version, so the notice names the exact .crate file and its sha256
from Cargo.lock. The pack licence check accepts only crate lines in the file; the MPL-2.0 text ships
in THIRD-PARTY/tei/LICENSES.txt. Writes nothing when no crate uses MPL-2.0.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

CRATE_URL = "https://static.crates.io/crates/{name}/{name}-{version}.crate"


def mpl_crates(about: dict) -> list[tuple[str, str]]:
    """The sorted (name, version) of every crate that the cargo-about report puts under MPL-2.0."""
    crates: set[tuple[str, str]] = set()
    for entry in about["licenses"]:
        if entry["id"] == "MPL-2.0":
            crates.update(
                (use["crate"]["name"], use["crate"]["version"]) for use in entry["used_by"]
            )
    return sorted(crates)


def checksums(lock: dict) -> dict[tuple[str, str], str]:
    """The sha256 of each registry package in a Cargo.lock, by name and version."""
    return {
        (pkg["name"], pkg["version"]): pkg["checksum"]
        for pkg in lock["package"]
        if "checksum" in pkg
    }


def notice(about: dict, lock: dict) -> str | None:
    """The notice text, or None when no crate uses MPL-2.0."""
    crates = mpl_crates(about)
    if not crates:
        return None
    sums = checksums(lock)
    lines = []
    for name, version in crates:
        checksum = sums.get((name, version))
        if checksum is None:
            raise SystemExit(f"Cargo.lock has no checksum for {name} {version}")
        lines.append(f"{CRATE_URL.format(name=name, version=version)}  {checksum}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        raise SystemExit(__doc__)
    about = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    lock = tomllib.loads(Path(argv[2]).read_text(encoding="utf-8"))
    text = notice(about, lock)
    if text is not None:
        Path(argv[3]).write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(sys.argv)
