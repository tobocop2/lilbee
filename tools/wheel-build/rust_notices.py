"""Write the notice files that a Rust binary's crates need beyond their licence texts.

Usage: rust_notices.py <cargo-about JSON> <Cargo.lock> <source checkout> <wheel tree>

It writes SOURCE-MPL-2.0 (the crates.io source URL and checksum of each MPL-2.0 crate) and
NOTICE-OpenSSL (the acknowledgements that the OpenSSL licence asks for) into the wheel tree.
It copies each NOTICE file, and each licence file under a third_party folder, of the source
checkout and of the crates in the graph into THIRD-PARTY/vendored. The crate sources come from
the cargo registry. The names of the files written go to standard output, one on each line.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tomllib
from pathlib import Path

CRATES_INDEX = "registry+https://github.com/rust-lang/crates.io-index"
OPENSSL_NOTICE = (
    "This product includes software developed by the OpenSSL Project for use in the OpenSSL Toolkit "
    "(http://www.openssl.org/).\n"
    "This product includes cryptographic software written by Eric Young (eay@cryptsoft.com).\n"
    "This product includes software written by Tim Hudson (tjh@cryptsoft.com).\n"
)
_LICENCE_PREFIXES = ("LICENSE", "LICENCE", "COPYING")


def lock_checksums(lock_text: str) -> dict[tuple[str, str], str]:
    """Map (name, version) to the checksum of each crates.io package in a Cargo.lock."""
    packages = tomllib.loads(lock_text)["package"]
    return {
        (pkg["name"], pkg["version"]): pkg["checksum"]
        for pkg in packages
        if pkg.get("source") == CRATES_INDEX and "checksum" in pkg
    }


def graph_crates(report: dict) -> list[tuple[str, str]]:
    """List the (name, version) of every crate in the report, sorted."""
    return sorted(
        {(use["crate"]["name"], use["crate"]["version"]) for entry in report["licenses"] for use in entry["used_by"]}
    )


def mpl_notice(report: dict, checksums: dict[tuple[str, str], str]) -> str | None:
    """Return the text of SOURCE-MPL-2.0, or None when no crate uses MPL-2.0."""
    entries = [entry for entry in report["licenses"] if entry["id"] == "MPL-2.0"]
    if not entries:
        return None
    lines = []
    for name, version in graph_crates({"licenses": entries}):
        if (name, version) not in checksums:
            raise SystemExit(f"{name} {version} is not a crates.io package in Cargo.lock: no source URL")
        lines.append(f"https://static.crates.io/crates/{name}/{name}-{version}.crate  {checksums[(name, version)]}")
    return "\n".join(lines) + "\n"


def vendored_files(root: Path) -> list[Path]:
    """List the NOTICE files, and the licence files under a third_party folder, below *root*."""
    found = []
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if not path.is_file() or "target" in parts or ".git" in parts:
            continue
        name = path.name.upper()
        if name.startswith("NOTICE") or ("third_party" in parts and name.startswith(_LICENCE_PREFIXES)):
            found.append(path)
    return found


def copy_vendored(roots: list[tuple[str, Path]], out: Path) -> list[str]:
    """Copy the vendored files of each (label, folder) into *out*; return the new names."""
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for label, root in roots:
        for path in vendored_files(root):
            name = f"{label}__{'__'.join(path.relative_to(root).parts)}"
            shutil.copyfile(path, out / name)
            copied.append(name)
    return copied


def write_notices(report: dict, lock_text: str, checkout: Path, registry: Path, tree: Path) -> list[str]:
    """Write every notice file into the wheel tree *tree*; return the names written."""
    names = []
    mpl = mpl_notice(report, lock_checksums(lock_text))
    if mpl is not None:
        (tree / "SOURCE-MPL-2.0").write_text(mpl, encoding="utf-8", newline="\n")
        names.append("SOURCE-MPL-2.0")
    if any(entry["id"] == "OpenSSL" for entry in report["licenses"]):
        (tree / "NOTICE-OpenSSL").write_text(OPENSSL_NOTICE, encoding="utf-8", newline="\n")
        names.append("NOTICE-OpenSSL")
    roots = [("mistral.rs", checkout)]
    for name, version in graph_crates(report):
        hits = sorted(registry.glob(f"*/{name}-{version}"))
        if hits:
            roots.append((f"{name}-{version}", hits[0]))
    names += copy_vendored(roots, tree / "THIRD-PARTY" / "vendored")
    return names


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    cargo_home = Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo")
    lock_text = Path(argv[2]).read_text(encoding="utf-8")
    for name in write_notices(report, lock_text, Path(argv[3]), cargo_home / "registry" / "src", Path(argv[4])):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
