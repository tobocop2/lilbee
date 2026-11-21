"""Write SOURCE-MPL-2.0 for a Python pack: one line for each MPL-2.0 package of the frozen tree.

Usage: mpl_pypi_notice.py <mpl> <sources> <output>

<mpl> is the list that collect_python_licences.py writes: one ``<name>-<version>`` for each package
of the tree whose terms hold MPL-2.0. <sources> is the committed list of source archives, one line
for each package: the PyPI URL of the source archive, then its sha256. MPL-2.0 section 3.2 needs a
way to get the source of each file under the licence, so the notice names the exact archive.

The script writes the line of each package that the tree holds, and nothing else. It stops when an
MPL-2.0 package has no line in <sources>, so a bump that adds an MPL-2.0 package cannot build a
wheel whose notice leaves it out. It writes nothing when <mpl> is empty. The pack licence check
accepts only the lines that engine_crates.MPL_PYPI_CHECKSUMS lists.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ARCHIVE = re.compile(r"(?P<name>.+)-(?P<version>\d[^-]*)")


def _key(archive: str) -> tuple[str, str] | None:
    """The (name, version) of ``<name>-<version>``, with the name in the form of PEP 503."""
    found = ARCHIVE.fullmatch(archive)
    if found is None:
        return None
    return re.sub(r"[-_.]+", "-", found["name"]).lower(), found["version"]


def notice(packages: list[str], sources: list[str]) -> str | None:
    """The notice text, or None when the tree holds no MPL-2.0 package."""
    if not packages:
        return None
    rows: dict[tuple[str, str], str] = {}
    for line in sources:
        parts = line.split()
        if len(parts) == 2 and (key := _key(parts[0].rsplit("/", 1)[-1].removesuffix(".tar.gz"))):
            rows[key] = " ".join(parts)
    lines = []
    for package in packages:
        row = rows.get(_key(package) or ("", ""))
        if row is None:
            raise SystemExit(f"the list of source archives has no line for {package}")
        lines.append(row)
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        raise SystemExit(__doc__)
    packages = Path(argv[1]).read_text(encoding="utf-8").split()
    sources = Path(argv[2]).read_text(encoding="utf-8").splitlines()
    text = notice(packages, sources)
    if text is not None:
        Path(argv[3]).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
