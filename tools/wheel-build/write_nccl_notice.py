"""Write the notice for the NCCL library that a CUDA wheel carries.

Usage: write_nccl_notice.py <libnccl file> <output file>

The library names its own version. The notice repeats the licence of that release, which NVIDIA
publishes in the NCCL repository under the tag `v<version>-1`.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

_VERSION = re.compile(rb"NCCL version (\d+\.\d+\.\d+)")
_LICENCE_URL = "https://raw.githubusercontent.com/NVIDIA/nccl/v{version}-1/LICENSE.txt"
_MIN_TEXT = 1000


def nccl_version(library: Path) -> str:
    """The NCCL release that *library* names in its own text."""
    with library.open("rb") as handle:
        found = _VERSION.search(handle.read())
    if found is None:
        raise SystemExit(f"{library.name} does not name its NCCL version")
    return found.group(1).decode("ascii")


def notice(library: str, version: str, text: str) -> str:
    return (
        f"The shared library bin/lib/{library} is NCCL {version}, from https://github.com/NVIDIA/nccl\n"
        f"(tag v{version}-1). NVIDIA publishes it under the BSD-3-Clause licence, which follows.\n\n"
        f"{text.strip()}\n"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    library = Path(argv[1])
    version = nccl_version(library)
    url = _LICENCE_URL.format(version=version)
    text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    if "BSD" not in text and "Redistribution" not in text or len(text) < _MIN_TEXT:
        raise SystemExit(f"the text from {url} is not the NCCL licence")
    Path(argv[2]).write_text(notice(library.name, version, text), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
