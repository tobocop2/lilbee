"""Set the version, the licence expression and the licence files in an engine wheel project.

Usage: stamp_engine_wheel.py <pyproject.toml> <version> <licence-expression>

The licence files are every LICENSE-*, NOTICE-*, SOURCE-* and THIRD-PARTY file beside the project file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LICENCE_FILES = '["LICENSE-*", "NOTICE-*", "SOURCE-*", "THIRD-PARTY/**"]'


def stamp(text: str, version: str, expression: str) -> str:
    """The project file text with the version, the licence expression and the licence files set."""
    text, count = re.subn(r'(?m)^version = ".*"$', f'version = "{version}"', text, count=1)
    if count != 1:
        raise SystemExit("the project file has no version line")
    marker = "requires-python"
    if marker not in text:
        raise SystemExit("the project file has no requires-python line")
    return text.replace(
        marker, f'license = "{expression}"\nlicense-files = {LICENCE_FILES}\n{marker}', 1
    )


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        raise SystemExit(__doc__)
    path = Path(argv[1])
    path.write_text(stamp(path.read_text(encoding="utf-8"), argv[2], argv[3]), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
