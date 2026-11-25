"""Coverage gate for ``make test``: every line, and every branch outside two packages.

pytest-cov runs with branch data on and its own threshold off (``--cov-fail-under=0``),
because coverage.py has no per-package branch switch: one threshold would count the
branches of the packages that are not ready. This script reads the ``coverage json`` report
instead. Every file must have no missed line. Every file outside ``LINE_ONLY`` must also
have no missed branch.

``LINE_ONLY`` shrinks as each package reaches zero missed branches. When it is empty,
delete this script and set ``branch = true`` in ``[tool.coverage.run]``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LINE_ONLY = ("src/lilbee/cli/tui/", "src/lilbee/providers/fleet/")


def violations(report: dict) -> list[str]:
    """Return one message for each file that misses a line, or a branch it must cover."""
    found = []
    for path, data in sorted(report["files"].items()):
        posix = path.replace("\\", "/")
        missing_lines = data["missing_lines"]
        if missing_lines:
            found.append(f"{posix}: missed lines {missing_lines}")
        if posix.startswith(LINE_ONLY):
            continue
        missing_branches = data["missing_branches"]
        if missing_branches:
            arcs = ", ".join(f"{src}->{dst}" for src, dst in missing_branches)
            found.append(f"{posix}: missed branches {arcs}")
    return found


def main(argv: list[str]) -> int:
    report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    found = violations(report)
    for line in found:
        print(f"FAIL {line}")
    if found:
        print(f"branch gate: {len(found)} file(s) below 100%")
        return 1
    print("branch gate: every line covered; every branch covered outside " + ", ".join(LINE_ONLY))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
