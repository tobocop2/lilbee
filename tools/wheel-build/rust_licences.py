"""Write the licence texts of a Rust binary's crates and print their SPDX ids.

Usage: rust_licences.py <cargo-about JSON> <output folder>

The JSON is the output of `cargo about generate --format json` over the locked dependency graph.
Each licence text becomes one file in the output folder; its header names the crates that the
text covers and where their sources are. The ids go to standard output, one on each line.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def write_licences(report: dict, out: Path) -> list[str]:
    """Write one file for each licence text of *report* into *out*; return the sorted ids."""
    texts = report["licenses"]
    if not texts:
        raise SystemExit("the report lists no licence text")
    out.mkdir(parents=True, exist_ok=True)
    count: dict[str, int] = {}
    for entry in texts:
        if not entry["text"].strip():
            raise SystemExit(f"the {entry['id']} text of {entry['used_by'][0]['crate']['name']} is empty")
        count[entry["id"]] = count.get(entry["id"], 0) + 1
        name = f"{_UNSAFE_NAME.sub('_', entry['id'])}-{count[entry['id']]:03d}.txt"
        crates = sorted(
            f"{use['crate']['name']} {use['crate']['version']} ({use['crate']['repository'] or 'no repository'})"
            for use in entry["used_by"]
        )
        head = f"{entry['name']} ({entry['id']}). It covers these crates:\n" + "\n".join(crates)
        (out / name).write_text(f"{head}\n\n{'-' * 72}\n\n{entry['text'].strip()}\n", encoding="utf-8", newline="\n")
    return sorted({entry["id"] for entry in texts})


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    for licence_id in write_licences(report, Path(argv[2])):
        print(licence_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
