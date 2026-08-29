#!/usr/bin/env python3
"""Render the "New model architectures" section for a release's notes.

A llama.cpp pin bump commit says only "Bump the bundled llama.cpp engine to
<ref>". The model support that bump adds is its user-visible payload, so the
notes generate the list rather than relying on someone to write it: this reads
``SUPPORTED_ARCHS`` out of ``lilbee/_generated/engine_archs.py`` at two tags and
tables the difference.

Text parsing, not import: both sides come from a tag rather than the working
tree, so neither is importable as a module.

Usage:
    release_arch_table.py --old-file <previous> --new-file <current>

The caller supplies the two files, because the release job that needs this
checks out at depth 1 and reads them over the API. To run it against two tags
by hand::

    git show <previous-tag>:src/lilbee/_generated/engine_archs.py > /tmp/old.py
    git show <tag>:src/lilbee/_generated/engine_archs.py > /tmp/new.py
    python tools/release_arch_table.py --old-file /tmp/old.py --new-file /tmp/new.py

Prints nothing when the pin did not move or added no architectures, so the
caller can append the output unconditionally.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = "https://github.com/ggml-org/llama.cpp"
_HEADING = "## New model architectures"
# The section runs to the next second-level heading, so a body that already has
# sections after it keeps them.
_SECTION = re.compile(rf"\n*{re.escape(_HEADING)}\n.*?(?=\n## |\Z)", re.DOTALL)

_ARCH_ENTRY = re.compile(r'^\s*"([a-z0-9._-]+)",\s*$', re.IGNORECASE | re.MULTILINE)
_SET_BLOCK = re.compile(r"SUPPORTED_ARCHS.*?frozenset\(\s*\{(.*?)\}\s*\)", re.DOTALL)
_REF = re.compile(r'^ENGINE_LLAMA_CPP_REF\s*=\s*"([^"]+)"', re.MULTILINE)
_COMMIT = re.compile(r'^LLAMA_CPP_COMMIT\s*=\s*"([^"]+)"', re.MULTILINE)


def _supported_archs(source: str) -> set[str]:
    """The architecture names in ``source``'s SUPPORTED_ARCHS frozenset."""
    block = _SET_BLOCK.search(source)
    if not block:
        return set()
    return set(_ARCH_ENTRY.findall(block.group(1)))


def engine_ref(source: str) -> str:
    """The llama.cpp ref the engine was built from, or "" when absent."""
    found = _REF.search(source)
    return found.group(1) if found else ""


def engine_commit(source: str) -> str:
    """The llama.cpp commit the engine was built from, or "" when absent."""
    found = _COMMIT.search(source)
    return found.group(1) if found else ""


def added_archs(old: str, new: str) -> list[str]:
    """Architecture names present in ``new`` and not in ``old``, sorted.

    Removals are not reported. An architecture leaving the engine's table is a
    support regression, which belongs in the notes as prose, not in a table
    headed "new".
    """
    old_set = _supported_archs(old)
    if not old_set:
        return []
    return sorted(_supported_archs(new) - old_set)


def render(old: str, new: str) -> str:
    """The markdown section for this bump, or "" when there is nothing to say."""
    added = added_archs(old, new)
    if not added:
        return ""
    old_commit = engine_commit(old)
    new_commit = engine_commit(new)
    total_new = len(_supported_archs(new))
    total_old = len(_supported_archs(old))

    lines = [
        "## New model architectures",
        "",
        f"The bundled llama.cpp engine moves to `{engine_ref(new)}`, which adds "
        f"{len(added)} architecture{'s' if len(added) != 1 else ''}.",
        "",
        "| architecture |",
        "|---|",
    ]
    lines += [f"| `{arch}` |" for arch in added]
    lines += [
        "",
        f"lilbee now runs {total_new} GGUF architectures, up from {total_old}. "
        "Pull any GGUF built on one of them and it works.",
    ]
    if old_commit and new_commit:
        lines.append(
            f"The engine changes are in [{old_commit[:9]}...{new_commit[:9]}]"
            f"({_REPO}/compare/{old_commit}...{new_commit})."
        )
    return "\n".join(lines) + "\n"


def apply_to_body(body: str, table: str) -> str:
    """``body`` carrying exactly ``table`` as its architecture section.

    Idempotent: an existing section is replaced rather than appended to, because
    release-selfheal.yml makes a rerun of the job that writes these notes
    routine, and a plain append would stack a copy per retry. An empty ``table``
    strips a section that no longer applies.
    """
    stripped = _SECTION.sub("", body).rstrip("\n")
    if not table:
        return stripped + "\n" if body.endswith("\n") else stripped
    return f"{stripped}\n\n{table.strip()}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-file", type=Path, required=True, help="engine_archs.py at the previous tag"
    )
    parser.add_argument(
        "--new-file", type=Path, required=True, help="engine_archs.py at the tag being released"
    )
    parser.add_argument(
        "--body-file",
        type=Path,
        help="existing release body; prints that body with the section applied instead of the "
        "section alone, so the caller can skip the write when nothing changed",
    )
    args = parser.parse_args(argv)

    table = render(
        args.old_file.read_text(encoding="utf-8"),
        args.new_file.read_text(encoding="utf-8"),
    )
    if args.body_file:
        sys.stdout.write(apply_to_body(args.body_file.read_text(encoding="utf-8"), table))
    else:
        sys.stdout.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
