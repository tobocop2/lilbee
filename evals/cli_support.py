"""Shared plumbing for the two command-line entry points.

Both CLIs write JSONL and both render a results file to markdown, and both had
their own copy. The rendering copy in particular was reported as duplication by
four consecutive review passes and deferred each time as pre-existing, which is
how a two-line smell survives a rewrite of everything around it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace ``path`` with one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Add rows to ``path`` without disturbing what is already there.

    Appending is what lets separate scoring stages, and separate machines,
    contribute to one results file rather than the last one to finish erasing
    the rest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def render_to_file(
    results: Path, out: Path, renderer: Callable[[list[dict[str, Any]]], str]
) -> int:
    """Render a results JSONL through ``renderer`` and write the markdown."""
    from evals.retrieval.checkpoint import load_jsonl

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(renderer(load_jsonl(results)))
    print(f"wrote {out}")
    return 0
