#!/usr/bin/env python3
"""Build the retrieval corpus from the pinned tree: paragraphs as documents, headings as queries.

    make_corpus.py <repo> <sha> <out.json>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

PATHS = ("README.md", "AGENTS.md", "CONTRIBUTING.md", "docs/usage.md", "docs/architecture.md", "docs/development.md")
MAX_CHARS = 700
MIN_CHARS = 80
QUERIES = 80


def main() -> int:
    repo, sha, out = sys.argv[1:4]
    docs: list[str] = []
    queries: list[str] = []
    for path in PATHS:
        text = subprocess.run(
            ["git", "-C", repo, "show", f"{sha}:{path}"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if block.startswith("#"):
                title = block.splitlines()[0].lstrip("# ").strip()
                if len(title) > 8:
                    queries.append(f"What does the documentation say about {title}?")
                continue
            if MIN_CHARS <= len(block):
                docs.append(block[:MAX_CHARS])
    seen: set[str] = set()
    unique = [d for d in docs if not (d in seen or seen.add(d))]
    step = max(1, len(queries) // QUERIES)
    payload = {"sha": sha, "docs": unique, "queries": queries[::step][:QUERIES]}
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print(f"docs={len(unique)} queries={len(payload['queries'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
