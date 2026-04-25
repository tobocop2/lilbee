"""Parse `lilbee --json self-check` output and assert the payload reports
a real chat response and a positive embedding-dimension count.

The CLI prints log lines before the final JSON line; we walk the lines
in reverse and pick the first one that starts with '{' so log noise
doesn't break parsing. Used by build-wheels.yml's wheel smoke, the
verify-pypi lane, and release.yml's exe smoke.

Usage: python tools/wheel-build/parse_self_check.py <output-file>
"""

from __future__ import annotations

import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <self-check-output-file>", file=sys.stderr)
        return 2
    raw = pathlib.Path(sys.argv[1]).read_text().strip().splitlines()
    payload = next(line for line in reversed(raw) if line.strip().startswith("{"))
    data = json.loads(payload)
    assert data.get("ok"), f"self-check failed: {data}"
    assert data.get("chat_response", "").strip(), f"empty chat response: {data}"
    dims = data.get("embedding_dims")
    assert isinstance(dims, int) and dims > 0, f"missing embedding_dims: {data}"
    print(f"self-check chat={data['chat_response']!r} embedding_dims={dims}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
