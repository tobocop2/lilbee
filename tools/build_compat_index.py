#!/usr/bin/env python3
"""Generate the lilbee.sh/compat/ PEP 503 index for the pre-Haswell lancedb wheels.

The compat lancedb wheels are built by the tobocop2/lancedb fork and hosted as
release assets there (releases tagged ``lancedb-v<version>+compat``). This emits
a simple-repository index under ``<out>/compat/`` linking to those assets, so
``pip install ... --extra-index-url https://lilbee.sh/compat/`` resolves them.
Run by pages.yml at deploy time alongside build_pep503_indexes.py.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

_JQ = (
    '.[] | select(.tag_name | test("compat")) | .assets[] '
    '| select(.name | endswith(".whl")) '
    "| {name, url: .browser_download_url, digest}"
)


def fetch_wheels(repo: str) -> list[tuple[str, str, str]]:
    """Return (filename, download_url, sha256) for every compat-release wheel asset."""
    # gh is a trusted CLI on PATH and `repo` is a controlled constant (the fork),
    # not user input (S603/S607 are ignored for this file in pyproject).
    out = subprocess.check_output(
        ["gh", "api", "--paginate", f"repos/{repo}/releases", "--jq", _JQ],
        text=True,
    )
    wheels: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        digest = (a.get("digest") or "").removeprefix("sha256:")
        if not digest:
            raise SystemExit(f"asset {a['name']} has no sha256 digest from the API")
        wheels.append((a["name"], a["url"], digest))
    return wheels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output site dir; writes <out>/compat/")
    ap.add_argument("--repo", default="tobocop2/lancedb")
    args = ap.parse_args()

    wheels = fetch_wheels(args.repo)
    if not wheels:
        print(f"no compat wheels found in {args.repo} releases", file=sys.stderr)
        return 1

    out = Path(args.out) / "compat"
    (out / "lancedb").mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f'    <a href="{html.escape(url)}#sha256={sha}">{html.escape(name)}</a><br/>'
        for name, url, sha in sorted(wheels)
    )
    (out / "lancedb" / "index.html").write_text(
        f"<!DOCTYPE html>\n<html><body>\n  <h1>Links for lancedb</h1>\n{rows}\n</body></html>\n"
    )
    (out / "index.html").write_text(
        '<!DOCTYPE html>\n<html><body>\n    <a href="lancedb/">lancedb</a><br/>\n</body></html>\n'
    )
    print(f"compat index: {len(wheels)} wheels from {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
