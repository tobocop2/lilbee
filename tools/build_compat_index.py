#!/usr/bin/env python3
"""Generate a PEP 503 index for wheels hosted as GitHub release assets on a fork.

Two indexes on lilbee.sh come from this script:

* ``compat/``: the pre-Haswell ``lancedb`` wheels built by the tobocop2/lancedb
  fork (releases tagged ``lancedb-v<version>+compat``).
* ``xberg/``: TEMPORARY. The ``xberg`` wheels built by the tobocop2/kreuzberg
  fork (releases tagged ``xberg-v<version>+lilbee<n>``), which lilbee pins only
  until upstream releases the EPUB fixes; its pages.yml step goes away with
  the pin. ``--mirror-into-backends`` also writes the
  project page into every per-backend index next to it, so one
  ``--extra-index-url https://lilbee.sh/<backend>/`` resolves the engine and
  xberg together.

Run by pages.yml at deploy time alongside build_pep503_indexes.py.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path


def fetch_wheels(repo: str, tag_pattern: str) -> list[tuple[str, str, str]]:
    """Return (filename, download_url, sha256) for every wheel asset on matching releases."""
    jq = (
        f'.[] | select(.tag_name | test("{tag_pattern}")) | .assets[] '
        '| select(.name | endswith(".whl")) '
        "| {name, url: .browser_download_url, digest}"
    )
    # gh is a trusted CLI on PATH and `repo` is a controlled constant (the fork),
    # not user input (S603/S607 are ignored for this file in pyproject).
    out = subprocess.check_output(
        ["gh", "api", "--paginate", f"repos/{repo}/releases", "--jq", jq],
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


def project_page(project: str, wheels: list[tuple[str, str, str]]) -> str:
    rows = "\n".join(
        f'    <a href="{html.escape(url)}#sha256={sha}">{html.escape(name)}</a><br/>'
        for name, url, sha in sorted(wheels)
    )
    return f"<!DOCTYPE html>\n<html><body>\n  <h1>Links for {project}</h1>\n{rows}\n</body></html>\n"


def write_index(root: Path, project: str, page: str) -> None:
    """Write the project page under ``root`` and make sure the root index links it."""
    (root / project).mkdir(parents=True, exist_ok=True)
    (root / project / "index.html").write_text(page)
    link = f'    <a href="{project}/">{project}</a><br/>\n'
    index = root / "index.html"
    if not index.exists():
        index.write_text(f"<!DOCTYPE html>\n<html><body>\n{link}</body></html>\n")
    elif link not in index.read_text():
        index.write_text(index.read_text().replace("</body>", f"{link}</body>", 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="site dir; writes <out>/<subdir>/")
    ap.add_argument("--repo", default="tobocop2/lancedb")
    ap.add_argument("--project", default="lancedb")
    ap.add_argument("--tag-pattern", default="compat", help="regex a release tag must match")
    ap.add_argument("--subdir", default="compat")
    ap.add_argument(
        "--mirror-into-backends",
        action="store_true",
        help="also write the project page into every <out>/<backend>/ index that exists",
    )
    args = ap.parse_args()

    wheels = fetch_wheels(args.repo, args.tag_pattern)
    if not wheels:
        print(f"no {args.project} wheels found in {args.repo} releases", file=sys.stderr)
        return 1

    page = project_page(args.project, wheels)
    out = Path(args.out)
    write_index(out / args.subdir, args.project, page)
    mirrored = []
    if args.mirror_into_backends:
        for backend in sorted(d for d in out.iterdir() if d.is_dir() and d.name != args.subdir):
            if (backend / "index.html").exists():
                write_index(backend, args.project, page)
                mirrored.append(backend.name)
    print(
        f"{args.subdir} index: {len(wheels)} {args.project} wheels from {args.repo}"
        + (f", mirrored into {mirrored}" if mirrored else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
