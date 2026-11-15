#!/usr/bin/env python3
"""Generate the lilbee.sh/compat/ PEP 503 index from fork release wheels.

Each fork hosts its wheels as release assets (releases whose tag ends in
``+compat`` or ``+lilbee``). This emits a simple-repository index under
``<out>/compat/`` with one project page per wheel project, linking to those
assets, so ``pip install ... --extra-index-url https://lilbee.sh/compat/``
resolves them. Run by pages.yml at deploy time alongside build_pep503_indexes.py.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

# A fork release tag ends in the local version its wheels carry: +compat, or +lilbee for xberg.
_COMPAT_TAG = r"\\+(compat|lilbee)$"
_JQ = (
    f'.[] | select(.tag_name | test("{_COMPAT_TAG}")) | .assets[] '
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


def project_of(wheel: str) -> str:
    """Return the PEP 503 normalized project name of a wheel filename."""
    return re.sub(r"[-_.]+", "-", wheel.split("-", 1)[0]).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output site dir; writes <out>/compat/")
    ap.add_argument(
        "--repo",
        action="append",
        default=[],
        help="fork whose compat wheels must exist (repeatable)",
    )
    ap.add_argument(
        "--optional-repo",
        action="append",
        default=[],
        help="fork whose compat wheels are listed when it has any (repeatable)",
    )
    args = ap.parse_args()
    if not args.repo and not args.optional_repo:
        ap.error("give at least one --repo or --optional-repo")

    projects: dict[str, list[tuple[str, str, str]]] = {}
    for repo in args.repo + args.optional_repo:
        wheels = fetch_wheels(repo)
        if not wheels and repo in args.repo:
            print(f"no compat wheels found in {repo} releases", file=sys.stderr)
            return 1
        if not wheels:
            print(f"no compat wheels in {repo} yet; skipped")
        for wheel in wheels:
            projects.setdefault(project_of(wheel[0]), []).append(wheel)

    out = Path(args.out) / "compat"
    out.mkdir(parents=True, exist_ok=True)
    for project, wheels in sorted(projects.items()):
        rows = "\n".join(
            f'    <a href="{html.escape(url)}#sha256={sha}">{html.escape(name)}</a><br/>'
            for name, url, sha in sorted(wheels)
        )
        (out / project).mkdir(exist_ok=True)
        page = f"  <h1>Links for {project}</h1>\n{rows}"
        (out / project / "index.html").write_text(
            f"<!DOCTYPE html>\n<html><body>\n{page}\n</body></html>\n"
        )
    root = "\n".join(f'    <a href="{p}/">{p}</a><br/>' for p in sorted(projects))
    (out / "index.html").write_text(f"<!DOCTYPE html>\n<html><body>\n{root}\n</body></html>\n")
    count = sum(len(w) for w in projects.values())
    print(f"compat index: {count} wheels in {len(projects)} projects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
