#!/usr/bin/env python3
"""Generate the lilbee.sh/packs/ PEP 503 index for the public engine packs.

Each engine pack is a wheel attached to a ``engine-packs-<date>[-<name>]`` release
of the lilbee repository. This writes one simple-repository page per project under
``<out>/packs/``, linking to those release assets with their sha256, so
``pip download lilbee-engine-tei --index-url https://lilbee.sh/packs/`` resolves
them. Draft releases answer 404 to the public and are skipped. Run by pages.yml at
deploy time next to build_compat_index.py.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

# A dated pack release. The pin tags (engine-packs-pin-N) and other releases do not match.
_PACK_TAG = re.compile(r"^engine-packs-\d{8}(-|$)")


def project_of(filename: str) -> str:
    """Return the PEP 503 normalized project name of a wheel filename."""
    return re.sub(r"[-_.]+", "-", filename.split("-", 1)[0]).lower()


def fetch_wheels(repo: str) -> list[tuple[str, str, str]]:
    """Return (filename, download_url, sha256) for every wheel in a public pack release."""
    # gh is a trusted CLI on PATH and `repo` is a controlled constant, not user input.
    out = subprocess.check_output(
        ["gh", "api", "--paginate", f"repos/{repo}/releases", "--jq", ".[]"],
        text=True,
    )
    wheels: dict[str, tuple[str, str, str]] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        release = json.loads(line)
        if release["draft"] or not _PACK_TAG.match(release["tag_name"]):
            continue
        for asset in release["assets"]:
            name = asset["name"]
            if not name.endswith(".whl"):
                continue
            digest = (asset.get("digest") or "").removeprefix("sha256:")
            if not digest:
                raise SystemExit(f"asset {name} has no sha256 digest from the API")
            if name in wheels and wheels[name][2] != digest:
                raise SystemExit(f"wheel {name} appears with two different digests")
            wheels[name] = (name, asset["browser_download_url"], digest)
    return list(wheels.values())


def _page(title: str, rows: list[str]) -> str:
    body = "\n".join(f"    {row}<br/>" for row in rows)
    return f"<!DOCTYPE html>\n<html><body>\n  <h1>{title}</h1>\n{body}\n</body></html>\n"


def write_index(out_dir: Path, wheels: list[tuple[str, str, str]]) -> None:
    """Write ``<out_dir>/packs/index.html`` and one page per project below it."""
    root = out_dir / "packs"
    by_project: dict[str, list[tuple[str, str, str]]] = {}
    for wheel in sorted(wheels):
        by_project.setdefault(project_of(wheel[0]), []).append(wheel)
    for project, items in by_project.items():
        (root / project).mkdir(parents=True, exist_ok=True)
        rows = [
            f'<a href="{html.escape(url)}#sha256={sha}">{html.escape(name)}</a>'
            for name, url, sha in items
        ]
        (root / project / "index.html").write_text(_page(f"Links for {project}", rows))
    (root / "index.html").write_text(
        _page("Simple index", [f'<a href="{p}/">{p}</a>' for p in sorted(by_project)])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output site dir; writes <out>/packs/")
    ap.add_argument("--repo", default="tobocop2/lilbee")
    args = ap.parse_args()

    wheels = fetch_wheels(args.repo)
    if not wheels:
        print(f"no engine pack wheels found in {args.repo} releases", file=sys.stderr)
        return 1
    write_index(Path(args.out), wheels)
    print(f"pack index: {len(wheels)} wheels from {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
