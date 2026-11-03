#!/usr/bin/env python3
"""Build the PEP 503 indexes behind lilbee.sh with dumb-pypi.

Usage:
    build_wheel_indexes.py engines ARTIFACTS SITE [--release-base-url URL]
    build_wheel_indexes.py compat SITE [--repo OWNER/NAME]

``engines`` reads the engine wheels from the ``wheel-multigpu-<os>-<backend>`` artifact
directories and writes one index for each backend under ``SITE/<backend>/``. ``compat`` writes
the index of the pre-Haswell lancedb wheels under ``SITE/compat/``. Each link points at a
release asset and carries the sha256 of the file. dumb-pypi is pinned in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_ARTIFACT_DIR = re.compile(r"^wheel-multigpu-(?P<rest>.+)$")
_ENGINE_WHEEL = re.compile(r"^lilbee_engine_llama_cpp-(?P<version>[^-]+)-")
_DEFAULT_RELEASE_BASE_URL = "https://github.com/tobocop2/lilbee/releases/download"
_NO_BUILD_TAG_PARTS = 5
# dumb-pypi links every file to one prefix. The real url replaces this one per file.
_PLACEHOLDER_URL = "https://packages.invalid"
_COMPAT_JQ = (
    '.[] | select(.tag_name | test("compat")) | .assets[] '
    '| select(.name | endswith(".whl")) '
    "| {name, url: .browser_download_url, digest}"
)


@dataclass(frozen=True)
class Wheel:
    """A wheel file name, its sha256 and the url of the release asset."""

    filename: str
    sha256: str
    url: str


def backend_of(artifact_dir: str) -> str | None:
    """The backend in ``wheel-multigpu-<os>-<backend>``, or None for another name."""
    found = _ARTIFACT_DIR.match(artifact_dir)
    return found.group("rest").rsplit("-", 1)[-1] if found else None


def release_name(wheel_name: str, backend: str) -> str:
    """The wheel name on the release: the build tag ``1.<backend>`` follows the version."""
    parts = wheel_name.removesuffix(".whl").split("-")
    if len(parts) != _NO_BUILD_TAG_PARTS:
        raise ValueError(
            f"wheel filename {wheel_name!r} does not match the no-buildtag layout "
            f"(project-version-python-abi-platform.whl)"
        )
    project, version, python, abi, platform = parts
    return f"{project}-{version}-1.{backend}-{python}-{abi}-{platform}.whl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def engine_wheels(artifacts: Path, release_base_url: str) -> dict[str, list[Wheel]]:
    """The engine wheels of each backend, named and linked as on the release."""
    by_backend: dict[str, list[Wheel]] = {}
    for child in sorted(artifacts.iterdir()):
        backend = backend_of(child.name) if child.is_dir() else None
        if backend is None:
            continue
        for path in sorted(child.glob("lilbee_engine_llama_cpp-*.whl")):
            found = _ENGINE_WHEEL.match(path.name)
            if found is None:
                print(f"skipping unparseable wheel filename: {path.name}", file=sys.stderr)
                continue
            asset = release_name(path.name, backend)
            url = f"{release_base_url.rstrip('/')}/v{found.group('version')}/{asset}"
            by_backend.setdefault(backend, []).append(Wheel(asset, _sha256(path), url))
    return by_backend


def compat_wheels(repo: str) -> list[Wheel]:
    """The wheel assets of every compat release of *repo*."""
    # gh is a trusted CLI on PATH and `repo` is a controlled constant (the fork).
    out = subprocess.check_output(
        ["gh", "api", "--paginate", f"repos/{repo}/releases", "--jq", _COMPAT_JQ],
        text=True,
    )
    wheels: list[Wheel] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        asset = json.loads(line)
        digest = (asset.get("digest") or "").removeprefix("sha256:")
        if not digest:
            raise SystemExit(f"asset {asset['name']} has no sha256 digest from the API")
        wheels.append(Wheel(asset["name"], digest, asset["url"]))
    return wheels


def write_index(wheels: Sequence[Wheel], root: Path) -> None:
    """Write the PEP 503 index of *wheels* so that ``root`` is the index url."""
    with tempfile.TemporaryDirectory() as scratch:
        listing = Path(scratch) / "packages.json"
        lines = (json.dumps({"filename": w.filename, "hash": f"sha256={w.sha256}"}) for w in wheels)
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        built = Path(scratch) / "built"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dumb_pypi.main",
                "--package-list-json",
                str(listing),
                "--output-dir",
                str(built),
                "--packages-url",
                _PLACEHOLDER_URL,
                "--title",
                "lilbee wheels",
                "--no-per-release-json",
                "--no-generate-timestamp",
            ],
            check=True,
        )
        _point_links_at_assets(built / "simple", wheels)
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(built / "simple", root, dirs_exist_ok=True)


def _point_links_at_assets(simple: Path, wheels: Sequence[Wheel]) -> None:
    pages = sorted(simple.glob("*/index.html"))
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for wheel in wheels:
            text = text.replace(f"{_PLACEHOLDER_URL}/{wheel.filename}#", f"{wheel.url}#")
        page.write_text(text, encoding="utf-8")
    left = [page for page in pages if _PLACEHOLDER_URL in page.read_text(encoding="utf-8")]
    if not pages or left:
        raise SystemExit("dumb-pypi wrote a link that this tool did not point at a release asset")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    engines = commands.add_parser("engines")
    engines.add_argument("artifacts", type=Path)
    engines.add_argument("site", type=Path)
    engines.add_argument("--release-base-url", default=_DEFAULT_RELEASE_BASE_URL)
    compat = commands.add_parser("compat")
    compat.add_argument("site", type=Path)
    compat.add_argument("--repo", default="tobocop2/lancedb")
    args = parser.parse_args(argv)

    if args.command == "compat":
        wheels = compat_wheels(args.repo)
        if not wheels:
            print(f"no compat wheels found in {args.repo} releases", file=sys.stderr)
            return 1
        write_index(wheels, args.site / "compat")
        print(f"compat index: {len(wheels)} wheels from {args.repo}")
        return 0

    if not args.artifacts.is_dir():
        print(f"input directory does not exist: {args.artifacts}", file=sys.stderr)
        return 1
    by_backend = engine_wheels(args.artifacts, args.release_base_url)
    if not by_backend:
        print("no wheel artifacts found; nothing to index", file=sys.stderr)
        return 0
    for backend, wheels in by_backend.items():
        write_index(wheels, args.site / backend)
        print(f"backend={backend} wheels={len(wheels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
