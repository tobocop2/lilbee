"""Write the pack record of the engine wheels from the release assets API.

Usage: gen_engine_packs.py TAG [TAG ...] > OUTPUT

OUTPUT is src/lilbee/providers/fleet/engine_packs.json.

Each TAG is a release of the repository. Each engine version and cell keeps the wheel with
the highest build number; a later TAG wins for the same file name. Every
``lilbee_engine_*.whl`` asset of those releases becomes one row: ``file``, ``url``,
``sha256`` and ``size``. The tool reads the ``name``, ``digest`` and ``size`` fields of
the asset. The url is the release base, the tag and the name: a draft release gives an
``untagged-`` download url that is not the final one. Nothing is downloaded.
The loader that lilbee uses at run time checks the rows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from lilbee.providers.fleet.engine_record import (
    PACKAGES_URL,
    PackRecordError,
    build_manifests,
    wheel_cell,
)

REPOSITORY = "tobocop2/lilbee-engine-packs"
_ASSETS = ".[] | {tag_name, draft, assets: [.assets[] | {name, digest, size}]}"


class ReleaseError(Exception):
    """The releases cannot give a row for the tags that were named."""


def fetch_releases() -> list[dict[str, Any]]:
    """The tag and the assets of every release, drafts included, from the GitHub API."""
    done = subprocess.run(
        ["gh", "api", "--paginate", f"repos/{REPOSITORY}/releases", "--jq", _ASSETS],
        capture_output=True,
        text=True,
        check=True,
    )
    return [json.loads(line) for line in done.stdout.splitlines() if line]


def _row(tag: str, asset: dict[str, Any]) -> dict[str, Any]:
    digest = asset["digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ReleaseError(f"{tag}: the asset {asset['name']} has no sha256 digest.")
    return {
        "file": asset["name"],
        "url": f"{PACKAGES_URL}{tag}/{asset['name']}",
        "sha256": digest.removeprefix("sha256:"),
        "size": asset["size"],
    }


def build_record(releases: list[dict[str, Any]], tags: list[str]) -> list[dict[str, Any]]:
    """The rows for the engine wheels of the releases named by *tags*, sorted by file.

    Each engine version and cell keeps one wheel: the one with the highest build number.
    A wheel of a later tag replaces the wheel of the same name in an earlier tag. Two
    different wheels of one cell, version and build are refused.
    """
    by_tag = {release["tag_name"]: release for release in releases}
    rows: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    for tag in tags:
        if tag not in by_tag:
            raise ReleaseError(f"{tag}: no such release.")
        for asset in by_tag[tag]["assets"]:
            name = asset["name"]
            if not (name.startswith("lilbee_engine_") and name.endswith(".whl")):
                continue
            engine, version, key, build, _ = wheel_cell(f"{tag}: {name}", name)
            cell = (engine, version, key)
            if cell in rows:
                kept_build, kept = rows[cell]
                if kept_build > build:
                    continue
                if kept_build == build and kept["file"] != name:
                    raise ReleaseError(
                        f"{tag}: {name} and {kept['file']} are both build {build} of "
                        f"{engine} {version} for {key}."
                    )
            rows[cell] = (build, _row(tag, asset))
    record = sorted((row for _, row in rows.values()), key=lambda row: row["file"])
    build_manifests(record)
    return record


def draft_tags(releases: list[dict[str, Any]], tags: list[str]) -> list[str]:
    """The tags among *tags* whose release is still a draft."""
    return [
        release["tag_name"]
        for release in releases
        if release["draft"] and release["tag_name"] in tags
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args(argv)
    try:
        releases = fetch_releases()
        record = build_record(releases, args.tags)
    except (ReleaseError, PackRecordError) as error:
        print(error, file=sys.stderr)
        return 1
    for tag in draft_tags(releases, args.tags):
        print(
            f"warning: {tag} is a draft; its urls answer 404 until it is published.",
            file=sys.stderr,
        )
    sys.stdout.write(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
