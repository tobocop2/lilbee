"""Write the pack record of the engine wheels from the release assets API.

Usage: gen_engine_packs.py TAG [TAG ...] > src/lilbee/providers/fleet/engine_packs.json

Each TAG is a release of the repository. A later TAG wins over an earlier one for the
same file name. Every ``lilbee_engine_*.whl`` asset of those releases becomes one row:
``file``, ``url``, ``sha256`` and ``size``. The tool reads the ``name``, ``digest`` and
``size`` fields of the asset. The url is the release base, the tag
and the name: a draft release gives an ``untagged-`` download url that is not the final one.
Nothing is downloaded. The rows are checked by the loader that lilbee uses at run time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from lilbee.providers.fleet.engine_record import PACKAGES_URL, PackRecordError, build_manifests

REPOSITORY = "tobocop2/lilbee"
_ASSETS = ".[] | {tag_name, assets: [.assets[] | {name, digest, size}]}"


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

    A wheel of a later tag replaces the wheel of the same name in an earlier tag.
    """
    by_tag = {release["tag_name"]: release for release in releases}
    rows: dict[str, dict[str, Any]] = {}
    for tag in tags:
        if tag not in by_tag:
            raise ReleaseError(f"{tag}: no such release.")
        for asset in by_tag[tag]["assets"]:
            if asset["name"].startswith("lilbee_engine_") and asset["name"].endswith(".whl"):
                rows[asset["name"]] = _row(tag, asset)
    record = sorted(rows.values(), key=lambda row: row["file"])
    build_manifests(record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args(argv)
    try:
        record = build_record(fetch_releases(), args.tags)
    except (ReleaseError, PackRecordError) as error:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
