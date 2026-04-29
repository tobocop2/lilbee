#!/usr/bin/env python3
"""Update version + per-platform sha256 in the Homebrew formula in place."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _replace_sha_for_asset(content: str, asset_name: str, new_sha: str, *, required: bool) -> str:
    pattern = re.compile(
        r'(url "[^"]*' + re.escape(asset_name) + r'"\s*\n\s*sha256 ")[0-9a-f]{64}(")',
    )
    new_content, count = pattern.subn(rf'\g<1>{new_sha}\g<2>', content)
    if count == 0:
        if required:
            raise SystemExit(f"expected sha256 match for {asset_name}, found none")
        print(f"note: {asset_name} not in formula yet; skipping", file=sys.stderr)
        return content
    if count > 1:
        raise SystemExit(f"expected 1 sha256 match for {asset_name}, found {count}")
    return new_content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("formula", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha-macos-arm64", required=True)
    parser.add_argument("--sha-linux-x86_64", required=True)
    parser.add_argument("--sha-macos-x86_64", default=None)
    args = parser.parse_args()

    content = args.formula.read_text()

    content, count = re.subn(r'  version "[^"]*"', f'  version "{args.version}"', content)
    if count != 1:
        sys.exit("expected exactly one version line")

    content = _replace_sha_for_asset(
        content, "lilbee-macos-arm64", args.sha_macos_arm64, required=True
    )
    content = _replace_sha_for_asset(
        content, "lilbee-linux-x86_64", args.sha_linux_x86_64, required=True
    )
    if args.sha_macos_x86_64:
        content = _replace_sha_for_asset(
            content, "lilbee-macos-x86_64", args.sha_macos_x86_64, required=False
        )

    args.formula.write_text(content)


if __name__ == "__main__":
    main()
