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
    new_content, count = pattern.subn(rf"\g<1>{new_sha}\g<2>", content)
    if count == 0:
        if required:
            raise SystemExit(f"expected sha256 match for {asset_name}, found none")
        print(f"note: {asset_name} not in formula yet; skipping", file=sys.stderr)
        return content
    if count > 1:
        raise SystemExit(f"expected 1 sha256 match for {asset_name}, found {count}")
    return new_content


def _replace_version(content: str, version: str) -> str:
    new_content, count = re.subn(r'  version "[^"]*"', f'  version "{version}"', content)
    if count != 1:
        sys.exit("expected exactly one version line")
    return new_content


def _render_default(content: str, args: argparse.Namespace) -> str:
    content = _replace_version(content, args.version)
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
    return content


def _render_cuda(content: str, args: argparse.Namespace) -> str:
    content = _replace_version(content, args.version)
    return _replace_sha_for_asset(
        content, "lilbee-linux-x86_64-cu125", args.sha_linux_cu125, required=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("formula", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Render the lilbee-cuda formula instead of the default lilbee formula.",
    )
    parser.add_argument("--sha-macos-arm64")
    parser.add_argument("--sha-linux-x86_64")
    parser.add_argument("--sha-macos-x86_64", default=None)
    parser.add_argument("--sha-linux-cu125")
    args = parser.parse_args()

    if args.cuda:
        if not args.sha_linux_cu125:
            parser.error("--cuda requires --sha-linux-cu125")
    else:
        missing = [
            flag
            for flag, value in (
                ("--sha-macos-arm64", args.sha_macos_arm64),
                ("--sha-linux-x86_64", args.sha_linux_x86_64),
            )
            if not value
        ]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")

    content = args.formula.read_text()
    content = _render_cuda(content, args) if args.cuda else _render_default(content, args)
    args.formula.write_text(content)


if __name__ == "__main__":
    main()
