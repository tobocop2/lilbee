#!/usr/bin/env python3
"""Update version + per-variant sha256 in the Scoop manifest in place."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _replace_once(content: str, pattern: re.Pattern[str], repl: str, what: str) -> str:
    new_content, count = pattern.subn(repl, content)
    if count != 1:
        sys.exit(f"expected exactly one {what}, found {count}")
    return new_content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha-windows", required=True)
    parser.add_argument("--sha-windows-cu125", required=True)
    args = parser.parse_args()

    content = args.manifest.read_text()
    # The version field, the literal version in the CPU download URL (the cu125
    # URL is built from $version at install time), the Scoop-verified CPU hash,
    # and the cu125 hash checked by post_install.
    content = _replace_once(
        content, re.compile(r'"version": "[^"]*"'), f'"version": "{args.version}"', "version line"
    )
    content = _replace_once(
        content,
        re.compile(r"(releases/download/v)[^/]+(/lilbee-windows-x86_64\.exe)"),
        rf"\g<1>{args.version}\g<2>",
        "CPU download URL",
    )
    content = _replace_once(
        content,
        re.compile(r'("hash": ")[0-9a-f]{64}(")'),
        rf"\g<1>{args.sha_windows}\g<2>",
        "CPU hash",
    )
    content = _replace_once(
        content,
        re.compile(r"(\$cudaHash = ')[0-9a-f]{64}(')"),
        rf"\g<1>{args.sha_windows_cu125}\g<2>",
        "cu125 hash",
    )
    args.manifest.write_text(content)


if __name__ == "__main__":
    main()
