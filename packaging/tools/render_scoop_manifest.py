#!/usr/bin/env python3
"""Update version + per-variant sha256 in the Scoop manifest in place."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _replace_version(content: str, version: str) -> str:
    new_content, count = re.subn(r'"version": "[^"]*"', f'"version": "{version}"', content)
    if count != 1:
        sys.exit("expected exactly one version line")
    return new_content


def _replace_hash(content: str, var_name: str, new_sha: str) -> str:
    pattern = re.compile(r"(\$" + re.escape(var_name) + r" = ')[0-9a-f]{64}(')")
    new_content, count = pattern.subn(rf"\g<1>{new_sha}\g<2>", content)
    if count != 1:
        sys.exit(f"expected exactly one {var_name} line, found {count}")
    return new_content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha-windows", required=True)
    parser.add_argument("--sha-windows-cu125", required=True)
    args = parser.parse_args()

    content = args.manifest.read_text()
    content = _replace_version(content, args.version)
    content = _replace_hash(content, "cpuHash", args.sha_windows)
    content = _replace_hash(content, "cudaHash", args.sha_windows_cu125)
    args.manifest.write_text(content)


if __name__ == "__main__":
    main()
