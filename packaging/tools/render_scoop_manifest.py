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


def _render_default(content: str, args: argparse.Namespace) -> str:
    # The version field, the literal version in the CPU download URL (the CUDA
    # URLs are built from $version at install time), the Scoop-verified CPU hash,
    # and each CUDA hash checked by post_install.
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
    for flavor, sha in (("cu125", args.sha_windows_cu125), ("cu124", args.sha_windows_cu124)):
        content = _replace_once(
            content,
            re.compile(rf"('{flavor}' = ')[0-9a-f]{{64}}(')"),
            rf"\g<1>{sha}\g<2>",
            f"{flavor} hash",
        )
    return content


def _render_cuda(content: str, args: argparse.Namespace) -> str:
    # The CUDA manifest is a plain single-arch package whose Scoop-verified
    # download is the cu125 exe: version, the version in its download URL, and
    # the hash. post_install only warns on old drivers; it downloads nothing.
    content = _replace_once(
        content, re.compile(r'"version": "[^"]*"'), f'"version": "{args.version}"', "version line"
    )
    content = _replace_once(
        content,
        re.compile(r"(releases/download/v)[^/]+(/lilbee-windows-x86_64-cu125\.exe)"),
        rf"\g<1>{args.version}\g<2>",
        "cu125 download URL",
    )
    return _replace_once(
        content,
        re.compile(r'("hash": ")[0-9a-f]{64}(")'),
        rf"\g<1>{args.sha_windows_cu125}\g<2>",
        "cu125 hash",
    )


def _render_compat(content: str, args: argparse.Namespace) -> str:
    # The pre-Haswell manifest is a plain single-arch package: version, the
    # version in its download URL, and the Scoop-verified hash. No cu125 path.
    content = _replace_once(
        content, re.compile(r'"version": "[^"]*"'), f'"version": "{args.version}"', "version line"
    )
    content = _replace_once(
        content,
        re.compile(r"(releases/download/v)[^/]+(/lilbee-compat-windows-x86_64\.exe)"),
        rf"\g<1>{args.version}\g<2>",
        "compat download URL",
    )
    return _replace_once(
        content,
        re.compile(r'("hash": ")[0-9a-f]{64}(")'),
        rf"\g<1>{args.sha_windows_compat}\g<2>",
        "compat hash",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--version", required=True)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument(
        "--compat",
        action="store_true",
        help="Render the lilbee-compat (pre-Haswell CPU) manifest.",
    )
    variant.add_argument(
        "--cuda",
        action="store_true",
        help="Render the lilbee-cuda (cu125-only) manifest.",
    )
    parser.add_argument("--sha-windows")
    parser.add_argument("--sha-windows-cu125")
    parser.add_argument("--sha-windows-cu124")
    parser.add_argument("--sha-windows-compat")
    args = parser.parse_args()

    if args.compat:
        if not args.sha_windows_compat:
            parser.error("--compat requires --sha-windows-compat")
    elif args.cuda:
        if not args.sha_windows_cu125:
            parser.error("--cuda requires --sha-windows-cu125")
    else:
        missing = [
            flag
            for flag, value in (
                ("--sha-windows", args.sha_windows),
                ("--sha-windows-cu125", args.sha_windows_cu125),
                ("--sha-windows-cu124", args.sha_windows_cu124),
            )
            if not value
        ]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")

    content = args.manifest.read_text()
    if args.compat:
        content = _render_compat(content, args)
    elif args.cuda:
        content = _render_cuda(content, args)
    else:
        content = _render_default(content, args)
    args.manifest.write_text(content)


if __name__ == "__main__":
    main()
