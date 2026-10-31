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


# The single-arch manifests: a plain package whose Scoop-verified download is one
# exe. Each renders the version, the version in that exe's download URL, and the
# hash. The key is the mode flag, the values are the asset, the digest flag's
# namespace name and the label error messages use.
_SINGLE_ASSET_MODES = {
    "cuda": ("lilbee-windows-x86_64-cu125.exe", "sha_windows_cu125", "cu125"),
    "compat": ("lilbee-compat-windows-x86_64.exe", "sha_windows_compat", "compat"),
    "compat-cuda": (
        "lilbee-compat-windows-x86_64-cu124.exe",
        "sha_windows_compat_cu124",
        "compat-cuda",
    ),
}


def _dest(name: str) -> str:
    return name.replace("-", "_")


def _render_single_asset(content: str, args: argparse.Namespace, mode: str) -> str:
    asset, digest, label = _SINGLE_ASSET_MODES[mode]
    content = _replace_once(
        content, re.compile(r'"version": "[^"]*"'), f'"version": "{args.version}"', "version line"
    )
    content = _replace_once(
        content,
        re.compile(r"(releases/download/v)[^/]+(/" + re.escape(asset) + ")"),
        rf"\g<1>{args.version}\g<2>",
        f"{label} download URL",
    )
    return _replace_once(
        content,
        re.compile(r'("hash": ")[0-9a-f]{64}(")'),
        rf"\g<1>{getattr(args, digest)}\g<2>",
        f"{label} hash",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--version", required=True)
    variant = parser.add_mutually_exclusive_group()
    for mode in _SINGLE_ASSET_MODES:
        variant.add_argument(
            f"--{mode}",
            action="store_true",
            help=f"Render the lilbee-{mode} manifest.",
        )
    parser.add_argument("--sha-windows")
    parser.add_argument("--sha-windows-cu125")
    parser.add_argument("--sha-windows-cu124")
    for _, digest, _ in _SINGLE_ASSET_MODES.values():
        if digest != "sha_windows_cu125":
            parser.add_argument(f"--{digest.replace('_', '-')}")
    args = parser.parse_args()

    mode = next((m for m in _SINGLE_ASSET_MODES if getattr(args, _dest(m))), None)
    if mode is not None:
        digest = _SINGLE_ASSET_MODES[mode][1]
        if not getattr(args, digest):
            parser.error(f"--{mode} requires --{digest.replace('_', '-')}")
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
    if mode is None:
        content = _render_default(content, args)
    else:
        content = _render_single_asset(content, args, mode)
    args.manifest.write_text(content)


if __name__ == "__main__":
    main()
