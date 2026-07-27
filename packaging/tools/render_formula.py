#!/usr/bin/env python3
"""Update version + per-platform sha256 in the Homebrew formula in place."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
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


Renderer = Callable[[str, argparse.Namespace], str]


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


def _render_rocm(content: str, args: argparse.Namespace) -> str:
    content = _replace_version(content, args.version)
    return _replace_sha_for_asset(
        content, "lilbee-linux-x86_64-rocm", args.sha_linux_rocm, required=True
    )


def _render_compat(content: str, args: argparse.Namespace) -> str:
    content = _replace_version(content, args.version)
    return _replace_sha_for_asset(
        content, "lilbee-compat-linux-x86_64", args.sha_linux_compat, required=True
    )


# Each GPU/CPU flavor of the formula: the flag that selects it, the renderer, and
# the one digest that flavor needs. A new flavor is an entry here, not another
# branch in main.
_FLAVORS = (
    ("cuda", _render_cuda, "sha_linux_cu125"),
    ("rocm", _render_rocm, "sha_linux_rocm"),
    ("compat", _render_compat, "sha_linux_compat"),
)


def _selected_flavor(args: argparse.Namespace) -> tuple[str, Renderer, str] | None:
    """The flavor *args* asks for, or None for the default formula."""
    return next((f for f in _FLAVORS if getattr(args, f[0])), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("formula", type=Path)
    parser.add_argument("--version", required=True)
    mode = parser.add_mutually_exclusive_group()
    for name, _, _ in _FLAVORS:
        mode.add_argument(
            f"--{name}",
            action="store_true",
            help=f"Render the lilbee-{name} formula instead of the default lilbee formula.",
        )
    parser.add_argument("--sha-macos-arm64")
    parser.add_argument("--sha-linux-x86_64")
    parser.add_argument("--sha-macos-x86_64", default=None)
    for _, _, digest in _FLAVORS:
        parser.add_argument(f"--{digest.replace('_', '-')}")
    args = parser.parse_args()

    flavor = _selected_flavor(args)
    if flavor is None:
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
        render = _render_default
    else:
        name, render, digest = flavor
        if not getattr(args, digest):
            parser.error(f"--{name} requires --{digest.replace('_', '-')}")

    args.formula.write_text(render(args.formula.read_text(), args))


if __name__ == "__main__":
    main()
