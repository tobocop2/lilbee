#!/usr/bin/env python3
"""Update version, license and per-platform sha256 in the Homebrew formula in place."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path


def _replace_sha_for_asset(content: str, asset_name: str, new_sha: str) -> str:
    pattern = re.compile(
        r'(url "[^"]*' + re.escape(asset_name) + r'"\s*\n\s*sha256 ")[0-9a-f]{64}(")',
    )
    new_content, count = pattern.subn(rf"\g<1>{new_sha}\g<2>", content)
    if count == 0:
        raise SystemExit(f"expected sha256 match for {asset_name}, found none")
    if count > 1:
        raise SystemExit(f"expected 1 sha256 match for {asset_name}, found {count}")
    return new_content


# The tap is the published artifact and nothing else rewrites it, so a formula already
# there keeps whatever it says: the seeds in packaging/homebrew are only used when a
# formula is absent. Every flavor carries the project license, so the renderer owns it.
LICENSE = "MIT"


def _replace_license(content: str) -> str:
    new_content, count = re.subn(r'  license "[^"]*"', f'  license "{LICENSE}"', content)
    if count != 1:
        raise SystemExit(f"expected exactly one license line, found {count}")
    return new_content


Renderer = Callable[[str, argparse.Namespace], str]


def _replace_version(content: str, version: str) -> str:
    new_content, count = re.subn(r'  version "[^"]*"', f'  version "{version}"', content)
    if count != 1:
        sys.exit("expected exactly one version line")
    return new_content


# The default formula's platforms. Rendering is all-or-nothing: a digest with no
# matching url block stops the publish. Skipping it instead is how the tap went on
# publishing a formula with no Intel macOS url long after the asset existed.
_DEFAULT_ASSETS = (
    ("lilbee-macos-arm64", "--sha-macos-arm64"),
    ("lilbee-macos-x86_64", "--sha-macos-x86_64"),
    ("lilbee-linux-x86_64", "--sha-linux-x86_64"),
)


def _dest(flag: str) -> str:
    """Where *flag*'s value lands on the namespace. Also passed to add_argument, so
    registration and lookup cannot disagree."""
    return flag[2:].replace("-", "_")


def _render_default(content: str, args: argparse.Namespace) -> str:
    content = _replace_version(content, args.version)
    for asset, flag in _DEFAULT_ASSETS:
        content = _replace_sha_for_asset(content, asset, getattr(args, _dest(flag)))
    return content


def _render_cuda(content: str, args: argparse.Namespace) -> str:
    content = _replace_version(content, args.version)
    return _replace_sha_for_asset(content, "lilbee-linux-x86_64-cu125", args.sha_linux_cu125)


def _render_rocm(content: str, args: argparse.Namespace) -> str:
    content = _replace_version(content, args.version)
    return _replace_sha_for_asset(content, "lilbee-linux-x86_64-rocm", args.sha_linux_rocm)


def _render_compat(content: str, args: argparse.Namespace) -> str:
    content = _replace_version(content, args.version)
    return _replace_sha_for_asset(content, "lilbee-compat-linux-x86_64", args.sha_linux_compat)


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
    for _, flag in _DEFAULT_ASSETS:
        parser.add_argument(flag, dest=_dest(flag))
    for _, _, digest in _FLAVORS:
        parser.add_argument(f"--{digest.replace('_', '-')}")
    args = parser.parse_args()

    flavor = _selected_flavor(args)
    if flavor is None:
        missing = [flag for _, flag in _DEFAULT_ASSETS if not getattr(args, _dest(flag))]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
        render = _render_default
    else:
        name, render, digest = flavor
        if not getattr(args, digest):
            parser.error(f"--{name} requires --{digest.replace('_', '-')}")

    args.formula.write_text(_replace_license(render(args.formula.read_text(), args)))


if __name__ == "__main__":
    main()
