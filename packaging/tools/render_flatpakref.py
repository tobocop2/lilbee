#!/usr/bin/env python3
"""Render a channel's .flatpakref installer from its metainfo and the repo's signing key."""

from __future__ import annotations

import argparse
import base64
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# build-update-repo publishes app/<id>/<arch>/master; no manifest sets a branch.
_BRANCH = "master"
_RUNTIME_REPO = "https://dl.flathub.org/repo/flathub.flatpakrepo"


def _text(component: ET.Element, path: str, what: str) -> str:
    found = component.find(path)
    if found is None or not found.text:
        sys.exit(f"metainfo has no {what}")
    return found.text.strip()


def _render(metainfo: Path, repo_url: str, gpg_key: bytes, remote_name: str) -> str:
    component = ET.parse(metainfo).getroot()  # noqa: S314  repo-committed metainfo
    fields = {
        "Name": _text(component, "id", "id"),
        "Branch": _BRANCH,
        "Url": repo_url,
        "Title": _text(component, "name", "name"),
        "Comment": _text(component, "summary", "summary"),
        "Homepage": _text(component, 'url[@type="homepage"]', "homepage url"),
        "IsRuntime": "false",
        "SuggestRemoteName": remote_name,
        "RuntimeRepo": _RUNTIME_REPO,
        "GPGKey": base64.b64encode(gpg_key).decode(),
    }
    body = "\n".join(f"{key}={value}" for key, value in fields.items())
    return f"[Flatpak Ref]\n{body}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metainfo", type=Path)
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--gpg-key-file", required=True, type=Path, help="gpg --export output.")
    parser.add_argument("--remote-name", required=True)
    args = parser.parse_args()

    if not args.gpg_key_file.is_file():
        sys.exit(f"no such gpg key file: {args.gpg_key_file}")
    gpg_key = args.gpg_key_file.read_bytes()
    if not gpg_key:
        sys.exit(f"empty gpg key file: {args.gpg_key_file}")
    sys.stdout.write(_render(args.metainfo, args.repo_url, gpg_key, args.remote_name))


if __name__ == "__main__":
    main()
