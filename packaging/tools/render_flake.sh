#!/usr/bin/env bash
# Update version + per-platform sha256 in flake.nix in place.
#
# Mirrors render_pkgbuild.sh: stamps the values written by the release
# workflow into the sentinel-tagged let-bindings in flake.nix. Sentinels
# are RENDERED:VERSION / RENDERED:SHA_LINUX / RENDERED:SHA_DARWIN so the
# regex never matches stray identifiers (e.g. meta.platforms entries).
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <flake-path> <version> <sha-linux-x86_64> <sha-macos-arm64>" >&2
    exit 2
fi

flake="$1"
version="$2"
sha_linux="$3"
sha_darwin="$4"

# Use [[:space:]] for BSD-sed (macOS) compatibility, even though CI runs on
# ubuntu (GNU sed) — keeps local manual reruns portable.
sed -i.bak \
    -e "s|^\([[:space:]]*version = \)\"[^\"]*\"\(; # RENDERED:VERSION\)|\1\"${version}\"\2|" \
    -e "s|^\([[:space:]]*x86_64-linux = \)\"[^\"]*\"\(; # RENDERED:SHA_LINUX\)|\1\"${sha_linux}\"\2|" \
    -e "s|^\([[:space:]]*aarch64-darwin = \)\"[^\"]*\"\(; # RENDERED:SHA_DARWIN\)|\1\"${sha_darwin}\"\2|" \
    "$flake"
rm -f "${flake}.bak"
