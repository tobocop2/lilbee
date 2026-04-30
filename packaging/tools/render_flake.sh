#!/usr/bin/env bash
# Update version + per-platform sha256 in flake.nix in place.
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <flake-path> <version> <sha-linux-x86_64> <sha-macos-arm64>" >&2
    exit 2
fi

flake="$1"
version="$2"
sha_linux="$3"
sha_darwin="$4"

# Each lilbee flake field that the release pipeline updates is tagged with
# a "# RENDERED:<KEY>" sentinel comment, e.g.
#     version = "0.6.66b456"; # RENDERED:VERSION
#     x86_64-linux = "abc..."; # RENDERED:SHA_LINUX
# Capture group 1 is everything through the opening quote of the value,
# group 2 is the closing quote and the sentinel comment. The quoted value
# between them gets replaced. The identifier portion is permissive on
# purpose: it matches both top-level fields (version) and attribute-set
# entries (x86_64-linux, aarch64-darwin) without per-key regexes.
readonly SENTINEL_PREFIX='[[:space:]]*[A-Za-z0-9_-]+ = '
readonly SENTINEL_SUFFIX='; # RENDERED:'

stamp() {
    local key="$1" value="$2"
    sed -i.bak -E \
        "s|^(${SENTINEL_PREFIX})\"[^\"]*\"(${SENTINEL_SUFFIX}${key})|\1\"${value}\"\2|" \
        "$flake"
}

stamp VERSION "$version"
stamp SHA_LINUX "$sha_linux"
stamp SHA_DARWIN "$sha_darwin"
rm -f "${flake}.bak"
