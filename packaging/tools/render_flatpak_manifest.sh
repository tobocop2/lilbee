#!/usr/bin/env bash
# Update the extra-data url/sha256/size in the Flatpak manifest in place.
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "usage: $0 <manifest-path> <version> <sha256> <size> [asset-name]" >&2
    exit 2
fi

manifest="$1"
version="$2"
sha_linux="$3"
size_linux="$4"
# Defaults to the stock asset so existing callers are unchanged; the compat
# channel passes lilbee-compat-linux-x86_64.
asset="${5:-lilbee-linux-x86_64}"

sed -i.bak \
    -e "s|^\( *url: \).*|\1https://github.com/tobocop2/lilbee/releases/download/v${version}/${asset}|" \
    -e "s|^\( *sha256: \).*|\1'${sha_linux}'|" \
    -e "s|^\( *size: \).*|\1${size_linux}|" \
    "$manifest"
rm -f "${manifest}.bak"
