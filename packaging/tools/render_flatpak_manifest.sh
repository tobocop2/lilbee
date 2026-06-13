#!/usr/bin/env bash
# Update the extra-data url/sha256/size in the Flatpak manifest in place.
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <manifest-path> <version> <sha256-linux-x86_64> <size-linux-x86_64>" >&2
    exit 2
fi

manifest="$1"
version="$2"
sha_linux="$3"
size_linux="$4"

sed -i.bak \
    -e "s|^\( *url: \).*|\1https://github.com/tobocop2/lilbee/releases/download/v${version}/lilbee-linux-x86_64|" \
    -e "s|^\( *sha256: \).*|\1'${sha_linux}'|" \
    -e "s|^\( *size: \).*|\1${size_linux}|" \
    "$manifest"
rm -f "${manifest}.bak"
