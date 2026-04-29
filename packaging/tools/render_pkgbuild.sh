#!/usr/bin/env bash
# Update version + sha256 in the AUR PKGBUILD in place.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <pkgbuild-path> <version> <sha256-linux-x86_64>" >&2
    exit 2
fi

pkgbuild="$1"
version="$2"
sha_linux="$3"

sed -i.bak \
    -e "s|^pkgver=.*|pkgver=${version}|" \
    -e "s|^pkgrel=.*|pkgrel=1|" \
    -e "s|^sha256sums_x86_64=.*|sha256sums_x86_64=('${sha_linux}')|" \
    "$pkgbuild"
rm -f "${pkgbuild}.bak"
