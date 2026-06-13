#!/usr/bin/env bash
# Update the version in snapcraft.yaml in place.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <snapcraft-yaml-path> <version>" >&2
    exit 2
fi

snapcraft_yaml="$1"
version="$2"

sed -i.bak \
    -e "s|^version: .*|version: \"${version}\"|" \
    "$snapcraft_yaml"
rm -f "${snapcraft_yaml}.bak"
