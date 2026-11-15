#!/usr/bin/env bash
# Install the pinned cargo-about release into a directory and check its sha256.
#
# Usage: install_cargo_about.sh <dest-dir>
#
# cargo-about lists the licences of the crates that a Rust binary links. The release archive for
# the host (Linux x86_64 or macOS arm64) is fetched from GitHub and checked against the digest below.
set -euo pipefail

version=0.8.4
test "$#" -eq 1 || { echo "usage: install_cargo_about.sh <dest-dir>" >&2; exit 2; }
dest="$1"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    triple=x86_64-unknown-linux-musl
    want=c7381aa0cdc41fc0ee662cec8daa260da7817ad8ddea04cd4ddad425460adf14
    ;;
  Darwin-arm64)
    triple=aarch64-apple-darwin
    want=d5255ead3ac861a11c785bf19d4f70f16e59ac8b9519c312da61213d3d57351d
    ;;
  *)
    echo "install_cargo_about: no pinned release for $(uname -s) $(uname -m)" >&2
    exit 1
    ;;
esac

name="cargo-about-${version}-${triple}"
work="$(mktemp -d)"
curl --retry 5 --retry-all-errors -fsSL -o "${work}/${name}.tar.gz" \
  "https://github.com/EmbarkStudios/cargo-about/releases/download/${version}/${name}.tar.gz"
if command -v sha256sum >/dev/null; then
  got="$(sha256sum "${work}/${name}.tar.gz" | cut -d' ' -f1)"
else
  got="$(shasum -a 256 "${work}/${name}.tar.gz" | cut -d' ' -f1)"
fi
test "${got}" = "${want}" || { echo "install_cargo_about: digest ${got}, wanted ${want}" >&2; exit 1; }
mkdir -p "${dest}"
tar -xzf "${work}/${name}.tar.gz" -C "${work}"
cp "${work}/${name}/cargo-about" "${dest}/cargo-about"
chmod +x "${dest}/cargo-about"
"${dest}/cargo-about" --version
