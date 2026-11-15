#!/usr/bin/env bash
# Write the third-party licences of a Rust engine binary, and the list of their SPDX names.
#
# Usage: rust_licences.sh <source-dir> <cargo-features> <target-triple> <out-dir> <terms-file>
#
# cargo-about reads the locked dependency graph of router/Cargo.toml under <source-dir> for the
# target and the features of the build, without the dev and build dependencies. A crate whose
# licence is not in cargo_about.toml stops the run. <out-dir> gets one text with every licence text
# and the crates that use it. <terms-file> gets one SPDX name for each line, sorted. The cargo-about
# report in JSON goes beside it, with the extension .json in place of .txt.
# cargo-about, jq and cargo must be on PATH.
set -euo pipefail

test "$#" -eq 5 || {
  echo "usage: rust_licences.sh <source-dir> <cargo-features> <target-triple> <out-dir> <terms-file>" >&2
  exit 2
}
src="$1" features="$2" target="$3" out="$4" terms="$5"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${out}" "$(dirname "${terms}")"
# The cargo-about runs below start in <source-dir>, so every output path must be absolute.
out="$(cd "${out}" && pwd)"
terms="$(cd "$(dirname "${terms}")" && pwd)/$(basename "${terms}")"

about() {
  (cd "${src}" && cargo about generate --locked --manifest-path router/Cargo.toml \
    --no-default-features --features "${features}" --target "${target}" \
    -c "${here}/cargo_about.toml" "$@")
}
about -o "${out}/LICENSES.txt" "${here}/cargo_about.hbs"
json="${terms%.txt}.json"
about --format json -o "${json}"
jq -r '.licenses[].id' "${json}" | sort -u > "${terms}"
test -s "${out}/LICENSES.txt" || { echo "rust_licences: empty licence text" >&2; exit 1; }
test -s "${terms}" || { echo "rust_licences: no licence names" >&2; exit 1; }
cat "${terms}"
