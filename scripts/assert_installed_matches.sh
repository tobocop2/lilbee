#!/usr/bin/env bash
# Fail unless the file the installer wrote is byte for byte the release asset the job smoked.
#
#   bash scripts/assert_installed_matches.sh <installed file> <smoked asset>
set -euo pipefail

installed="${1:-}"
smoked="${2:-}"
[ -f "${installed}" ] && [ -f "${smoked}" ] || { echo "usage: $0 <installed file> <smoked asset>" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

installed_sum="$(sha256_of "${installed}")"
smoked_sum="$(sha256_of "${smoked}")"
if [ "${installed_sum}" != "${smoked_sum}" ]; then
  echo "assert_installed_matches: ${installed} is ${installed_sum}, the smoked asset ${smoked} is ${smoked_sum}" >&2
  exit 1
fi
echo "assert_installed_matches: ${installed} equals ${smoked} (${installed_sum})"
