#!/usr/bin/env bash
# Publish SHA256SUMS on a release: one sha256sum line per standalone binary
# and ROCm gfx manifest, taken from the digest GitHub records for each asset.
# site/install.sh and site/install.ps1 refuse any download without a line here.
#
#   bash scripts/write_release_sums.sh v0.6.90b693 [owner/repo]
set -euo pipefail

tag="${1:-}"
repo="${2:-${GITHUB_REPOSITORY:-tobocop2/lilbee}}"
[ -n "${tag}" ] || { echo "usage: $0 <tag> [owner/repo]" >&2; exit 1; }

# Same names the installers pick from; the wheels, snaps and flatpakrefs are not installer assets.
binary_name='^lilbee(-compat)?-(macos|linux|windows)-[a-z0-9_]+(-(cu12[0-9]|rocm))?(\.exe|\.gfx\.txt)?$'

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
assets="${work}/assets.json"
sums="${work}/SHA256SUMS"

gh release view "${tag}" --repo "${repo}" --json assets --jq '.assets' > "${assets}"

missing="$(jq -r --arg re "${binary_name}" \
  '.[] | select(.name | test($re)) | select((.digest // "") | startswith("sha256:") | not) | .name' "${assets}")"
if [ -n "${missing}" ]; then
  echo "write_release_sums: ${tag} has no sha256 digest for: ${missing}" >&2
  exit 1
fi

jq -r --arg re "${binary_name}" \
  '[.[] | select(.name | test($re))] | sort_by(.name) | .[] | "\(.digest | ltrimstr("sha256:"))  \(.name)"' \
  "${assets}" > "${sums}"

if ! grep -q '[^[:space:]]' "${sums}"; then
  echo "write_release_sums: ${tag} has no binary assets to list" >&2
  exit 1
fi

gh release upload "${tag}" "${sums}" --repo "${repo}" --clobber
echo "write_release_sums: ${tag} lists $(wc -l < "${sums}" | tr -d ' ') assets in SHA256SUMS"
