#!/usr/bin/env bash
# Map an engine cache key to its mirror asset name. The key is content-addressed
# (every engine source is pinned in engine-versions.env), so the asset name is
# too: same name means same bytes. Container images appear in the key and carry
# slashes, dots and colons, so anything a filename or URL would object to
# collapses to an underscore.
#
# Single source of truth on purpose: the download side (bundle-llama-server) and
# the upload side (publish-engine-asset) must agree exactly or the mirror silently
# never hits.
set -euo pipefail

key="${1:?usage: engine_asset_name.sh <engine-cache-key>}"
printf '%s.tar.gz\n' "$(printf '%s' "${key}" | tr -c 'A-Za-z0-9._-' '_')"
