#!/usr/bin/env bash
# Write a copy of uv.lock with lilbee's own version line removed, so a cache key
# hashed from the copy tracks the dependency set and not the release counter.
#
#   bash scripts/strip_own_version.sh uv.lock uv-deps.lock
#
# scripts/release.sh rewrites that one line on every release commit.

set -euo pipefail

src="${1:?path to uv.lock is required}"
dst="${2:?output path is required}"

# Only the version inside the `name = "lilbee"` package block. Every
# dependency's version stays in the output. The trailing-space class carries the
# carriage return a Windows checkout adds.
# The same pass counts what it drops. Comparing line counts of the two files
# instead would miscount a lock that ends without a newline.
if ! awk '
  /^\[\[package\]\][[:space:]]*$/ { own = 0 }
  /^name = "lilbee"[[:space:]]*$/ { own = 1 }
  own && /^version = "/ { dropped++; next }
  { print }
  END { if (dropped != 1) exit 1 }
' "${src}" > "${dst}"; then
  echo "strip_own_version: ${src} does not give up exactly one version line." >&2
  echo "strip_own_version: the lock must carry one lilbee package entry with a version." >&2
  exit 1
fi
