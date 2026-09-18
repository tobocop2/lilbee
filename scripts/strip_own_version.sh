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
awk '
  /^\[\[package\]\][[:space:]]*$/ { own = 0 }
  /^name = "lilbee"[[:space:]]*$/ { own = 1 }
  own && /^version = "/ { next }
  { print }
' "${src}" > "${dst}"

dropped=$(( $(wc -l < "${src}") - $(wc -l < "${dst}") ))
if [ "${dropped}" -ne 1 ]; then
  echo "strip_own_version: dropped ${dropped} lines from ${src}, expected exactly 1." >&2
  echo "strip_own_version: the lock must carry one lilbee package entry with a version." >&2
  exit 1
fi
