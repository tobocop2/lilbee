#!/usr/bin/env bash
# Print SOURCE_DATE_EPOCH=<seconds>, the commit time of a git source tree.
#
# Usage: source_date_epoch.sh <git-dir>
#
# The wheel build stamps every entry with this time instead of the time of the file on disk, so
# one source commit gives one wheel layout. The caller exports the line or appends it to GITHUB_ENV.
# repack_mistralrs.py prints the same line from the time of the tool in an upstream archive.
set -euo pipefail

test "$#" -eq 1 || {
  echo "usage: source_date_epoch.sh <git-dir>" >&2
  exit 2
}
epoch="$(git -C "$1" log -1 --format=%ct)"
case "${epoch}" in
  '' | *[!0-9]*)
    echo "no commit time in $1" >&2
    exit 1
    ;;
esac
printf 'SOURCE_DATE_EPOCH=%s\n' "${epoch}"
