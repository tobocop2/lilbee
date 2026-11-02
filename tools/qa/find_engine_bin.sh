#!/usr/bin/env bash
# Print the one lilbee_engine/bin directory under a build output directory.
#
# Nuitka names its standalone tree after the entry module, so the tree is found
# by shape. Fail when there is no match, and when there is more than one: a
# second copy (an onefile intermediate) would let the check read the wrong tree.
#
# Usage: find_engine_bin.sh <build output dir>

set -euo pipefail

root="${1:?usage: find_engine_bin.sh <build output dir>}"
if [ ! -d "${root}" ]; then
  echo "no bundled engine found: ${root}/ does not exist" >&2
  exit 1
fi
matches=$(find "${root}" -type d -path '*lilbee_engine/bin')

if [ -z "${matches}" ]; then
  echo "no bundled engine found under ${root}/" >&2
  exit 1
fi
if [ "$(printf '%s\n' "${matches}" | wc -l)" -ne 1 ]; then
  echo "more than one bundled engine under ${root}/, cannot tell which ships:" >&2
  printf '%s\n' "${matches}" >&2
  exit 1
fi
printf '%s\n' "${matches}"
