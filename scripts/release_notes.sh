#!/usr/bin/env bash
# Print release notes for a tag as simple headings: one "## <pull request title>"
# per merged PR with a link, followed by the full changelog line. Reads the
# changelog GitHub generates from the tag diff, so it stays automated.
# Usage: release_notes.sh <owner/repo> <tag> [previous_tag]
set -euo pipefail

repo=$1
tag=$2
prev=${3:-}

args=(-f tag_name="$tag")
[ -n "$prev" ] && args+=(-f previous_tag_name="$prev")

gh api "repos/${repo}/releases/generate-notes" "${args[@]}" --jq .body | awk '
  /^\* / {
    s = $0; sub(/^\* /, "", s)
    url = s; sub(/^.* in /, "", url)
    title = s; sub(/ by @[^ ]+ in .*$/, "", title)
    num = url; sub(/^.*\/pull\//, "", num)
    if (url ~ /\/pull\//) printf "## %s\n[#%s](%s)\n\n", title, num, url
    else printf "## %s\n\n", title
    next
  }
  /^\*\*Full Changelog/ { print; next }
  { next }
'
