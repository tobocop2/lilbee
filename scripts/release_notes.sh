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

# A pin bump reaches the notes as one PR title, which hides the model support it
# adds. Same section attach-prerelease appends to the pre-release body, so the
# manual promote path does not silently drop it. Needs both tags in the local
# history, which a release checkout has.
if [ -n "$prev" ]; then
  old_archs=$(mktemp)
  new_archs=$(mktemp)
  archs_path="src/lilbee/_generated/engine_archs.py"
  root=$(cd "$(dirname "$0")/.." && pwd)
  if git -C "$root" show "${prev}:${archs_path}" > "$old_archs" 2>/dev/null \
    && git -C "$root" show "${tag}:${archs_path}" > "$new_archs" 2>/dev/null; then
    table=$(python3 "${root}/tools/release_arch_table.py" \
      --old-file "$old_archs" --new-file "$new_archs")
    if [ -n "$table" ]; then
      printf '\n%s\n' "$table"
    fi
  fi
  rm -f "$old_archs" "$new_archs"
fi
