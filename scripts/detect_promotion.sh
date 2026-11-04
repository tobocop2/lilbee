#!/usr/bin/env bash
# Detect a same-source promotion: a tag whose parent is itself a v* tag, whose
# only change is the version files, and whose commit subject starts with "Promote ".
# A back-to-back release changes the same files, so the subject tells them apart. Prints promoted_from= for $GITHUB_OUTPUT.
set -euo pipefail

tag="$1"
here=$(cd "$(dirname "$0")" && pwd)
excludes=()
while IFS= read -r file; do excludes+=(":(exclude)${file}"); done < <("${here}/release_version.sh" files)
promoted_from=""
parent=$(git rev-parse -q --verify HEAD^ 2>/dev/null || echo "")
if [ -n "${parent}" ]; then
  # ls-remote instead of fetching all tags. An annotated tag carries the commit
  # sha only on its ^{} peel line, so match that too and strip the suffix.
  from_tag=$(git ls-remote --tags origin \
    | awk -v sha="${parent}" '$1 == sha && $2 ~ /^refs\/tags\/v/ { sub(/\^\{\}$/, "", $2); sub("refs/tags/", "", $2); print $2; exit }')
  subject=$(git log -1 --format=%s HEAD)
  if [ -n "${from_tag}" ] && [ "${from_tag}" != "${tag}" ] && [[ "${subject}" == "Promote "* ]] \
    && git diff --quiet HEAD^ HEAD -- . "${excludes[@]}"; then
    promoted_from="${from_tag}"
    echo "${tag} is a same-source promotion of ${from_tag}" >&2
  fi
fi
echo "promoted_from=${promoted_from}"
