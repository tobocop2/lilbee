#!/usr/bin/env bash
# Copy the NOTICE files and the vendored licence files of the built crates and the source tree.
#
# Usage: vendored_notices.sh <source-dir> <out-dir>
#
# cargo-about reads the licence that each crate declares. It does not read a NOTICE file (Apache-2.0
# section 4(d)) or the licence of code that a crate vendors in a third_party folder. This script copies
# those files, with their path under the crate, from the cargo registry sources and the cargo git
# checkouts that the build fetched, and from <source-dir>. <out-dir> gets one folder for each crate or tree that has such files.
set -euo pipefail

test "$#" -eq 2 || {
  echo "usage: vendored_notices.sh <source-dir> <out-dir>" >&2
  exit 2
}
src="$1" out="$2"
registry="${CARGO_HOME:-${HOME}/.cargo}/registry/src"
test -d "${registry}" || { echo "vendored_notices: no cargo registry at ${registry}" >&2; exit 1; }
mkdir -p "${out}"

copy_licences() {
  local root="$1" label="$2" file rel
  while IFS= read -r file; do
    rel="${file#"${root}"/}"
    mkdir -p "${out}/${label}/$(dirname "${rel}")"
    cp "${file}" "${out}/${label}/${rel}"
  done
}

copy_from() {
  copy_licences "$1" "$2" < <(find "$1" -path "$1/target" -prune -o -type f \
    \( -iname 'NOTICE*' -o -path '*/third_party/*' -o -path '*/third-party/*' \) \
    \( -iname 'NOTICE*' -o -iname 'LICEN[CS]E*' -o -iname 'COPYING*' \) -print | sort)
}

# A git dependency can hold a git submodule (candle-flash-attn holds cutlass) whose licence file sits
# outside a third_party folder, so every licence file of a git checkout is copied.
copy_git_checkout() {
  copy_licences "$1" "$2" < <(find "$1" -name .git -prune -o -type f \
    \( -iname 'NOTICE*' -o -iname 'LICEN[CS]E*' -o -iname 'COPYING*' \) -print | sort)
}

for crate in "${registry}"/*/*/; do
  copy_from "${crate%/}" "$(basename "${crate}")"
done
for checkout in "${CARGO_HOME:-${HOME}/.cargo}"/git/checkouts/*/*/; do
  test -d "${checkout}" || continue
  copy_git_checkout "${checkout%/}" "git-$(basename "$(dirname "${checkout}")")-$(basename "${checkout}")"
done
copy_from "${src}" "text-embeddings-inference"
find "${out}" -type f | wc -l
