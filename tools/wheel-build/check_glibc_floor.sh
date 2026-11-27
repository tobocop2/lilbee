#!/usr/bin/env bash
# Fail when a file in a directory imports a glibc symbol newer than 2.34 or a libstdc++
# symbol newer than GLIBCXX_3.4.29: the floor of the manylinux_2_34 wheel tag on RHEL 9.
#
# Argument:
#   the directory to scan, recursively; every file with undefined dynamic symbols counts
# Reads:
#   MAX_GLIBC     highest glibc version a file may import (default 2.34)
#   MAX_GLIBCXX   highest libstdc++ version a file may import (default 3.4.29)

set -euo pipefail

dir="${1:?the directory to scan is required}"
max_glibc="${MAX_GLIBC:-2.34}"
max_glibcxx="${MAX_GLIBCXX:-3.4.29}"

# Is version $1 newer than version $2?
newer() {
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ]
}

# The highest version of one symbol family among a file's imports.
highest() {
  local imports="$1" family="$2"
  printf '%s\n' "${imports}" | grep -o "${family}_[0-9.]*" | sed "s/${family}_//" | sort -V | tail -1 || true
}

checked=0
failed=0
while IFS= read -r -d '' file; do
  imports="$(objdump -T "${file}" 2>/dev/null | grep -F '*UND*' || true)"
  [ -n "${imports}" ] || continue
  checked=$((checked + 1))
  glibc="$(highest "${imports}" GLIBC)"
  glibcxx="$(highest "${imports}" GLIBCXX)"
  echo "floor ${file#"${dir}"/}: glibc ${glibc:-none}, libstdc++ ${glibcxx:-none}"
  if [ -n "${glibc}" ] && newer "${glibc}" "${max_glibc}"; then
    echo "FLOOR ${file#"${dir}"/} needs glibc ${glibc}, above ${max_glibc}" >&2
    failed=1
  fi
  if [ -n "${glibcxx}" ] && newer "${glibcxx}" "${max_glibcxx}"; then
    echo "FLOOR ${file#"${dir}"/} needs libstdc++ ${glibcxx}, above ${max_glibcxx}" >&2
    failed=1
  fi
done < <(find "${dir}" -type f -print0)

[ "${checked}" -gt 0 ] || { echo "no dynamic file found under ${dir}" >&2; exit 1; }
echo "floor: ${checked} files checked"
[ "${failed}" -eq 0 ]
