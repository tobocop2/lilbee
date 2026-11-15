#!/usr/bin/env bash
# Copy the licence texts of the code that a llama.cpp build ships into a wheel tree.
# The script holds no list of components. It reads two kinds of source tree.
#
# The checkout of the tag the wheel ships:
#   - every licence file by name (LICENSE*, LICENCE*, COPYING*), copied at its own path;
#   - every source file that opens with a licence grant (an MIT, BSD or Unlicense text in the
#     first lines), whose leading comment is written as <path>.NOTICE;
#   - every file under vendor/ that names the MIT licence in its first lines but carries no text (an
#     SPDX tag or "MIT License"): its header lines, which hold the copyright, then the MIT terms of
#     the checkout's own LICENSE file, written as <path>.NOTICE;
#   - the licence texts at the end of the single-header libraries that hold them there (the stb image
#     header, miniaudio, and the stb resize header when the checkout has it).
# The trees that the build fetched (the _deps folder of a CMake build directory):
#   - every licence file by name, copied under _deps/<name>/. A fetched tree with no licence file
#     fails the build.
# Files of the project itself (header names "The ggml authors") are skipped. So are the top-level
# folders that hold no shipped code (tests, docs, examples, scripts, gguf-py, pocs, benches) and
# every folder named with --skip.
#
# The list of copied files is written to MANIFEST in the output directory, so a later check
# can tell that no file vanished.
#
# Usage: collect_vendored_licences.sh [--skip <folder>]... <checkout> <out> <terms> [<deps>]
#   --skip   a folder of the checkout whose code the wheel does not ship, such as tools/ui
#   checkout the llama.cpp checkout at the tag the wheel ships
#   out      the directory that receives the copies
#   terms    a file that receives the SPDX names of the Unlicense, MIT-0, LLVM exception and Boost
#            texts found in the copies
#   deps     a folder that holds the fetched trees as <name>-src, such as the _deps folder of a
#            CMake build directory; a deps folder that is missing or holds no tree fails the build

set -euo pipefail

skips=()
while [ "${1:-}" = --skip ]; do
  skips+=("${2:?--skip needs a folder}")
  shift 2
done
src="${1:?the llama.cpp checkout is required}"
out="${2:?the output directory is required}"
terms="${3:?the terms file is required}"
deps="${4:-}"

grant='Permission is hereby granted|free and unencumbered software|Redistribution and use in source and binary forms'

copy_to() {
  mkdir -p "$(dirname "${out}/$1")"
  cat > "${out}/$1"
}

# in_head <file> <pattern> [grep flag]: whether the first 40 lines match. grep reads all of its
# input, so a first line larger than a pipe buffer cannot end the pipe early.
in_head() {
  [ "$(head -n 40 "$1" | grep -a -c "${3:--E}" -- "$2" || true)" -gt 0 ]
}

# header_comment <file>: the leading comments (/* */ blocks and // lines).
header_comment() {
  awk 'block {print; if (/\*\//) block = 0; next}
       /^[ \t]*$/ {print; next}
       /^[ \t]*\/\*/ {print; if (!/\*\//) block = 1; next}
       /^[ \t]*\/\// {print; next}
       {exit}' "$1"
}

# vendor_header <file>: the opening comment lines, up to the first blank line.
vendor_header() {
  awk '/^[ \t]*$/ {exit} !/^[ \t]*(\/\/|\/\*|\*)/ {exit} {print}' "$1"
}

# mit_terms: the permission text of the checkout's LICENSE, from its first grant line to the end.
mit_terms() {
  awk '/Permission is hereby granted/ {on = 1} on' "${src}/LICENSE"
}

# tail_text <file> <first line pattern> <marker>: the text from the pattern to the end of the file,
# which must hold the marker, so a moved text fails the build instead of shipping nothing.
tail_text() {
  local file="$1" start="$2" marker="$3" rel="${1#"${src}"/}"
  awk -v start="${start}" '$0 ~ start {on = 1} on' "${file}" | copy_to "${rel}.NOTICE"
  grep -q -F "${marker}" "${out}/${rel}.NOTICE" \
    || { echo "${rel}: no '${marker}' in the licence text" >&2; exit 1; }
}

# tree_files <root> [skipped folder]...: the files of a tree, in order, without the folders that no wheel ships.
tree_files() {
  local root="$1" prune=(-name .git -o -name node_modules) skip
  shift
  for skip in tests docs examples scripts gguf-py pocs benches ${1+"$@"}; do
    prune+=(-o -path "${root}/${skip}")
  done
  find "${root}" \( "${prune[@]}" \) -prune -o -type f -print | sort
}

is_licence_name() {
  case "$(basename "$1")" in
    LICENSE* | LICENCE* | COPYING*) return 0 ;;
  esac
  return 1
}

found=0
while IFS= read -r file; do
  rel="${file#"${src}"/}"
  if is_licence_name "${file}"; then
    [ "${rel}" = LICENSE ] && continue # the project's own licence ships as LICENSE-llama.cpp
    copy_to "${rel}" < "${file}"
    found=$((found + 1))
  elif [[ "${rel}" == vendor/* ]] && ! in_head "${file}" "Permission is hereby granted" \
    && in_head "${file}" 'SPDX-License-Identifier: *MIT$|^[/ *#]*MIT License'; then
    header="$(vendor_header "${file}")"
    terms_text="$(mit_terms)"
    { [ -n "${terms_text}" ] && printf '%s\n' "${header}" | grep -a -q -i 'copyright'; } \
      || { echo "${rel}: no copyright line in the header, or no MIT terms in LICENSE" >&2; exit 1; }
    printf '%s\n\n%s\n' "${header}" "${terms_text}" | copy_to "${rel}.NOTICE"
    found=$((found + 1))
  elif in_head "${file}" "${grant}" && ! in_head "${file}" "The ggml authors" -F; then
    header="$(header_comment "${file}")"
    if [ "$(printf '%s\n' "${header}" | grep -a -c -E "${grant}" || true)" -gt 0 ]; then
      printf '%s\n' "${header}" | copy_to "${rel}.NOTICE"
      found=$((found + 1))
    else
      echo "${rel}: the grant is not in the leading comment, so no notice was written" >&2
    fi
  fi
done < <(tree_files "${src}" ${skips[@]+"${skips[@]}"})
[ "${found}" -gt 0 ] || { echo "no licence file or header found under ${src}" >&2; exit 1; }

tail_text "${src}/vendor/stb/stb_image.h" '^This software is available under 2 licenses' "ALTERNATIVE A - MIT License"
tail_text "${src}/vendor/miniaudio/miniaudio.h" '^ALTERNATIVE 1 - Public Domain' "ALTERNATIVE 2 - MIT No Attribution"
# The ik_llama.cpp checkout also vendors the resize header, which the llama.cpp checkout does not.
if [ -f "${src}/vendor/stb/stb_image_resize2.h" ]; then
  tail_text "${src}/vendor/stb/stb_image_resize2.h" '^This software is available under 2 licenses' "ALTERNATIVE A - MIT License"
fi

# Each tree that the build fetched carries its own licence files.
if [ -n "${deps}" ]; then
  [ -d "${deps}" ] || { echo "the deps folder ${deps} does not exist" >&2; exit 1; }
  trees=0
  for tree in "${deps}"/*-src; do
    [ -d "${tree}" ] || continue
    trees=$((trees + 1))
    name="$(basename "${tree}")"
    name="${name%-src}"
    count=0
    while IFS= read -r file; do
      is_licence_name "${file}" || continue
      copy_to "_deps/${name}/${file#"${tree}"/}" < "${file}"
      count=$((count + 1))
    done < <(tree_files "${tree}")
    [ "${count}" -gt 0 ] || { echo "the fetched tree ${name} holds no licence file" >&2; exit 1; }
  done
  [ "${trees}" -gt 0 ] || { echo "the deps folder ${deps} holds no fetched tree (<name>-src)" >&2; exit 1; }
fi

# The SPDX names come from the copied texts, so a licence that a vendored file offers is stated.
: > "${terms}"
if grep -rqF "This is free and unencumbered software released into the public domain" "${out}"; then
  echo Unlicense >> "${terms}"
fi
if grep -rqF "MIT No Attribution" "${out}"; then
  echo MIT-0 >> "${terms}"
fi
if grep -rqF "LLVM Exceptions to the Apache 2.0 License" "${out}"; then
  echo "Apache-2.0 WITH LLVM-exception" >> "${terms}"
fi
if grep -rqF "Boost Software License - Version 1.0" "${out}"; then
  echo BSL-1.0 >> "${terms}"
fi
manifest="$(cd "${out}" && find . -type f | sed 's|^\./||' | sort)"
printf '%s\n' "${manifest}" | copy_to MANIFEST
find "${out}" -type f | sort
