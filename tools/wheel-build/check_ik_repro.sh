#!/usr/bin/env bash
# Decide whether two or more CUDA builds of one ik_llama.cpp commit are equal enough.
#
# Rule: every file in the server folder is byte-equal, except libggml.so. For libggml.so, the host code
# (.text, .rodata, .dynsym, .dynstr) must be equal. Its GPU code (.nv_fatbin) and the sections that hold
# addresses after it may differ: the register allocation of cicc gives other GPU code from run to run.
# A file that is not an ELF file is always byte-compared. A section that objcopy cannot extract, or that
# is empty, is a difference.
#
# Arguments:
#   two or more server folders, one for each build
# Exit status:
#   0 when the builds are equal under the rule, 1 when a file differs, 2 on a usage error

set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "check_ik_repro: give two or more server folders" >&2
  exit 2
fi
dirs=("$@")
host_sections=(.text .rodata .dynsym .dynstr)

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# Print the first 16 hex digits of the sha256 of a file, or "none" when the file is missing.
digest() {
  if [ -f "$1" ]; then sha256sum < "$1" | cut -c1-16; else echo none; fi
}

# Do all the folders hold one value for the file? Prints the distinct digests.
distinct() {
  local d
  for d in "${dirs[@]}"; do digest "${d}/$1"; done | sort -u
}

# Print the digest of one section of one file; fail when objcopy fails or the section is empty.
section_digest() {
  local out="${tmp}/section.$RANDOM.bin"
  objcopy -O binary --only-section="$2" "$1" "${out}" || return 1
  [ -s "${out}" ] || return 1
  sha256sum < "${out}" | cut -c1-16
}

# Are the host-code sections of the file equal in every folder?
host_code_equal() {
  local file="$1" s d sums equal=0
  for s in "${host_sections[@]}"; do
    sums=""
    for d in "${dirs[@]}"; do
      if ! sums="${sums}$(section_digest "${d}/${file}" "${s}")"$'\n'; then
        echo "DIFF ${file}:${s} (objcopy failed or the section is empty)"
        equal=1
        continue 2
      fi
    done
    if [ "$(printf '%s' "${sums}" | sort -u | wc -l)" -eq 1 ]; then
      echo "SAME ${file}:${s}"
    else
      echo "DIFF ${file}:${s}"
      equal=1
    fi
  done
  return "${equal}"
}

status=0
while IFS= read -r f; do
  sums="$(distinct "${f}")"
  if [ "$(echo "${sums}" | wc -l)" -eq 1 ]; then
    echo "SAME ${f}"
  elif [ "${f}" = libggml.so ]; then
    host_code_equal "${f}" || status=1
  else
    echo "DIFF ${f} $(echo "${sums}" | tr '\n' ' ')"
    status=1
  fi
done < <(for d in "${dirs[@]}"; do (cd "${d}" && find . -type f | cut -c3-); done | sort -u)

if [ "${status}" -eq 0 ]; then echo "ACCEPT: ${#dirs[@]} builds are equal"; else echo "REJECT: ${#dirs[@]} builds differ"; fi
exit "${status}"
