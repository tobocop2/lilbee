#!/usr/bin/env bash
# Functions that the engine pack scripts share. Source this file; the caller sets `set -euo pipefail`.

# The host as GitHub names it: Linux, macOS or Windows.
host_os() {
  if [ -n "${RUNNER_OS:-}" ]; then
    printf '%s\n' "${RUNNER_OS}"
    return
  fi
  case "$(uname -s)" in
    Linux) echo Linux ;;
    Darwin) echo macOS ;;
    MINGW* | MSYS* | CYGWIN*) echo Windows ;;
    *)
      echo "packs do not build on $(uname -s)" >&2
      return 1
      ;;
  esac
}

# Print the name of the first interpreter on PATH that is Python 3.12.
python_312() {
  local name version
  for name in python python3.12 python3; do
    command -v "${name}" >/dev/null || continue
    version="$("${name}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null | tr -d '\r')" || continue
    if [ "${version}" = 3.12 ]; then
      printf '%s\n' "${name}"
      return
    fi
  done
  echo "Python 3.12 is not on PATH. Install it and run the command again." >&2
  return 1
}

# Print the sha256 line of one file, "<digest>  <file name>", as sha256sum writes it in the folder of the file.
sha256_line() {  # file
  local dir name
  dir="$(dirname "$1")"
  name="$(basename "$1")"
  if command -v sha256sum >/dev/null; then
    (cd "${dir}" && sha256sum "${name}")
  else
    (cd "${dir}" && shasum -a 256 "${name}")
  fi
}

# Download one asset of a GitHub release into a folder and compare it to the digest that GitHub lists.
fetch_release_asset() {  # repository tag asset folder
  local repo="$1" tag="$2" name="$3" dir="$4" want got
  gh release download "${tag}" -R "${repo}" --pattern "${name}" --dir "${dir}" || return 1
  want="$(gh release view "${tag}" -R "${repo}" --json assets \
    --jq ".assets[] | select(.name==\"${name}\") | .digest" | tr -d '\r')" || return 1
  got="$(sha256_line "${dir}/${name}")" || return 1
  got="sha256:${got%% *}"
  [ "${want}" = "${got}" ] || { echo "digest mismatch ${name}: ${want} ${got}" >&2; return 1; }
}

# Join the licence terms on stdin, one per line, with " AND ". A term keeps its first place; a repeat is dropped.
merge_terms() {
  tr -d '\r' | grep -v '^$' \
    | awk '!seen[$0]++ { printf "%s%s", (n++ ? " AND " : ""), $0 } END { print "" }'
}

# Copy the one wheel of a build folder into the output folder and write its sha256 file beside it.
collect_wheel() {  # build-folder output-folder
  local from="$1" out="$2" wheels name
  mkdir -p "${out}"
  cp "${from}"/*.whl "${out}/" || return 1
  wheels=("${out}"/*.whl)
  [ "${#wheels[@]}" -eq 1 ] || { echo "expected one wheel in ${out}, found ${#wheels[@]}" >&2; return 1; }
  name="$(basename "${wheels[0]}")"
  sha256_line "${wheels[0]}" > "${out}/${name}.sha256" || return 1
  cat "${out}/${name}.sha256"
  ls -l "${out}"
}
