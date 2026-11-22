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

# Put a venv of Python 3.12 first on PATH: the build tools do not go into the Python of the computer.
use_build_python() {
  local base
  base="$(python_312)"
  "${base}" -m venv pack-venv
  if [ "$(host_os)" = Windows ]; then PATH="${PWD}/pack-venv/Scripts:${PATH}"; else PATH="${PWD}/pack-venv/bin:${PATH}"; fi
  export PATH
}

# Print one file of the dist-info folder of a wheel with LF line ends: a Windows build writes CRLF.
dist_info() {  # wheel file
  python -c 'import pathlib, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
paths = [pathlib.PurePosixPath(n) for n in z.namelist()]
(path,) = [p for p in paths if p.name == sys.argv[2] and p.parent.suffix == ".dist-info"]
sys.stdout.buffer.write(b"\n".join(z.read(str(path)).splitlines()) + b"\n")' "$1" "$2"
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

# Install the pinned build tools into a Python. Each file is a requirements file that lists every tool, and what the
# tool needs, with one version and its digests. pip refuses a line without a version or a digest, an archive with
# another digest, and a listed tool that needs one that no file lists. pip does not know which tools the build needs, so
# it accepts a file that lacks one: the tests of the packs check the names. This function refuses a file that pins no
# tool, because pip installs nothing from it and succeeds.
# pip does not check the digest of a tool that the Python already holds at that version. --force-reinstall makes pip
# install each listed tool again from an archive that it checks. A package that no file lists stays in the Python and
# the build sees it, so the caller passes the Python of a fresh venv.
# A constraints file is not enough: a pip before 26.1 refuses a named tool that only a constraint pins.
install_build_tools() {  # python requirements-file...
  local python="$1" file files=()
  shift
  [ "$#" -gt 0 ] || { echo "install_build_tools: name a requirements file" >&2; return 2; }
  for file in "$@"; do
    grep -qE '^[[:alnum:]].*==' "${file}" || { echo "install_build_tools: ${file} pins no tool" >&2; return 2; }
    files+=(-r "${file}")
  done
  "${python}" -m pip install --require-hashes --force-reinstall "${files[@]}"
}

# Build the wheel of the project in the current folder with the tools that install_build_tools installed.
# The build makes no environment of its own, so it installs nothing: a tool that the project needs and the
# Python lacks stops the build.
build_wheel_with_tools() {  # python build-argument...
  local python="$1"
  shift
  "${python}" -m build --wheel --no-isolation "$@"
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
