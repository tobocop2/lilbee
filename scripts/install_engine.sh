#!/usr/bin/env bash
# Download a prebuilt llama.cpp release and print the directory that holds
# llama-server. The integration lane needs a real engine: the lilbee-engine
# wheel ships an empty bin/ outside the release wheel build, so resolution
# falls through to PATH and CI must put a real binary there.
#
# Usage: install_engine.sh <llama-cpp-tag> <runner-os> <dest-dir>
# Prints the directory containing llama-server to stdout.

set -euxo pipefail

readonly DOWNLOAD_ATTEMPTS=3
readonly ENGINE_REPO=ggml-org/llama.cpp

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

tag="${1-}"
runner_os="${2-}"
dest="${3-}"

if [ -z "${tag}" ] || [ -z "${runner_os}" ] || [ -z "${dest}" ]; then
  echo "install_engine.sh: usage: install_engine.sh <tag> <runner-os> <dest-dir>" >&2
  exit 2
fi

# The asset suffix names the platform and carries the archive format, so the
# mapping from a runner to an asset lives here alone. The glob anchors on the
# suffix: a bare -arm64 name must not match the -kleidiai variant.
case "${runner_os}" in
  Linux)   suffix=bin-ubuntu-x64.tar.gz ;;
  macOS)   suffix=bin-macos-arm64.tar.gz ;;
  Windows) suffix=bin-win-cpu-x64.zip ;;
  *)
    echo "install_engine.sh: no asset pattern for runner os '${runner_os}'" >&2
    exit 2
    ;;
esac

workdir=$(mktemp -d)
trap 'rm -rf "${workdir}"' EXIT
archive="${workdir}/engine-${suffix}"

bash "${script_dir}/ci_retry.sh" "${DOWNLOAD_ATTEMPTS}" \
  gh release download "${tag}" --repo "${ENGINE_REPO}" \
  --pattern "*${suffix}" --output "${archive}" --clobber

mkdir -p "${dest}"
case "${suffix}" in
  *.zip)    unzip -q -o "${archive}" -d "${dest}" ;;
  *.tar.gz) tar -xzf "${archive}" -C "${dest}" ;;
esac

# Test the binary, not the string that names its directory. `dirname ''` is
# '.', so a guard on the dirname output passes after find matches nothing, and
# the caller then puts the working directory on PATH.
exe=$(find "${dest}" \( -name llama-server -o -name llama-server.exe \) -type f | head -1)
if [ ! -x "${exe:-/nonexistent}" ]; then
  echo "install_engine.sh: no llama-server in the ${runner_os} ${tag} asset" >&2
  exit 1
fi

dirname "${exe}"
