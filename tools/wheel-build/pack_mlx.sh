#!/usr/bin/env bash
# Build and check the mlx engine pack of its one cell. Run it from a fresh lilbee checkout:
#
#   OUT=/path/to/out BACKEND=metal ... bash tools/wheel-build/pack_mlx.sh
#
# The engine-packs repository sets the variables from its cells file (scripts/pack.sh):
#   BACKEND, PLATFORM   the backend and the platform tag of the wheel: MLX_BACKEND and MLX_PLATFORM of engines/mlx.env
#   BUILD_NUMBER        the build number of the wheel
#   LICENSE_TERMS       the licence terms of mlx-lm and mlx, which the licence expression of the wheel starts with
#   OUT                 the folder that receives the checked wheel, its sha256 file and its licence expression
#
# The other pins of the build are in engines/mlx.env. tools/wheel-build/mlx_wheel.sh builds the wheel. The script
# needs macOS on arm64, Python 3.12, curl and zstd. It refuses a checkout that an earlier build used.

set -euo pipefail

# The mode of each file goes into the wheel.
umask 022

# shellcheck source=tools/wheel-build/pack_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/pack_common.sh"

PIN=engines/mlx.env

# Print the value of one key of the pin file.
pin_value() {  # key
  sed -n "s/^$1=//p" "${PIN}" | tr -d '\r'
}

require_inputs() {
  local name want
  for name in BACKEND PLATFORM BUILD_NUMBER LICENSE_TERMS OUT; do
    [ -n "${!name:-}" ] || { echo "pack_mlx: ${name} is not set" >&2; return 2; }
  done
  [ "$(host_os)" = macOS ] || { echo "pack_mlx: the mlx pack builds on macOS only" >&2; return 2; }
  for name in curl zstd; do
    command -v "${name}" > /dev/null || { echo "pack_mlx: ${name} is not installed" >&2; return 2; }
  done
  want="$(pin_value MLX_BACKEND)"
  [ "${BACKEND}" = "${want}" ] || { echo "pack_mlx: the cell names the backend ${BACKEND}; ${PIN} names ${want}" >&2; return 2; }
  want="$(pin_value MLX_PLATFORM)"
  [ "${PLATFORM}" = "${want}" ] || { echo "pack_mlx: the cell names the platform ${PLATFORM}; ${PIN} names ${want}" >&2; return 2; }
  for name in out wheel pack-venv check; do
    [ ! -e "${name}" ] || { echo "pack_mlx: an earlier build used this checkout (${name} exists). Use a fresh one." >&2; return 2; }
  done
}

# Refuse a wheel whose name or WHEEL file does not carry the version, the build and the platform of the cell.
check_tag() {  # wheel
  local name want
  name="$(basename "$1")"
  want="lilbee_engine_mlx-$(pin_value MLX_LM_VERSION)-${BUILD_NUMBER}.${BACKEND}-py3-none-${PLATFORM}.whl"
  [ "${name}" = "${want}" ] || { echo "pack_mlx: the wheel is ${name}, not ${want}" >&2; return 1; }
  dist_info "$1" WHEEL | grep -qx "Tag: py3-none-${PLATFORM}" \
    || { echo "pack_mlx: the WHEEL file of ${name} does not carry ${PLATFORM}" >&2; return 1; }
}

# Write the licence expression of the wheel to a file. It must start with the terms of the cell.
write_expression() {  # wheel file
  local expression
  expression="$(dist_info "$1" METADATA | metadata_header | sed -n 's/^License-Expression: //p')"
  case "${expression}" in
    "${LICENSE_TERMS}" | "${LICENSE_TERMS} AND "*) ;;
    *)
      echo "pack_mlx: the licence expression '${expression}' does not start with the cell terms '${LICENSE_TERMS}'" >&2
      return 1
      ;;
  esac
  printf '%s\n' "${expression}" > "$2"
}

# Unpack the wheel with the modes of its files and ask its chat server for its help: the bundled interpreter loads
# mlx-lm, mlx and Metal.
start_server() {  # wheel
  local server="${PWD}/check/lilbee_engine_mlx/bin/mlx-lm-server"
  python -c 'import os, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
[os.chmod(z.extract(i, sys.argv[2]), (i.external_attr >> 16) or 0o644) for i in z.infolist()]' "$1" check
  echo "server: ${server}"
  (cd / && "${server}" --help > /dev/null) \
    || { echo "pack_mlx: the chat server of the wheel does not start" >&2; return 1; }
}

main() {
  local wheel
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  require_inputs
  use_build_python
  install_build_tools python tools/wheel-build/build-tools.constraints tools/wheel-build/mlx-tools-macos.constraints
  mkdir out
  BUILD_NUMBER="${BUILD_NUMBER}" bash tools/wheel-build/mlx_wheel.sh out
  collect_wheel out wheel
  wheel="$(ls wheel/*.whl)"
  check_tag "${wheel}"
  write_expression "${wheel}" wheel/license-expression.txt
  start_server "${wheel}"
  # Only a wheel that passed its checks reaches the output folder.
  mkdir -p "${OUT}"
  cp wheel/* "${OUT}/"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main
fi
