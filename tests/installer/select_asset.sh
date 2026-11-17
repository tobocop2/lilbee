#!/bin/sh
# Runs install.sh's select_asset on one machine case.
# Usage: select_asset.sh os arch avx2 cuda kfd gfx rocm_userland variant
# Prints the asset name; exits 1 with the reason when the host has no build.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
LILBEE_INSTALL_LIB=1
export LILBEE_INSTALL_LIB
# shellcheck source-path=SCRIPTDIR source=../../site/install.sh
. "${here}/../../site/install.sh"
SUMS_FILE="${here}/SHA256SUMS"
GFX_FILE="${here}/rocm.gfx.txt"
export SUMS_FILE GFX_FILE
select_asset "$@"
