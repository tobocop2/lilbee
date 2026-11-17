#!/bin/sh
# Runs install.sh's probe_hardware for one host and prints the facts it read.
# Usage: probe_hardware.sh os arch
# Prints: avx2=<yes|no> cuda=<version|-> kfd=<yes|no> gfx=<list|-> userland=<yes|no>
set -eu
here=$(cd "$(dirname "$0")" && pwd)
LILBEE_INSTALL_LIB=1
export LILBEE_INSTALL_LIB
# shellcheck source-path=SCRIPTDIR source=../../site/install.sh
. "${here}/../../site/install.sh"
probe_hardware "$1" "$2"
echo "avx2=${avx2} cuda=${cuda} kfd=${kfd} gfx=${gfx} userland=${userland}"
