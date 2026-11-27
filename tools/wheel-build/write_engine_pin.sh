#!/usr/bin/env bash
# Write the pin of the llama.cpp engine wheel: the effective source versions and the backend flavor.
# Two lilbee processes share one engine fleet only when their pins match, so a cu125 wheel and a Vulkan wheel of
# the same versions must have different pins.
#
# The step that builds or installs the engine wheel runs this script, on every path. The engine binaries come
# from the mirror, the Actions cache or build_llama_server.sh, and the pin does not depend on which.
#
# Reads:
#   BACKEND         cpu|vulkan|metal|cu121..cu125|rocm
#   LLAMA_CPP_REF   llama.cpp ref of the build (default: the pin in engine-versions.env)

set -euo pipefail

backend="${BACKEND:?BACKEND is required}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
source "${script_dir}/../../engine-versions.env"

# An override ships under its own identity, so two different builds never alias each other's fleets.
version="${LLAMA_CPP_REF:-${ENGINE_LLAMA_CPP_REF}}"

cat > "${script_dir}/../../packaging/engine-wheel/lilbee_engine_llama_cpp/pins.py" <<PINS
"""Engine build identity; written by tools/wheel-build/write_engine_pin.sh."""

ENGINE_PIN = "llama-cpp-${version}+swap-${ENGINE_LLAMA_SWAP_VERSION}+gguf-${ENGINE_GGUF_PARSER_REF}+${backend}"
PINS
