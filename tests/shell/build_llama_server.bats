#!/usr/bin/env bats
# The Vulkan module check of tools/wheel-build/build_llama_server.sh. The block runs alone, with stubs for uname and ldd:
# the script itself needs a llama.cpp checkout and a compiler.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  BIN="${BATS_TEST_TMPDIR}/bin"
  PKG="${BATS_TEST_TMPDIR}/pkg bin"
  mkdir -p "${BIN}" "${PKG}"
  printf '#!/bin/sh\necho Linux\n' > "${BIN}/uname"
  # The block is the "if vulkan" statement that tests target_arch, from its first line to the first "fi" in column 0.
  BLOCK="$(awk '/^if \[ "\$\{backend\}" = "vulkan".*target_arch/ {on = 1} on {print} on && /^fi$/ {exit}' \
    "${REPO_ROOT}/tools/wheel-build/build_llama_server.sh")"
  [ -n "${BLOCK}" ]
}

# Run the block with a stub ldd whose body is the argument.
check_module() {
  printf '#!/bin/sh\n%s\n' "$1" > "${BIN}/ldd"
  chmod +x "${BIN}/uname" "${BIN}/ldd"
  : > "${PKG}/libggml-vulkan.so.0"
  PATH="${BIN}:${PATH}" backend=vulkan target_arch="" pkg_bin_dir="${PKG}" run bash -euo pipefail -c "${BLOCK}"
}

@test "a Vulkan module that ldd cannot read stops the build" {
  check_module 'echo "	not a dynamic executable"; exit 1'
  [ "$status" -eq 1 ]
  [[ "$output" == *"ldd failed on"*"libggml-vulkan.so.0"* ]]
}

@test "an ldd that fails with any other output stops the build" {
  check_module 'echo "ldd: error while loading"; exit 1'
  [ "$status" -eq 1 ]
  [[ "$output" == *"ldd failed on"*"libggml-vulkan.so.0"*"error while loading"* ]]
}

@test "a Vulkan module with a library not found stops the build" {
  check_module 'echo "	libvulkan.so.1 => not found"'
  [ "$status" -eq 1 ]
  [[ "$output" == *"libggml-vulkan.so.0 does not resolve from the bundle"*"libvulkan.so.1"* ]]
}

@test "a Vulkan module whose libraries resolve passes" {
  check_module 'echo "	libvulkan.so.1 => /usr/lib/libvulkan.so.1 (0x1)"'
  [ "$status" -eq 0 ]
}
