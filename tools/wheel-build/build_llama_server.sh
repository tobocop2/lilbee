#!/usr/bin/env bash
# Build a SELF-CONTAINED llama.cpp `llama-server` binary for lilbee's local
# engine fleet. The binary plus its ggml/llama/mtmd shared libraries are copied
# into packaging/engine-wheel/ with a baked rpath (`$ORIGIN` on Linux,
# `@loader_path` on macOS), so the wheel carries everything it needs and lilbee
# depends on no separate inference library.
#
# Reads:
#   BACKEND            cpu|vulkan|metal|cu121..cu125|rocm|sycl
#   LLAMA_CPP_VERSION  llama.cpp source tag (via the llama-cpp-python release that
#                      vendors it; defaults to the pin below)
#   TARGET_ARCH        cross-compile target (optional; defaults to host)
#   LLAMA_BUILD_DIR    work dir (default /tmp/llama-build)

set -euxo pipefail

# Pinned llama.cpp source: the llama-cpp-python release tag whose vendored
# llama.cpp commit we build the server from. Bump deliberately (and re-run the
# Metal/CPU/GPU self-check matrix) rather than tracking latest. llama-cpp-python
# is only a BUILD-TIME source here -- lilbee no longer depends on it at runtime.
_DEFAULT_LLAMA_CPP_VERSION="0.3.23"

# Pinned source tags for the two Go engine helpers bundled alongside llama-server.
# Built from source (deterministic, no release-asset-name guessing); the wheel-build
# job provides the Go toolchain. Bump deliberately.
_LLAMA_SWAP_VERSION="v223"
_GGUF_PARSER_REF="main"

backend="${BACKEND:?BACKEND is required}"
build_dir="${LLAMA_BUILD_DIR:-/tmp/llama-build}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_arch="${TARGET_ARCH:-}"
pkg_bin_dir="${script_dir}/../../packaging/engine-wheel/lilbee_engine/bin"
version="${LLAMA_CPP_VERSION:-${_DEFAULT_LLAMA_CPP_VERSION}}"

# rpath so the binary and libs find each other from the same dir at runtime.
case "$(uname -s)" in
  Darwin) rpath='@loader_path' ;;
  *)      rpath='$ORIGIN' ;;
esac

# llama-cpp-python vendors llama.cpp as a submodule; clone at the matching tag so
# the server's GGUF support is a known-good combination.
# Windows MAX_PATH (260 chars): llama.cpp's vendored server webui has paths long
# enough to fail submodule checkout without long-path support. No-op elsewhere.
git config --global core.longpaths true
src="${build_dir}/llama-cpp-python-${version}"
mkdir -p "${build_dir}"
if [ ! -d "${src}" ]; then
  git clone --depth 1 --branch "v${version}" --recurse-submodules \
    https://github.com/abetlen/llama-cpp-python "${src}"
fi

# Same backend flags as the wheel build (GGML_* cmake flags apply to the server
# target verbatim), plus the server target and a baked install-rpath. SSL/CURL
# off: the fleet only talks to localhost servers, so we avoid the OpenSSL/libcurl
# link deps. BUILD_SHARED_LIBS=ON keeps ggml/llama/mtmd as separate libs we ship
# next to the binary, so a CUDA fatbin isn't statically duplicated per server.
eval "$(BACKEND="${backend}" TARGET_ARCH="${target_arch}" "${script_dir}/cmake_args.sh")"

# CMAKE_CUDA_ARCHITECTURES=all-major is a CMake 3.23+ keyword. On older cmake it
# expands to an empty arch and nvcc fails ("Unsupported gpu architecture
# compute_"). Substitute an explicit arch list so older boxes still build.
_CUDA_ARCH_FALLBACK="70;75;80;86;89;90"
if [[ "${CMAKE_ARGS}" == *"all-major"* ]]; then
  cmake_version="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  cmake_major="${cmake_version%%.*}"
  cmake_minor="$(printf '%s' "${cmake_version}" | cut -d. -f2)"
  if (( cmake_major < 3 || (cmake_major == 3 && cmake_minor < 23) )); then
    echo "cmake ${cmake_version} < 3.23: substituting CUDA arch list ${_CUDA_ARCH_FALLBACK} for all-major" >&2
    CMAKE_ARGS="${CMAKE_ARGS/all-major/${_CUDA_ARCH_FALLBACK}}"
  fi
fi
# shellcheck disable=SC2086
cmake -S "${src}/vendor/llama.cpp" -B "${src}/server-build" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_SERVER_SSL=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="${rpath}" ${CMAKE_ARGS}
cmake --build "${src}/server-build" --target llama-server --config Release -j

binary=$(find "${src}/server-build" -type f \( -name 'llama-server' -o -name 'llama-server.exe' \) | head -1)
[ -n "${binary}" ] || { echo "llama-server binary not found after build" >&2; exit 1; }
bindir=$(dirname "${binary}")

# Reset the bundle dir so a stale lib from a previous build can't ship.
rm -rf "${pkg_bin_dir}"
mkdir -p "${pkg_bin_dir}"
cp "${binary}" "${pkg_bin_dir}/"

# Bundle EVERY shared lib the server links: ggml (+ its backend split libs),
# llama, and mtmd. They sit beside the binary, and the baked rpath resolves them
# there, so the wheel is self-contained on every platform.
shopt -s nullglob
for lib in "${bindir}"/*.so "${bindir}"/*.so.* "${bindir}"/*.dylib "${bindir}"/*.dll; do
  cp "${lib}" "${pkg_bin_dir}/"
done
shopt -u nullglob

# Build the two Go engine helpers into the same wheel bin/. llama-swap is the
# process supervisor + OpenAI proxy; gguf-parser is the UMA-aware VRAM estimator.
# Both are single static binaries with no shared libs (unlike llama-server).
command -v go >/dev/null || { echo "go toolchain required to build llama-swap/gguf-parser" >&2; exit 1; }
go_build_dir="${LLAMA_BUILD_DIR:-/tmp/llama-build}/go-engine"
exe_suffix=""
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) exe_suffix=".exe" ;; esac

rm -rf "${go_build_dir}"
mkdir -p "${go_build_dir}"
git clone -q --depth 1 --branch "${_LLAMA_SWAP_VERSION}" https://github.com/mostlygeek/llama-swap.git "${go_build_dir}/llama-swap"
( cd "${go_build_dir}/llama-swap" && go build -trimpath -o "${pkg_bin_dir}/llama-swap${exe_suffix}" . )

# gguf-parser's cmd has a nested go.mod, so build from inside cmd/gguf-parser.
git clone -q --depth 1 --branch "${_GGUF_PARSER_REF}" https://github.com/gpustack/gguf-parser-go.git "${go_build_dir}/gguf-parser-go"
( cd "${go_build_dir}/gguf-parser-go/cmd/gguf-parser" && go build -trimpath -o "${pkg_bin_dir}/gguf-parser${exe_suffix}" . )

echo "Built self-contained engine (${backend}: llama-server + llama-swap + gguf-parser) -> ${pkg_bin_dir}/"
ls -lh "${pkg_bin_dir}/"
