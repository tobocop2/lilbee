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
_DEFAULT_LLAMA_CPP_VERSION="0.3.30"

# Pinned source tags for the two Go engine helpers bundled alongside llama-server.
# Built from source (deterministic, no release-asset-name guessing); the wheel-build
# job provides the Go toolchain. Bump deliberately.
_LLAMA_SWAP_VERSION="v223"
_GGUF_PARSER_REF="v0.24.1"

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

# GitHub sometimes 403s unauthenticated clones from shared runner IPs; retry
# with backoff, clearing any partial checkout first.
clone_with_retry() {
  local dest="${!#}" attempt
  for attempt in 1 2 3; do
    rm -rf "${dest}"
    if git clone "$@"; then
      return 0
    fi
    if [ "${attempt}" -lt 3 ]; then
      echo "git clone failed (attempt ${attempt}/3); retrying" >&2
      sleep $((attempt * 20))
    fi
  done
  return 1
}

# llama-cpp-python vendors llama.cpp as a submodule; clone at the matching tag so
# the server's GGUF support is a known-good combination.
# Windows MAX_PATH (260 chars): llama.cpp's vendored server webui has paths long
# enough to fail submodule checkout without long-path support. No-op elsewhere.
git config --global core.longpaths true
src="${build_dir}/llama-cpp-python-${version}"
mkdir -p "${build_dir}"
if [ ! -d "${src}" ]; then
  clone_with_retry --depth 1 --branch "v${version}" --recurse-submodules \
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
# CMAKE_DISABLE_FIND_PACKAGE_OpenSSL: the vendored cpp-httplib links whatever
# OpenSSL the build host has (Homebrew on macOS runners, distro libssl on
# Linux) even with LLAMA_SERVER_SSL=OFF, baking in a library path that does
# not exist on user machines. The fleet only talks to localhost; hide OpenSSL
# from the build entirely.
cmake -S "${src}/vendor/llama.cpp" -B "${src}/server-build" \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_SERVER_SSL=OFF -DLLAMA_CURL=OFF -DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=ON \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="${rpath}" ${CMAKE_ARGS}
# Bounded parallelism: a bare -j lets make spawn unlimited jobs, and the CUDA/ROCm
# translation units OOM-kill the compilers on 7GB CI runners. ENGINE_BUILD_JOBS
# overrides; the default is the host's core count.
build_jobs="${ENGINE_BUILD_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
cmake --build "${src}/server-build" --target llama-server --config Release -j "${build_jobs}"

# Prefer the collected bin/ output: CMake also leaves a per-target copy of the
# binary in its target directory WITHOUT the shared libs beside it, and find's
# traversal order is filesystem-dependent, so picking "whichever comes first"
# shipped lib-less bundles.
binary=""
for candidate in "${src}/server-build/bin/llama-server" "${src}/server-build/bin/Release/llama-server.exe" "${src}/server-build/bin/llama-server.exe"; do
  if [ -f "${candidate}" ]; then
    binary="${candidate}"
    break
  fi
done
if [ -z "${binary}" ]; then
  binary=$(find "${src}/server-build" -type f \( -name 'llama-server' -o -name 'llama-server.exe' \) | head -1)
fi
[ -n "${binary}" ] || { echo "llama-server binary not found after build" >&2; exit 1; }

# Reset the bundle dir so a stale lib from a previous build can't ship.
rm -rf "${pkg_bin_dir}"
mkdir -p "${pkg_bin_dir}"
cp "${binary}" "${pkg_bin_dir}/"

# Bundle EVERY shared lib the server links: ggml (+ its backend split libs),
# llama, mtmd, and the server-impl split. Search the whole build tree, not just
# the binary's directory: this llama.cpp scatters library outputs across target
# directories. They sit beside the binary and the baked rpath resolves them
# there, so the wheel is self-contained on every platform.
# Symlinks included: the SONAME names the binary loads by (libllama.0.dylib)
# are symlinks to the versioned files; cp dereferences each into a regular
# file under the loadable name.
while IFS= read -r -d '' lib; do
  cp "${lib}" "${pkg_bin_dir}/"
done < <(find "${src}/server-build" \( -name CMakeFiles -o -name CMakeScratch -o -path '*vulkan-shaders-gen-prefix*' \) -prune -o \
  \( -type f -o -type l \) \( -name '*.so' -o -name '*.so.*' -o -name '*.dylib' -o -name '*.dll' \) -print0)

# The copied closure must actually resolve: exec the bundled binary from the
# bundle dir. A missing lib fails here, at build time, instead of on a user's
# machine. Skipped when cross-compiling (the host can't exec the target).
if [ -z "${target_arch}" ]; then
  "${pkg_bin_dir}/llama-server" --version
fi

# Build the two Go engine helpers into the same wheel bin/. llama-swap is the
# process supervisor + OpenAI proxy; gguf-parser is the UMA-aware VRAM estimator.
# Both are single static binaries with no shared libs (unlike llama-server).
command -v go >/dev/null || { echo "go toolchain required to build llama-swap/gguf-parser" >&2; exit 1; }
go_build_dir="${LLAMA_BUILD_DIR:-/tmp/llama-build}/go-engine"
exe_suffix=""
case "$(uname -s)" in MINGW* | MSYS* | CYGWIN*) exe_suffix=".exe" ;; esac

rm -rf "${go_build_dir}"
mkdir -p "${go_build_dir}"
clone_with_retry -q --depth 1 --branch "${_LLAMA_SWAP_VERSION}" https://github.com/mostlygeek/llama-swap.git "${go_build_dir}/llama-swap"
( cd "${go_build_dir}/llama-swap" && go build -trimpath -o "${pkg_bin_dir}/llama-swap${exe_suffix}" . )

# gguf-parser's cmd has a nested go.mod, so build from inside cmd/gguf-parser.
clone_with_retry -q --depth 1 --branch "${_GGUF_PARSER_REF}" https://github.com/gpustack/gguf-parser-go.git "${go_build_dir}/gguf-parser-go"
( cd "${go_build_dir}/gguf-parser-go/cmd/gguf-parser" && go build -trimpath -o "${pkg_bin_dir}/gguf-parser${exe_suffix}" . )

echo "Built self-contained engine (${backend}: llama-server + llama-swap + gguf-parser) -> ${pkg_bin_dir}/"
ls -lh "${pkg_bin_dir}/"
