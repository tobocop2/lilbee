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

# A CUDA build links the CUDA 12 runtime dynamically, and those libraries live in the
# toolkit, never in the build output copied above. Only libcuda / nvcuda comes from the
# driver; cudart, cublas and cublasLt do not, so they ship beside the binary like every
# other lib here. The baked $ORIGIN runpath resolves them on Linux, and on Windows the
# executable's own directory is the first place the loader looks.
#
# Both platforms broke without this, in different ways. On Windows cudart is a hard
# import of the process, so llama-server.exe died before binding its port with a
# "cudart64_12.dll was not found" dialog. On Linux ggml dlopens libggml-cuda.so and
# tolerates the failure, so the GPU silently disappeared and work fell back to CPU.
case "${backend}_$(uname -s)" in
  cu12*_MINGW* | cu12*_MSYS* | cu12*_CYGWIN*) cuda_platform="windows" ;;
  cu12*_Linux)                                cuda_platform="linux" ;;
  *)                                          cuda_platform="" ;;
esac

if [ -n "${cuda_platform}" ]; then
  # nvrtc is optional: nothing in the current ggml links it (readelf shows no
  # DT_NEEDED for it), and the Windows toolkit install omits the sub-package.
  # Copy it when the toolkit provides it; the other three are hard requirements.
  case "${cuda_platform}" in
    windows)
      cuda_lib_dir="$(cygpath -u "${CUDA_PATH:?CUDA_PATH must be set for a Windows CUDA build}")/bin"
      required_libs=(cudart64 cublas64 cublasLt64)
      shopt -s nullglob
      cuda_libs=(
        "${cuda_lib_dir}"/cudart64_*.dll
        "${cuda_lib_dir}"/cublas64_*.dll
        "${cuda_lib_dir}"/cublasLt64_*.dll
        "${cuda_lib_dir}"/nvrtc64_*.dll
      )
      shopt -u nullglob
      ;;
    linux)
      # install_gpu_toolkit.sh exports CUDA_HOME; fall back to the versionless symlink.
      cuda_lib_dir="${CUDA_HOME:-/usr/local/cuda}/lib64"
      [ -d "${cuda_lib_dir}" ] || cuda_lib_dir="${CUDA_HOME:-/usr/local/cuda}/targets/x86_64-linux/lib"
      required_libs=(libcudart.so.12 libcublas.so.12 libcublasLt.so.12)
      # Copy the SONAME names the binary loads by. These are symlinks to the versioned
      # files in the toolkit; cp dereferences each into a real file under the loadable
      # name, as the ggml libs above are handled.
      shopt -s nullglob
      cuda_libs=(
        "${cuda_lib_dir}"/libcudart.so.12
        "${cuda_lib_dir}"/libcublas.so.12
        "${cuda_lib_dir}"/libcublasLt.so.12
        "${cuda_lib_dir}"/libnvrtc.so.12
      )
      shopt -u nullglob
      ;;
  esac

  [ -d "${cuda_lib_dir}" ] || { echo "CUDA library dir ${cuda_lib_dir} does not exist" >&2; exit 1; }
  for lib in ${cuda_libs[@]+"${cuda_libs[@]}"}; do
    cp -L "${lib}" "${pkg_bin_dir}/"
  done

  # The bundle can't be verified by exec here: a driverless build host has no
  # libcuda/nvcuda, so the CUDA backend never loads and the server starts clean
  # whether or not these are present. Gate on the copy instead, and fail the build
  # rather than ship a bundle that only breaks on a user's machine.
  for required in "${required_libs[@]}"; do
    case "${cuda_platform}" in
      windows) pattern="${pkg_bin_dir}/${required}_*.dll" ;;
      linux)   pattern="${pkg_bin_dir}/${required}" ;;
    esac
    compgen -G "${pattern}" >/dev/null || {
      echo "no ${required} under ${cuda_lib_dir}: the ${cuda_platform} CUDA bundle is incomplete" >&2
      exit 1
    }
  done
fi

# The copied closure must actually resolve: exec the bundled binary from the
# bundle dir. A missing bundled lib fails here, at build time, instead of on a
# user's machine. Skipped when cross-compiling (the host can't exec the target)
# and for GPU-driver backends, whose binaries link the driver runtime
# (libcuda.so.1 / libamdhip64.so) that is absent on the driverless build host;
# the CPU/Vulkan/Metal cells exercise the identical copy logic, so the closure
# is still gated on every push.
case "${backend}" in
  cu* | rocm | sycl) _can_exec="" ;;
  *)                 _can_exec="1" ;;
esac
if [ -n "${_can_exec}" ] && [ -z "${target_arch}" ]; then
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
