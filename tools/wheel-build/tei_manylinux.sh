#!/usr/bin/env bash
# Build the text-embeddings-router wheel for Linux x86_64 inside the manylinux_2_34 image
# (AlmaLinux 9, glibc 2.34). Runs as root in the container.
#
# Reads:
#   TEI_REPO, TEI_TAG    the router source to build
#   RUSTC_VERSION        the Rust toolchain of the tag (rust-toolchain.toml)
#   CUDA_CC              the CUDA compute capability to build for, for example 89; empty for the CPU build
#   CUDA_PACKAGE_SUFFIX  the toolkit version in package names, for example 12-9
#   CUDA_VERSION         the full toolkit version, for example 12.9.1; names the EULA text of the notice
#   WHEEL_VERSION        the version of the wheel
#   BACKEND              the backend of the wheel build tag, for example cuda-sm89
#   WORK                 the repository checkout (bind mounted)
#   OUT                  the directory for the wheel and its notes (bind mounted)
#   OWNER                uid:gid that owns the files left behind
# Writes OUT/repaired/*.whl, OUT/license-expression.txt and, for a CUDA build, OUT/stubs/libcuda.so.1.
#
# The CUDA runtime and the cuBLAS libraries are linked into the router as static libraries and are
# not changed by this script, except that nvprune drops the GPU code of other architectures from
# libcublas_static.a, as the upstream Dockerfile-cuda does. The notice for them is written
# after the build, from the libraries the build log shows linked.

set -euo pipefail

: "${TEI_REPO:?}" "${TEI_TAG:?}" "${RUSTC_VERSION:?}" "${WHEEL_VERSION:?}" "${BACKEND:?}" "${WORK:?}" "${OUT:?}" "${OWNER:?}"
CUDA_CC="${CUDA_CC:-}"
jobs="$(nproc)"
py=/opt/python/cp312-cp312/bin

packages=(gcc-toolset-13-gcc-c++ openssl-devel pkgconf-pkg-config jq)
if [ -n "${CUDA_CC}" ]; then
  : "${CUDA_PACKAGE_SUFFIX:?}" "${CUDA_VERSION:?}"
  curl --retry 5 --retry-all-errors -fsSL -o /etc/yum.repos.d/cuda-rhel9.repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
  packages+=("cuda-nvcc-${CUDA_PACKAGE_SUFFIX}" "cuda-cudart-devel-${CUDA_PACKAGE_SUFFIX}"
    "libcublas-devel-${CUDA_PACKAGE_SUFFIX}" "cuda-driver-devel-${CUDA_PACKAGE_SUFFIX}"
    "cuda-nvprune-${CUDA_PACKAGE_SUFFIX}" "cuda-nvrtc-devel-${CUDA_PACKAGE_SUFFIX}"
    "libcurand-devel-${CUDA_PACKAGE_SUFFIX}")
fi
dnf install -y "${packages[@]}"
"${py}/python" -m pip install -q cmake ninja build wheel

curl --proto '=https' --tlsv1.2 --retry 5 --retry-all-errors -fsSL https://sh.rustup.rs -o /tmp/rustup.sh
sh /tmp/rustup.sh -y --profile minimal --default-toolchain "${RUSTC_VERSION}"
export PATH="${HOME}/.cargo/bin:${py}:${PATH}"
test "$(rustc --version | cut -d' ' -f2)" = "${RUSTC_VERSION}"
bash "${WORK}/tools/wheel-build/install_cargo_about.sh" /usr/local/bin

src=/tmp/tei
git clone -q --depth 1 --branch "${TEI_TAG}" "${TEI_REPO}" "${src}"
# The wheel build, the retag and auditwheel stamp every entry with the commit time of the source.
export "$(bash "${WORK}/tools/wheel-build/source_date_epoch.sh" "${src}")"

features=candle,http
if [ -n "${CUDA_CC}" ]; then
  cuda_dir="/usr/local/cuda-${CUDA_PACKAGE_SUFFIX//-/.}"
  ln -sfn "${cuda_dir}" /usr/local/cuda
  [ -e "${cuda_dir}/lib64" ] || ln -s targets/x86_64-linux/lib "${cuda_dir}/lib64"
  export PATH="${cuda_dir}/bin:${PATH}"
  export CUDA_PATH="${cuda_dir}" CUDA_COMPUTE_CAP="${CUDA_CC}"
  # The linker needs libcuda to link against; the stub does that, and the GPU driver supplies the real file.
  export LIBRARY_PATH="${cuda_dir}/lib64/stubs:${cuda_dir}/lib64"
  gcc13=/opt/rh/gcc-toolset-13/root/usr/bin
  export CC="${gcc13}/gcc" CXX="${gcc13}/g++" NVCC_CCBIN="${gcc13}/g++"
  # The router is a position-independent executable; the flash attention version 1 kernels are built without -fPIC.
  export NVCC_APPEND_FLAGS="-Xcompiler -fPIC"
  nvcc --version | tail -2
  # Keep the cuBLAS code of this architecture only, as the upstream Dockerfile-cuda does.
  case "${CUDA_CC}" in
    7[5-9]) codes=(sm_"${CUDA_CC}") ;;
    8[0-9]) codes=(sm_80 sm_"${CUDA_CC}") ;;
    90 | 100 | 120) codes=(sm_"${CUDA_CC}") ;;
    *) echo "compute capability ${CUDA_CC} has no build" >&2; exit 1 ;;
  esac
  args=()
  for code in $(printf '%s\n' "${codes[@]}" | sort -u); do args+=(--generate-code "code=${code}"); done
  nvprune "${args[@]}" "${cuda_dir}/lib64/libcublas_static.a" -o /tmp/libcublas_static.a
  cp /tmp/libcublas_static.a "${cuda_dir}/lib64/libcublas_static.a"
  # Flash attention version 2 needs compute capability 8.0; Turing builds use version 1.
  case "${CUDA_CC}" in 7[5-9]) features=candle-cuda-turing,static-linking,http ;; *) features=candle-cuda,static-linking,http ;; esac
fi

# The upstream .cargo/config.toml builds for the CPU of the build host (target-cpu=native), and a
# runner with AVX-512 then yields a router that stops with SIGILL on a CPU without it. RUSTFLAGS
# replaces that setting. The x86-64-v2 baseline needs SSE4.2 and nothing newer.
export RUSTFLAGS="-C target-cpu=x86-64-v2"

build_log=/tmp/cargo-build.log
(cd "${src}" && cargo build -vv --release --locked --bin text-embeddings-router --no-default-features \
  --features "${features}" --jobs "${jobs}") 2>&1 | tee "${build_log}"
binary="${src}/target/release/text-embeddings-router"
test -x "${binary}"
if [ -n "${CUDA_CC}" ]; then
  # The notice names the NVIDIA libraries that the build linked, and stops on one that Attachment A does not list.
  "${py}/python" "${WORK}/tools/wheel-build/write_cuda_notice.py" "${WORK}/packaging/engine-wheel-tei/NOTICE-NVIDIA-CUDA" \
    "${CUDA_VERSION}" "${build_log}"
fi
"${binary}" --version || true

mkdir -p "${OUT}"
export WHEEL_VERSION OPENSSL_LICENSE=/usr/share/licenses/openssl-libs/LICENSE.txt PYTHON="${py}/python"
bash "${WORK}/tools/wheel-build/tei_wheel.sh" "${src}" "${binary}" "${features}" x86_64-unknown-linux-gnu \
  "${BACKEND}" linux_x86_64 "${OUT}"

# The router loads libcuda from the GPU driver; every other library is in the image or grafted.
exclude=()
[ -z "${CUDA_CC}" ] || exclude=(--exclude libcuda.so.1)
auditwheel repair --plat manylinux_2_34_x86_64 --disable-isa-ext-check ${exclude[@]+"${exclude[@]}"} \
  -w "${OUT}/repaired" "${OUT}"/dist/*.whl

if [ -n "${CUDA_CC}" ]; then
  mkdir -p "${OUT}/stubs"
  cp "${cuda_dir}/lib64/stubs/libcuda.so" "${OUT}/stubs/libcuda.so.1"
fi
chown -R "${OWNER}" "${OUT}" "${WORK}/packaging/engine-wheel-tei"
