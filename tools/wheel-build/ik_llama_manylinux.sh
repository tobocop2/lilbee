#!/usr/bin/env bash
# Build the Linux ik_llama.cpp server pack inside the manylinux_2_34 image (AlmaLinux 9, glibc 2.34).
# The upstream has no release assets, so this builds the pinned commit. Runs as root in the container.
#
# Reads:
#   BACKEND    cpu or cuda
#   SRC        the checkout at the pinned commit (bind mounted; the licence step reads the same tree)
#   BIN_DIR    the wheel bin/ directory (bind mounted)
#   OWNER      uid:gid that owns the files left behind
#   SHA_FILE   cuda only: where to write the sha256 of the three toolkit libraries
#   DEPS_DIR   cuda only: where to copy the licence text of the toolkit code that the build compiles in
#   BUILD_DIR  where to build (default /tmp/build-<backend>); a bind mounted folder keeps the objects
#   BUILD_BUDGET_MIN  minutes the build may run; at the limit the script exits with 75
#   STUB_DIR   cuda only: where to copy the stub of the driver library, for a start on a host with no GPU
# Writes BIN_DIR/llama-server, plus the three NVIDIA runtime libraries beside it for the cuda build.
#
# The server links ggml statically, so BIN_DIR holds one binary. A cuda build adds only the files
# that the NVIDIA EULA lists for redistribution; they are copied as they are: no strip, no patchelf (EULA 2.3).

set -euo pipefail

: "${BACKEND:?}" "${SRC:?}" "${BIN_DIR:?}" "${OWNER:?}"
case "${BACKEND}" in cpu | cuda) ;; *) echo "BACKEND must be cpu or cuda, not ${BACKEND}" >&2; exit 2 ;; esac
cuda_dash=12-4
cuda_dir=/usr/local/cuda-12.4
jobs="$(nproc)"

dnf install -y gcc-toolset-13-gcc-c++ patchelf
/opt/python/cp312-cp312/bin/python -m pip install -q cmake ninja
export PATH="/opt/python/cp312-cp312/bin:${PATH}"
gcc13=/opt/rh/gcc-toolset-13/root/usr/bin
"${gcc13}/g++" --version | head -1

args=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_C_COMPILER="${gcc13}/gcc" -DCMAKE_CXX_COMPILER="${gcc13}/g++"
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=OFF)
if [ "${BACKEND}" = cuda ]; then
  curl -fsSL -o /etc/yum.repos.d/cuda-rhel9.repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
  dnf install -y "cuda-nvcc-${cuda_dash}" "cuda-cudart-devel-${cuda_dash}" "libcublas-devel-${cuda_dash}" \
    "cuda-driver-devel-${cuda_dash}" "cuda-cccl-${cuda_dash}"
  export PATH="${cuda_dir}/bin:${PATH}"
  nvcc --version | tail -2
  # Turing is the oldest architecture (compute 7.5). The server links every kernel image into one file,
  # and five images pass the 2 GB limit of the linker. Code for 8.0 runs on 8.6 and 8.9 cards, and
  # Hopper compiles from the 7.5 PTX at its first start.
  args+=(-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="${cuda_dir}/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="${gcc13}/g++"
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS:-75-real;80-real;89-real;75-virtual}")
  if [ -n "${EXTRA_CMAKE:-}" ]; then args+=("${EXTRA_CMAKE}"); fi
  if [ -n "${CUDA_FLAGS:-}" ]; then args+=("-DCMAKE_CUDA_FLAGS=${CUDA_FLAGS}"); fi
  if [ -n "${HOST_FLAGS:-}" ]; then args+=("-DCMAKE_CXX_FLAGS=${HOST_FLAGS}" "-DCMAKE_C_FLAGS=${HOST_FLAGS}"); fi
fi

build="${BUILD_DIR:-/tmp/build-${BACKEND}}"
# The container runs as root; whatever the build leaves in a bind mounted folder returns to its owner.
trap 'chown -R "${OWNER}" "${build}" 2>/dev/null || true' EXIT
cmake -S "${SRC}" -B "${build}" "${args[@]}"
# A runner has four cores and a job has a time limit. With a budget, the build stops in time and
# leaves its objects in BUILD_DIR; the next run continues from them.
rc=0
if [ "${BUILD_BUDGET_MIN:-0}" -gt 0 ]; then
  timeout "${BUILD_BUDGET_MIN}m" cmake --build "${build}" --parallel "${jobs}" --target llama-server || rc=$?
else
  cmake --build "${build}" --parallel "${jobs}" --target llama-server || rc=$?
fi
if [ "${rc}" -eq 124 ]; then
  chown -R "${OWNER}" "${build}"
  echo "the build ran out of its ${BUILD_BUDGET_MIN} minutes; its objects stay in ${build}" >&2
  exit 75
fi
[ "${rc}" -eq 0 ] || exit "${rc}"

mkdir -p "${BIN_DIR}"
server="$(find "${build}" -name llama-server -type f -perm -u+x | head -1)"
test -n "${server}" || { echo "llama-server was not built" >&2; exit 1; }
cp "${server}" "${BIN_DIR}/llama-server"

if [ "${BACKEND}" = cuda ]; then
  # The server finds the NVIDIA files beside itself.
  # shellcheck disable=SC2016  # $ORIGIN is a literal for the dynamic loader
  patchelf --set-rpath '$ORIGIN' "${BIN_DIR}/llama-server"
  : "${SHA_FILE:?}" "${DEPS_DIR:?}"
  for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
    cp -L "${cuda_dir}/targets/x86_64-linux/lib/${lib}" "${BIN_DIR}/${lib}"
  done
  (cd "${cuda_dir}/targets/x86_64-linux/lib" && sha256sum -b libcudart.so.12 libcublas.so.12 libcublasLt.so.12) \
    > "${SHA_FILE}"
  # A runner has no driver. The toolkit stub stands in for libcuda.so.1, so the server can start there.
  : "${STUB_DIR:?}"
  mkdir -p "${STUB_DIR}"
  cp -L "${cuda_dir}/targets/x86_64-linux/lib/stubs/libcuda.so" "${STUB_DIR}/libcuda.so.1"
  chown -R "${OWNER}" "${STUB_DIR}"
  # The build compiles in the CUB headers of the toolkit; their licence text goes into the wheel.
  mkdir -p "${DEPS_DIR}/cub-src"
  find "${cuda_dir}/include" -ipath '*cub*' \( -iname 'LICENSE*' -o -iname 'COPYING*' \) -type f \
    -exec cp {} "${DEPS_DIR}/cub-src/" \;
  chown -R "${OWNER}" "${DEPS_DIR}"
fi

# glibc and libstdc++ symbols the server needs: 2.34 is the floor of the manylinux_2_34 tag.
readelf -d "${BIN_DIR}/llama-server" | grep -E 'NEEDED|RUNPATH|RPATH'
# The host driver (libcuda.so.1) is the only library that may stay unresolved in the build container.
unresolved="$(ldd "${BIN_DIR}/llama-server" | grep 'not found' || true)"
printf '%s\n' "${unresolved}" | grep -v 'libcuda\.so\.1' | grep . && { echo "llama-server has an unresolved library" >&2; exit 1; }

chown -R "${OWNER}" "${BIN_DIR}"
ls -l "${BIN_DIR}"
