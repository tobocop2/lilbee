#!/usr/bin/env bash
# Build the Linux CUDA llama.cpp pack inside the manylinux_2_34 image (AlmaLinux 9, glibc 2.34).
# Upstream ships Linux CUDA only for 12.8 and 13.4, built on glibc 2.38, so this builds
# from the upstream tag with CUDA 12.4 instead. Runs as root in the container.
#
# Reads:
#   UPSTREAM_REPO, UPSTREAM_TAG   the llama.cpp source to build
#   BIN_DIR                       the wheel bin/ directory (bind mounted)
#   OWNER                         uid:gid that owns the files left behind
#   SHA_FILE                      where to write the sha256 of the three toolkit libraries
# Writes BIN_DIR/* and SHA_FILE (the byte check compares the wheel against it).
#
# The NVIDIA files are copied as they are: no strip, no patchelf, no auditwheel graft (EULA 2.3).

set -euo pipefail

: "${UPSTREAM_REPO:?}" "${UPSTREAM_TAG:?}" "${BIN_DIR:?}" "${OWNER:?}" "${SHA_FILE:?}"
cuda_dash=12-4
cuda_dir=/usr/local/cuda-12.4
jobs="$(nproc)"

curl -fsSL -o /etc/yum.repos.d/cuda-rhel9.repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
# nvcc 12.4 accepts gcc 13 at most; the image default is newer.
dnf install -y "cuda-nvcc-${cuda_dash}" "cuda-cudart-devel-${cuda_dash}" "libcublas-devel-${cuda_dash}" \
  "cuda-driver-devel-${cuda_dash}" \
  gcc-toolset-13-gcc-c++
/opt/python/cp312-cp312/bin/python -m pip install -q cmake ninja
export PATH="/opt/python/cp312-cp312/bin:${cuda_dir}/bin:${PATH}"
gcc13=/opt/rh/gcc-toolset-13/root/usr/bin
"${gcc13}/g++" --version | head -1
nvcc --version | tail -2

src=/tmp/llama.cpp
git clone -q --depth 1 --branch "${UPSTREAM_TAG}" "https://github.com/${UPSTREAM_REPO}.git" "${src}"

# CMAKE_CUDA_ARCHITECTURES is left to the upstream default for CUDA 12.4:
# 50 61 70 75 80 89 90 as virtual, 86 and 89 as real.
cmake -S "${src}" -B "${src}/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${gcc13}/gcc" -DCMAKE_CXX_COMPILER="${gcc13}/g++" \
  -DCMAKE_CUDA_COMPILER="${cuda_dir}/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="${gcc13}/g++" \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_OPENMP=OFF \
  -DGGML_CUDA=ON -DGGML_CUDA_CCCL_VERSION=v3.4.3 \
  -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF
cmake --build "${src}/build" --parallel "${jobs}"

mkdir -p "${BIN_DIR}"
# Copy each library once under the name its dependents load (its SONAME). A backend
# module has no SONAME and keeps its file name, because ggml loads it by that name.
copy_one() {
  local real name
  real="$(readlink -f "$1")"
  name="$(patchelf --print-soname "${real}")"
  [ -n "${name}" ] || name="$(basename "${real}")"
  if [ -e "${BIN_DIR}/${name}" ] && ! cmp -s "${real}" "${BIN_DIR}/${name}"; then
    echo "two different files install as ${name}" >&2
    exit 1
  fi
  cp "${real}" "${BIN_DIR}/${name}"
}
cp "${src}/build/bin/llama-server" "${BIN_DIR}/llama-server"
while IFS= read -r -d '' lib; do copy_one "${lib}"; done \
  < <(find "${src}/build/bin" \( -type f -o -type l \) \( -name '*.so' -o -name '*.so.*' \) -print0)

# The first-party files find each other beside the binary. The NVIDIA files stay as they are.
for f in "${BIN_DIR}"/llama-server "${BIN_DIR}"/libggml* "${BIN_DIR}"/libllama* "${BIN_DIR}"/libmtmd*; do
  [ -e "${f}" ] || continue
  rpath="$(patchelf --print-rpath "${f}")"
  # shellcheck disable=SC2016  # $ORIGIN is a literal for the dynamic loader
  [ "${rpath}" = '$ORIGIN' ] || patchelf --set-rpath '$ORIGIN' "${f}"
done

for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
  cp -L "${cuda_dir}/targets/x86_64-linux/lib/${lib}" "${BIN_DIR}/${lib}"
done
(cd "${cuda_dir}/targets/x86_64-linux/lib" && sha256sum -b libcudart.so.12 libcublas.so.12 libcublasLt.so.12) \
  > "${SHA_FILE}"

# glibc and libstdc++ the wheel needs: 2.34 is the floor of the manylinux_2_34 tag.
worst=0
for f in "${BIN_DIR}"/*; do
  max="$(objdump -T "${f}" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1 || true)"
  [ -z "${max}" ] || echo "${f##*/} glibc ${max}"
  if [ -n "${max}" ] && [ "$(printf '%s\n2.34\n' "${max}" | sort -V | tail -1)" != "2.34" ]; then worst=1; fi
done
[ "${worst}" -eq 0 ] || { echo "a file needs glibc newer than 2.34" >&2; exit 1; }
readelf -d "${BIN_DIR}/llama-server" | grep -E 'NEEDED|RUNPATH|RPATH'
readelf -d "${BIN_DIR}/libggml-cuda.so" | grep -E 'NEEDED|RUNPATH|RPATH'
if ldd "${BIN_DIR}/llama-server" | grep 'not found'; then echo "llama-server has an unresolved library" >&2; exit 1; fi
find "${BIN_DIR}" -name "libggml-cpu-*.so" | wc -l

chown -R "${OWNER}" "${BIN_DIR}" "${SHA_FILE}"
