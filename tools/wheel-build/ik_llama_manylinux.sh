#!/usr/bin/env bash
# Build the ik_llama.cpp server pack for Linux x86_64 inside the manylinux_2_34 image
# (AlmaLinux 9, glibc 2.34), as a CPU build or a CUDA 12.4 build. Runs as root in the container.
# The engine has no loadable backends and no CPU variants, so one binary holds one CPU baseline
# (AVX2, FMA, F16C) and, for CUDA, the GPU code of every architecture the pack serves.
#
# Reads:
#   UPSTREAM_REPO, UPSTREAM_COMMIT  the source to build; the commit is a full hash, never a branch
#   BACKEND                         cpu or cuda
#   BIN_DIR                         the wheel bin/ directory (bind mounted)
#   OWNER                           uid:gid that owns the files left behind
#   SHA_FILE                        cuda only: where to write the sha256 of the three toolkit libraries
#   NOTICE_FILE                     cuda only: where to write NOTICE-NVIDIA-CUDA (beside the wheel project)
#   SRC_DIR                         where to leave the source checkout for the licence collector
# Writes BIN_DIR/*, SHA_FILE (the byte check compares the wheel against it), NOTICE_FILE and SRC_DIR.
#
# The NVIDIA files are copied as they are: no strip, no patchelf, no auditwheel graft (EULA 2.3).

set -euo pipefail

: "${UPSTREAM_REPO:?}" "${UPSTREAM_COMMIT:?}" "${BACKEND:?}" "${BIN_DIR:?}" "${OWNER:?}" "${SRC_DIR:?}"
case "${BACKEND}" in
  cpu | cuda) ;;
  *) echo "BACKEND must be cpu or cuda, not ${BACKEND}" >&2; exit 1 ;;
esac
if [ "${BACKEND}" = cuda ]; then : "${SHA_FILE:?}" "${NOTICE_FILE:?}"; fi
cuda_dash=12-4
cuda_full=12.4.1
cuda_dir=/usr/local/cuda-12.4
jobs="$(nproc)"
# shellcheck source=tools/wheel-build/pack_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/pack_common.sh"
# The Python of the image holds other packages, so the build tools go into a new venv.
venv=/tmp/build-venv
[ ! -e "${venv}" ] || { echo "ik_llama_manylinux: an earlier build used this container (${venv} exists). Use a new one." >&2; exit 2; }

packages=(gcc-toolset-13-gcc-c++)
if [ "${BACKEND}" = cuda ]; then
  curl --retry 5 --retry-all-errors -fsSL -o /etc/yum.repos.d/cuda-rhel9.repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
  # nvcc 12.4 accepts gcc 13 at most; the image default is newer.
  packages+=("cuda-nvcc-${cuda_dash}" "cuda-cudart-devel-${cuda_dash}" "libcublas-devel-${cuda_dash}"
    "cuda-driver-devel-${cuda_dash}")
fi
dnf install -y "${packages[@]}"
/opt/python/cp312-cp312/bin/python -m venv "${venv}"
install_build_tools "${venv}/bin/python" "$(dirname "${BASH_SOURCE[0]}")/ik-tools.constraints"
export PATH="${venv}/bin:${PATH}"
gcc13=/opt/rh/gcc-toolset-13/root/usr/bin
"${gcc13}/g++" --version | head -1

mkdir -p "${SRC_DIR}"
git config --global --add safe.directory "${SRC_DIR}"
git -C "${SRC_DIR}" init -q
git -C "${SRC_DIR}" fetch -q --depth 1 "https://github.com/${UPSTREAM_REPO}.git" "${UPSTREAM_COMMIT}"
git -C "${SRC_DIR}" checkout -q FETCH_HEAD
test "$(git -C "${SRC_DIR}" rev-parse HEAD)" = "${UPSTREAM_COMMIT}"
# The engine compiles its GPU code with nvcc -lineinfo, which only helps a profiler. It puts the mtime of each source
# file (the time of the checkout) and a process ID in the symbol names into the GPU code, so two builds of one commit
# differ. nvcc has no flag to undo -lineinfo, so the build removes it from the CMake file and stops if it is still there.
sed -i 's/ -lineinfo)/)/' "${SRC_DIR}/ggml/src/CMakeLists.txt"
if grep -rq -e '-lineinfo' "${SRC_DIR}" --include=CMakeLists.txt --include='*.cmake'; then
  echo "ik_llama_manylinux: -lineinfo is still in a CMake file of the source" >&2
  exit 1
fi
# ACCEPTED DIFFERENCE. Two CUDA builds of one commit are not byte-equal, and nothing needs them to be. The register
# allocation of cicc (the GPU front end of nvcc) gives other code for one or two mmvq kernels from run to run, with
# one thread, one process ID and no address randomization, and no nvcc flag removes it. The rule that holds: llama-server,
# libllama.so, libmtmd.so and the NVIDIA libraries are byte-equal; libggml.so has equal host code (.text, .rodata,
# .dynsym, .dynstr); only its GPU code (.nv_fatbin) and the sections that hold addresses after it (.nvFatBinSegment,
# .eh_frame, .eh_frame_hdr, .rela.dyn) may differ. Wheel digests come from the release asset, never from a rebuild.

# One CPU baseline for every host: AVX2, FMA and F16C, no -march=native. NCCL is off: the engine
# links it when the image has it, and the wheel must not need it.
args=(-G Ninja -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER="${gcc13}/gcc" -DCMAKE_CXX_COMPILER="${gcc13}/g++"
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_NCCL=OFF
  -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF)
if [ "${BACKEND}" = cpu ]; then
  args+=(-DBUILD_SHARED_LIBS=OFF)
else
  export PATH="${cuda_dir}/bin:${PATH}"
  nvcc --version | tail -2
  # Each real architecture adds about 1 GB of GPU code, and the linker refuses more than 2 GB in one
  # library. Code for 8.0 runs on every 8.x card (A100, A10, RTX 30, L4, RTX 40). A card above the
  # real architectures runs the PTX of the highest virtual one (9.0), compiled by the driver at the
  # first start, and the host code of the engine picks its kernels for that same architecture. A
  # virtual architecture below a real one would make the host pick kernels that the PTX lacks.
  args+=(-DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DGGML_OPENMP=OFF
    "-DCMAKE_CUDA_ARCHITECTURES=75-virtual;80-real;90-virtual"
    -DCMAKE_CUDA_COMPILER="${cuda_dir}/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="${gcc13}/g++")
fi
cmake -S "${SRC_DIR}" -B "${SRC_DIR}/build" "${args[@]}"
cmake --build "${SRC_DIR}/build" --parallel "${jobs}" --target llama-server

mkdir -p "${BIN_DIR}"
cp "${SRC_DIR}/build/bin/llama-server" "${BIN_DIR}/llama-server"
# Any shared library the build left beside the server is copied once under the name its
# dependents load (its SONAME).
while IFS= read -r -d '' lib; do
  real="$(readlink -f "${lib}")"
  name="$(patchelf --print-soname "${real}")"
  [ -n "${name}" ] || name="$(basename "${real}")"
  cp "${real}" "${BIN_DIR}/${name}"
done < <(find "${SRC_DIR}/build" \( -type f -o -type l \) \( -name '*.so' -o -name '*.so.*' \) -print0)

# The first-party files find each other beside the binary. The NVIDIA files stay as they are.
# nvcc names its temporary files after its process ID, and the name stays in the symbol table of each GPU object.
# The strip removes it, so two builds of one commit are equal.
for f in "${BIN_DIR}"/llama-server "${BIN_DIR}"/libggml* "${BIN_DIR}"/libllama*; do
  [ -e "${f}" ] || continue
  strip --strip-unneeded "${f}"
  # shellcheck disable=SC2016  # $ORIGIN is a literal for the dynamic loader
  patchelf --set-rpath '$ORIGIN' "${f}"
done

if [ "${BACKEND}" = cuda ]; then
  for lib in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
    cp -L "${cuda_dir}/targets/x86_64-linux/lib/${lib}" "${BIN_DIR}/${lib}"
  done
  (cd "${cuda_dir}/targets/x86_64-linux/lib" && sha256sum -b libcudart.so.12 libcublas.so.12 libcublasLt.so.12) \
    > "${SHA_FILE}"
  # The notice names the libraries just copied and stops on one that Attachment A does not list.
  python "$(dirname "${BASH_SOURCE[0]}")/write_cuda_notice.py" "${NOTICE_FILE}" "${cuda_full}" "${SHA_FILE}"
fi

# glibc the wheel needs: 2.34 is the floor of the manylinux_2_34 tag.
worst=0
for f in "${BIN_DIR}"/*; do
  max="$(objdump -T "${f}" 2>/dev/null | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1 || true)"
  [ -z "${max}" ] || echo "${f##*/} glibc ${max}"
  if [ -n "${max}" ] && [ "$(printf '%s\n2.34\n' "${max}" | sort -V | tail -1)" != "2.34" ]; then worst=1; fi
done
[ "${worst}" -eq 0 ] || { echo "a file needs glibc newer than 2.34" >&2; exit 1; }
first_party=("${BIN_DIR}"/llama-server)
for f in "${BIN_DIR}"/libggml* "${BIN_DIR}"/libllama*; do [ -e "${f}" ] && first_party+=("${f}"); done
for f in "${first_party[@]}"; do
  echo "${f##*/}"
  readelf -d "${f}" | grep -E 'NEEDED|RUNPATH|RPATH'
done
# A wheel carries the backend it claims: the CUDA build names the CUDA libraries and the CPU build does not.
needed="$(readelf -d "${first_party[@]}" | grep NEEDED)"
if [ "${BACKEND}" = cuda ]; then
  grep -q 'libcudart\.so\.12' <<< "${needed}"
  grep -q 'libcublas\.so\.12' <<< "${needed}"
elif grep -q 'libcudart\|libcublas' <<< "${needed}"; then
  echo "the CPU build needs a CUDA library" >&2
  exit 1
fi
# The NVIDIA driver library comes from the host, so the container does not have it.
if ldd "${BIN_DIR}/llama-server" | grep 'not found' | grep -v 'libcuda\.so\.1'; then
  echo "llama-server has an unresolved library" >&2
  exit 1
fi

chown -R "${OWNER}" "${BIN_DIR}" "${SRC_DIR}"
if [ "${BACKEND}" = cuda ]; then chown "${OWNER}" "${SHA_FILE}" "${NOTICE_FILE}"; fi
