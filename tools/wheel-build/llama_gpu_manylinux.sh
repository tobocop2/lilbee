#!/usr/bin/env bash
# Build the Linux Vulkan or ROCm llama.cpp pack inside the manylinux_2_34 image (AlmaLinux 9,
# glibc 2.34). The upstream Linux assets need glibc 2.38 or libstdc++ 3.4.30, so this builds
# from the upstream tag. Runs as root in the container, with the repository mounted at /work.
#
# Reads:
#   BACKEND                       vulkan or rocm
#   UPSTREAM_REPO, UPSTREAM_TAG   the llama.cpp source to build
#   BIN_DIR                       the wheel bin/ directory (bind mounted)
#   OWNER                         uid:gid that owns the files left behind
#   LICENSE_DIR                   rocm only: where the ROCm licence texts go
# Writes BIN_DIR/* and, for rocm, LICENSE_DIR/LICENSE-ROCm-*.
#
# Both builds use GGML_BACKEND_DL and CPU_ALL_VARIANTS with OpenMP off, as the CPU pack does.
# The Vulkan loader and the six ROCm host libraries (see assert_rocm_host_libs.sh) stay with
# the host. The last step fails when any shipped file needs glibc above 2.34 or libstdc++
# above 3.4.29.

set -euo pipefail

: "${BACKEND:?}" "${UPSTREAM_REPO:?}" "${UPSTREAM_TAG:?}" "${BIN_DIR:?}" "${OWNER:?}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${script_dir}/../../engine-versions.env"
jobs="$(nproc)"

df -h / | tail -1
dnf install -y gcc-toolset-13-gcc-c++ xz
/opt/python/cp312-cp312/bin/python -m pip install -q cmake ninja
export PATH="/opt/python/cp312-cp312/bin:${PATH}"
gcc13=/opt/rh/gcc-toolset-13/root/usr/bin
"${gcc13}/g++" --version | head -1

backend_args=()
case "${BACKEND}" in
  vulkan)
    # The LunarG SDK carries glslc and the SPIR-V headers that ggml-vulkan needs.
    sdk_dir=/opt/vulkan
    mkdir -p "${sdk_dir}"
    curl -fsSL "https://sdk.lunarg.com/sdk/download/${ENGINE_VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${ENGINE_VULKAN_SDK_VERSION}.tar.xz" \
      | tar -xJ -C "${sdk_dir}" --strip-components=1
    export VULKAN_SDK="${sdk_dir}/x86_64"
    export PATH="${VULKAN_SDK}/bin:${PATH}"
    glslc --version | head -1
    backend_args=(-DGGML_VULKAN=ON -DGGML_HIP=OFF)
    ;;
  rocm)
    cat > /etc/yum.repos.d/rocm.repo <<REPO
[rocm]
name=ROCm ${ENGINE_ROCM_VERSION}
baseurl=https://repo.radeon.com/rocm/rhel9/${ENGINE_ROCM_VERSION%.*}/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
REPO
    # hipcc and rocm-llvm compile the device code. The -devel packages carry the cmake
    # configs that ggml looks up. hipblaslt is a runtime dependency of rocblas.
    dnf install -y --setopt=install_weak_deps=False \
      hipcc rocm-llvm rocm-device-libs hip-devel hipblas-devel rocblas-devel hipblaslt
    rpm -qa --qf '%{SIZE} %{NAME}-%{VERSION}-%{RELEASE}\n' | sort -n | tail -40
    df -h / | tail -1
    # The wheel ships the licence texts of the packages whose libraries it carries.
    : "${LICENSE_DIR:?}"
    for pkg in hsa-rocr comgr hip-runtime-amd rocblas hipblas hipblaslt rocm-llvm; do
      rpm -q --qf "${pkg}: %{LICENSE}\n" "${pkg}"
      out="${LICENSE_DIR}/LICENSE-ROCm-${pkg}"
      : > "${out}"
      while IFS= read -r f; do
        [ -f "${f}" ] || continue
        { echo "== ${f}"; cat "${f}"; echo; } >> "${out}"
      done < <(rpm -ql "${pkg}" | grep -E '/share/doc/[^/]+/LICENSE[^/]*$')
      [ -s "${out}" ] || { echo "${pkg} ships no licence file" >&2; exit 1; }
      chown "${OWNER}" "${out}"
    done
    export ROCM_PATH="/opt/rocm-${ENGINE_ROCM_VERSION}"
    [ -d "${ROCM_PATH}" ] || { echo "no ROCm install at ${ROCM_PATH}" >&2; exit 1; }
    export HIP_PATH="${ROCM_PATH}"
    # The same filter as the extra: only targets with device code and GEMM kernels.
    # shellcheck disable=SC2046  # the argument string splits into words on purpose
    rocm_targets="$(printf '%s\n' $(BACKEND=rocm RUNNER_OS=Linux "${script_dir}/cmake_args.sh" --print) \
      | sed -n 's/^-DAMDGPU_TARGETS=//p')"
    echo "gfx targets: ${rocm_targets}"
    # The HIP compiler reads the GCC 13 headers and libstdc++, so the module keeps the floor.
    gcc13_dir="$(ls -d /opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/*)"
    backend_args=(-DGGML_HIP=ON -DGGML_VULKAN=OFF "-DGPU_TARGETS=${rocm_targets}"
      "-DCMAKE_HIP_COMPILER=${ROCM_PATH}/lib/llvm/bin/clang++"
      "-DCMAKE_HIP_FLAGS=--gcc-install-dir=${gcc13_dir}")
    ;;
  *) echo "unknown BACKEND ${BACKEND}" >&2; exit 1 ;;
esac

src=/tmp/llama.cpp
git clone -q --depth 1 --branch "${UPSTREAM_TAG}" "https://github.com/${UPSTREAM_REPO}.git" "${src}"

cmake -S "${src}" -B "${src}/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${gcc13}/gcc" -DCMAKE_CXX_COMPILER="${gcc13}/g++" \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_OPENMP=OFF \
  -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF "${backend_args[@]}"
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

# The first-party files find each other beside the binary.
for f in "${BIN_DIR}"/llama-server "${BIN_DIR}"/libggml* "${BIN_DIR}"/libllama* "${BIN_DIR}"/libmtmd*; do
  [ -e "${f}" ] || continue
  rpath="$(patchelf --print-rpath "${f}")"
  # shellcheck disable=SC2016  # $ORIGIN is a literal for the dynamic loader
  [ "${rpath}" = '$ORIGIN' ] || patchelf --set-rpath '$ORIGIN' "${f}"
done

case "${BACKEND}" in
  vulkan)
    test -f "${BIN_DIR}/libggml-vulkan.so"
    if find "${BIN_DIR}" -name 'libvulkan*' | grep .; then echo "the Vulkan loader must stay with the host" >&2; exit 1; fi
    ;;
  rocm)
    test -f "${BIN_DIR}/libggml-hip.so"
    ROCM_TARGETS="${rocm_targets}" "${script_dir}/bundle_rocm_runtime.sh" "${BIN_DIR}"
    ;;
esac
test "$(find "${BIN_DIR}" -name 'libggml-cpu-*.so' | wc -l)" -ge 2

"${script_dir}/check_glibc_floor.sh" "${BIN_DIR}"
if ldd "${BIN_DIR}/llama-server" | grep 'not found'; then echo "llama-server has an unresolved library" >&2; exit 1; fi
du -sh "${BIN_DIR}"
df -h / | tail -1

chown -R "${OWNER}" "${BIN_DIR}"
