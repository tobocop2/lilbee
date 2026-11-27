#!/usr/bin/env bash
# Build the Linux CPU, Vulkan, ROCm or OpenVINO llama.cpp pack inside the manylinux_2_34 image
# (AlmaLinux 9, glibc 2.34). The upstream Linux assets need glibc 2.38 or libstdc++ 3.4.30, so this
# builds from the upstream tag. Runs as root in the container, with the repository mounted at /work.
#
# Reads:
#   BACKEND                       cpu, vulkan, rocm or openvino
#   UPSTREAM_REPO, UPSTREAM_TAG   the llama.cpp source to build
#   BIN_DIR                       the wheel bin/ directory (bind mounted)
#   OWNER                         uid:gid that owns the files left behind
#   LICENSE_DIR                   rocm and openvino: where the runtime licence texts go
# Writes BIN_DIR/* and, for rocm, LICENSE_DIR/LICENSE-ROCm-*; for openvino,
# LICENSE_DIR/LICENSE-OpenVINO-Apache-2.0, LICENSE-oneTBB, LICENSE-ocl-icd and THIRD-PARTY/openvino/.
#
# Every build uses GGML_BACKEND_DL and CPU_ALL_VARIANTS with OpenMP off, as the CPU pack does.
# The Vulkan loader and the six ROCm host libraries (see assert_rocm_host_libs.sh) stay with
# the host. The OpenVINO pack ships the runtime that engines/openvino.env pins; the Intel GPU
# compute runtime and the NPU driver stay with the host. The last step fails when any shipped file needs glibc above 2.34 or libstdc++
# above 3.4.29.

set -euo pipefail

: "${BACKEND:?}" "${UPSTREAM_REPO:?}" "${UPSTREAM_TAG:?}" "${BIN_DIR:?}" "${OWNER:?}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${script_dir}/../../engine-versions.env"
# shellcheck source=tools/wheel-build/pack_common.sh
. "${script_dir}/pack_common.sh"
jobs="$(nproc)"
# The Python of the image holds other packages, so the build tools go into a new venv.
venv=/tmp/build-venv
[ ! -e "${venv}" ] || { echo "llama_gpu_manylinux: an earlier build used this container (${venv} exists). Use a new one." >&2; exit 2; }

df -h / | tail -1
dnf install -y gcc-toolset-13-gcc-c++ xz
/opt/python/cp312-cp312/bin/python -m venv "${venv}"
install_build_tools "${venv}/bin/python" "${script_dir}/llama-tools-linux.constraints"
export PATH="${venv}/bin:${PATH}"
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
    : "${LICENSE_DIR:?}"
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
  openvino)
    # shellcheck source=/dev/null
    source "${script_dir}/../../engines/openvino.env"
    : "${LICENSE_DIR:?}"
    # ggml-openvino includes the OpenCL C++ headers and links the OpenCL loader.
    dnf install -y dnf-plugins-core
    dnf config-manager --set-enabled crb
    dnf install -y opencl-headers ocl-icd-devel
    # The PyPI wheel holds the libraries and cache.json in openvino/libs, the cmake configs of
    # OpenVINO and oneTBB in openvino/cmake, and its licence texts in the dist-info.
    fetch_checked() { # <url> <sha256> <file>
      curl -fsSL --retry 3 -o "$3" "$1"
      echo "$2  $3" | sha256sum -c -
    }
    ov_wheel=/tmp/openvino.whl
    fetch_checked "${OPENVINO_WHEEL_URL}" "${OPENVINO_WHEEL_SHA256}" "${ov_wheel}"
    ov_unpack=/opt/openvino-wheel
    mkdir -p "${ov_unpack}"
    python -m zipfile -e "${ov_wheel}" "${ov_unpack}"
    ov_root="${ov_unpack}/openvino"
    ov_dist="${ov_unpack}/openvino-${OPENVINO_VERSION}.dist-info/licenses"
    # The notice and licence of oneTBB come from its own source tree, not from the OpenVINO copy.
    ov_tbb_notice=/tmp/onetbb-third-party-programs.txt
    ov_tbb_licence=/tmp/onetbb-LICENSE.txt
    fetch_checked "${ONETBB_NOTICE_URL}" "${ONETBB_NOTICE_SHA256}" "${ov_tbb_notice}"
    fetch_checked "${ONETBB_LICENSE_URL}" "${ONETBB_LICENSE_SHA256}" "${ov_tbb_licence}"
    export OpenVINO_DIR="${ov_root}/cmake"
    export TBB_DIR="${ov_root}/cmake"
    backend_args=(-DGGML_OPENVINO=ON -DGGML_VULKAN=OFF -DGGML_HIP=OFF)
    ;;
  cpu)
    backend_args=(-DGGML_VULKAN=OFF -DGGML_HIP=OFF)
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
  openvino)
    test -f "${BIN_DIR}/libggml-openvino.so"
    # The runtime, its CPU, GPU and NPU plugins with the plugins that combine them, the IR and
    # GGUF frontends, oneTBB with hwloc, and the OpenCL loader, as the upstream OpenVINO release
    # asset ships them. The model-conversion frontends and the C API are left out: llama.cpp
    # builds its graph in memory and links neither. The Intel GPU compute runtime and the NPU
    # driver stay with the host.
    while IFS= read -r -d '' lib; do copy_one "${lib}"; done < <(
      find "${ov_root}/libs" -maxdepth 1 -type f -name 'lib*.so*' \
        ! -name 'libopenvino_onnx_frontend*' ! -name 'libopenvino_paddle_frontend*' \
        ! -name 'libopenvino_tensorflow*' ! -name 'libopenvino_pytorch_frontend*' \
        ! -name 'libopenvino_jax_frontend*' ! -name 'libopenvino_c.so*' ! -name 'libtbbmalloc_proxy*' -print0)
    cp "${ov_root}/libs/cache.json" "${BIN_DIR}/cache.json"
    copy_one "$(ldconfig -p | awk '$1 == "libOpenCL.so.1" { print $NF; exit }')"
    for f in "${BIN_DIR}"/libopenvino* "${BIN_DIR}"/libtbb* "${BIN_DIR}"/libhwloc* "${BIN_DIR}"/libOpenCL*; do
      # shellcheck disable=SC2016  # $ORIGIN is a literal for the dynamic loader
      patchelf --set-rpath '$ORIGIN' "${f}"
    done
    for plugin in intel_cpu intel_gpu intel_npu auto hetero; do
      test -f "${BIN_DIR}/libopenvino_${plugin}_plugin.so"
    done
    lic="${ov_dist}/licensing"
    cp "${ov_dist}/LICENSE" "${LICENSE_DIR}/LICENSE-OpenVINO-Apache-2.0"
    cp "${ov_tbb_licence}" "${LICENSE_DIR}/LICENSE-oneTBB"
    cp /usr/share/licenses/ocl-icd/COPYING "${LICENSE_DIR}/LICENSE-ocl-icd"
    rpm -q --qf 'ocl-icd: %{LICENSE}\n' ocl-icd
    mkdir -p "${LICENSE_DIR}/THIRD-PARTY/openvino"
    for f in runtime-third-party-programs.txt onednn_third-party-programs.txt; do
      cp "${lic}/${f}" "${LICENSE_DIR}/THIRD-PARTY/openvino/${f}"
    done
    cp "${ov_tbb_notice}" "${LICENSE_DIR}/THIRD-PARTY/openvino/onetbb_third-party-programs.txt"
    chown -R "${OWNER}" "${LICENSE_DIR}"/LICENSE-OpenVINO-* "${LICENSE_DIR}/LICENSE-oneTBB" \
      "${LICENSE_DIR}/LICENSE-ocl-icd" "${LICENSE_DIR}/THIRD-PARTY/openvino"
    ;;
  rocm)
    test -f "${BIN_DIR}/libggml-hip.so"
    ROCM_TARGETS="${rocm_targets}" "${script_dir}/bundle_rocm_runtime.sh" "${BIN_DIR}"
    # The wheel ships the licence texts of the packages that own the libraries it carries.
    packages="$(for f in "${BIN_DIR}"/*; do
      for dir in "${ROCM_PATH}/lib" "${ROCM_PATH}/lib/llvm/lib"; do
        if [ -f "${dir}/${f##*/}" ]; then
          rpm -qf --qf '%{NAME}\n' "$(readlink -f "${dir}/${f##*/}")"
        fi
      done
    done | sort -u)"
    [ -n "${packages}" ] || { echo "no ROCm package owns a shipped library" >&2; exit 1; }
    # libamd_comgr embeds the LLVM compiler, so the licence of rocm-llvm ships with it.
    packages="$(printf '%s\nrocm-llvm\n' "${packages}" | sort -u)"
    for pkg in ${packages}; do
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
    ;;
esac
test "$(find "${BIN_DIR}" -name 'libggml-cpu-*.so' | wc -l)" -ge 2

"${script_dir}/check_glibc_floor.sh" "${BIN_DIR}"
assert_libraries_resolve "${BIN_DIR}/llama-server"
du -sh "${BIN_DIR}"
df -h / | tail -1

chown -R "${OWNER}" "${BIN_DIR}"
