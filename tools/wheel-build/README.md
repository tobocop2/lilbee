# Wheel build scripts

Top-level scripts called by `build-multigpu.yml` (the engine wheel),
`build-gpu-executables.yml`, and `release.yml` (the standalone executables).
Each script runs locally (`bash tools/wheel-build/<script>.sh`) and in CI with
the same behavior, so a developer can reproduce a CI build off-runner.

## What lives here

- `cmake_args.sh` — emits the per-OS / per-backend `CMAKE_ARGS` string (the
  ggml compile flags). Single source of truth for compile flags.
- `install_gpu_toolkit.sh` — installs the build-time GPU SDK on the runner
  (Vulkan SDK on Linux/Windows; CUDA Toolkit when `BACKEND=cuXXX`; ROCm when
  `BACKEND=rocm`; no-op on macOS where Metal ships with the OS).
- `build_llama_server.sh` — builds the self-contained `llama-server` (binary +
  ggml/llama/mtmd libs with a baked rpath) into the `lilbee-engine-llama-cpp` wheel
  package's `bin/` for the requested backend.
- `write_engine_pin.sh` — writes the wheel's engine pin: the source versions and
  the backend. The step that builds or installs the wheel runs it, because a
  mirror or cache hit skips `build_llama_server.sh`.
- `bundle_rocm_runtime.sh` — packs the ROCm userspace `llama-server` links
  (discovered by walking `DT_NEEDED`, plus the rocBLAS Tensile kernels) beside
  the binary, so an AMD user needs only a kernel driver and six host libraries
  (libnuma, libelf, libdrm, libdrm_amdgpu, zlib, zstd; the script bundles none
  of them). Called by `build_llama_server.sh` for the rocm backend; tested on its
  own.
- `llama_gpu_manylinux.sh` — builds the Linux CPU, Vulkan or ROCm engine pack in the
  manylinux_2_34 image for the build workflow of the engine-packs repository.
- `pack_mistralrs.sh` — builds and checks the mistral.rs engine pack of one cell. The
  engine-packs repository runs it through its `scripts/pack.sh`, which sets the variables
  the script names at its top.
- `check_mistralrs_wheel.sh` — checks a built mistral.rs pack: the tag, the macOS floor of
  the Metal libraries (`metal_floor.py`), the licences, that each DLL and exe of a Windows
  wheel imports only a DLL that the wheel holds or Windows provides
  (`tools/qa/assert_engine_bundle.py --imports-only`), and that the tool starts.
  `pack_mistralrs.sh` runs it before the pack leaves the build.
- `pack_tei.sh` — builds and checks the text-embeddings-inference engine pack of one
  cell, in the same way. A Linux cell builds in the manylinux image (`tei_manylinux.sh`),
  the macOS cell on the Mac. `check_tei_wheel.py` checks the wheel and `smoke_tei.sh` starts
  its router.
- `pack_ik.sh` — builds and checks the ik_llama.cpp engine pack of one cell, in the same
  way. The server builds in the manylinux image (`ik_llama_manylinux.sh`), and
  `check_ik_wheel.py` checks the notices and the licence expression of the wheel.
- `pack_llama.sh` — builds the llama.cpp engine pack of one cell, in the same way. A Linux
  cell builds the server in the manylinux image (`llama_gpu_manylinux.sh`,
  `llama_cuda_manylinux.sh`). `repack_llama_archives.py` repacks the upstream archives of
  the other cells, and `llama_cuda_notice.py` writes the NVIDIA notice.
- `check_llama_wheel.sh` — installs a built llama.cpp pack, checks its files, licences and
  libraries for the backend and the host, and starts the server. `pack_llama.sh` runs it
  before the pack leaves the build.
- `pack_mlx.sh` — builds and checks the mlx engine pack of its one macOS cell, in the same
  way. `mlx_wheel.sh` builds the wheel from the pins of `engines/mlx.env`. The script checks
  the tag and the licence expression of the wheel, and starts its chat server.
- `pack_common.sh` — the functions that the pack scripts share.
- `build-tools.constraints` — the pinned Python build tools of the packs.
  `tei-tools-linux.constraints` and `tei-tools-macos.constraints` add the ones that only one
  build of the text-embeddings-inference pack needs.
  `ik-tools.constraints` pins the ones that build the ik_llama.cpp server.
  `mlx-tools-macos.constraints` pins uv and delocate for the mlx pack.
  `llama-tools-linux.constraints`, `llama-tools-macos.constraints` and
  `llama-tools-windows.constraints` do the same for the llama.cpp pack.
  `windows-bundle-tools.constraints` pins pefile, which the Windows packs of mistral.rs and
  llama.cpp use to read import tables. It is not a build tool. A Windows pack prunes each
  bundled Visual C++ runtime DLL that no file of the wheel imports
  (`tools/qa/assert_engine_bundle.py --prune-runtime`).
- `check_glibc_floor.sh` — fails when a file in a pack imports glibc above 2.34
  or libstdc++ above 3.4.29.
- `check_ik_repro.sh` — decides whether two or more CUDA builds of one ik commit
  are equal. Every file must be byte-equal, except `libggml.so`: its host code
  (`.text`, `.rodata`, `.dynsym`, `.dynstr`) must be equal and its GPU code may differ.
- `build_lilbee_binary.sh` — Nuitka one-file build of the lilbee standalone
  executable, bundling the engine wheel built above.

## Environment contract

Scripts read these env vars:

- `LLAMA_CPP_REPO` / `LLAMA_CPP_REF` — the llama.cpp repo and ref to build
  from (optional; the build script defaults to the pin in engine-versions.env).
- `BACKEND` — one of `cpu`, `vulkan`, `metal`, `cu121`, `cu122`, `cu123`,
  `cu124`, `rocm`.
- `LLAMA_BUILD_DIR` — output directory for the built/fetched wheel.
- `RUNNER_OS` — `Linux`, `macOS`, or `Windows`. (CI sets this; locally,
  scripts auto-detect.)

A script that doesn't make sense for the requested OS+backend combination
(`metal` on Linux, for example) exits with a non-zero status and a clear
error.
