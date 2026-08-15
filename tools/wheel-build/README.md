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
  ggml/llama/mtmd libs with a baked rpath) into the `lilbee-engine` wheel
  package's `bin/` for the requested backend.
- `bundle_rocm_runtime.sh` — packs the ROCm userspace `llama-server` links
  (discovered by walking `DT_NEEDED`, plus the rocBLAS Tensile kernels) beside
  the binary, so an AMD user needs only a kernel driver. Called by
  `build_llama_server.sh` for the rocm backend; tested on its own.
- `build_lilbee_binary.sh` — Nuitka one-file build of the lilbee standalone
  executable, bundling the engine wheel built above.

## Environment contract

Scripts read these env vars:

- `LLAMA_CPP_REPO` / `LLAMA_CPP_REF` — the llama.cpp repo and ref to build
  from (optional; the build script defaults to the pin in engine-versions.env).
- `BACKEND` — one of `cpu`, `vulkan`, `metal`, `cu121`, `cu122`, `cu123`,
  `cu124`, `rocm`, `sycl`.
- `LLAMA_BUILD_DIR` — output directory for the built/fetched wheel.
- `RUNNER_OS` — `Linux`, `macOS`, or `Windows`. (CI sets this; locally,
  scripts auto-detect.)

A script that doesn't make sense for the requested OS+backend combination
(`metal` on Linux, for example) exits with a non-zero status and a clear
error.
