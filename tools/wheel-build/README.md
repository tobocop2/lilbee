# Wheel build scripts

Top-level scripts called by `.github/workflows/build-wheels.yml`. Each script
runs locally (`bash tools/wheel-build/<script>.sh`) and in CI with the same
behavior, so a developer can reproduce a CI build off-runner.

## What lives here

- `cmake_args.sh` — emits the per-OS / per-backend `CMAKE_ARGS` string for an
  llama-cpp-python source build. Single source of truth for compile flags.
- `install_gpu_toolkit.sh` — installs the build-time GPU SDK on the runner
  (Vulkan SDK on Linux/Windows; CUDA Toolkit when `BACKEND=cuXXX`; no-op on
  macOS where Metal ships with the OS).
- `install_gpu_runtime.sh` — installs only the runtime loader needed to
  `import llama_cpp` (Vulkan loader, CUDA driver shim). Used by the
  verify-pypi job and self-check smoke.
- `build_llama_cpp.sh` — runs `pip wheel llama-cpp-python==<version>` from
  source with the right `CMAKE_ARGS` for the requested backend. Output goes
  to `${LLAMA_BUILD_DIR}` (defaults to `/tmp/llama-build`).
- `fetch_llama_cpp.sh` — alternative to `build_llama_cpp.sh`: downloads the
  prebuilt wheel from abetlen's index for backends abetlen ships (cpu,
  cu121–cu124, metal). Used to avoid rebuilding wheels that already exist.

## Environment contract

Scripts read these env vars:

- `LLAMA_CPP_VERSION` — exact version, e.g. `0.3.19`.
- `BACKEND` — one of `cpu`, `vulkan`, `metal`, `cu121`, `cu122`, `cu123`,
  `cu124`, `rocm`, `sycl`.
- `LLAMA_BUILD_DIR` — output directory for the built/fetched wheel.
- `RUNNER_OS` — `Linux`, `macOS`, or `Windows`. (CI sets this; locally,
  scripts auto-detect.)

A script that doesn't make sense for the requested OS+backend combination
(`metal` on Linux, for example) exits with a non-zero status and a clear
error.
