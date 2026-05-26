# lilbee-llama-server

A per-platform, self-contained wheel that bundles the compiled llama.cpp
`llama-server` together with its `ggml`/`llama`/`mtmd` shared libraries (rpath
baked, so it carries everything it needs). This is lilbee's local inference
engine; `lilbee` depends on it, so `pip install lilbee` brings it.

## How it is built (CI)

`.github/workflows/build-multigpu.yml` builds it per backend cell (CUDA /
Vulkan / ROCm / Metal / CPU); cells are `continue-on-error` so a slow GPU cell
never holds up a release.

1. `tools/wheel-build/build_llama_server.sh` compiles `llama-server` from a
   pinned llama.cpp source with the per-backend flags from `cmake_args.sh`
   (SSL/CURL off, since the fleet only talks to localhost), bundles the
   `ggml`/`llama`/`mtmd` libs next to the binary, and bakes the rpath
   (`$ORIGIN` / `@loader_path`).
2. `bin/` is filled, the wheel version is set to the lilbee version, and the
   wheel is built and retagged to the platform.
3. The wheel uploads as `wheel-multigpu-*`.

## How it is distributed

- **Default backends** (Vulkan on Linux/Win, Metal on macOS) publish to PyPI, so
  a plain `pip install lilbee` resolves the matching engine.
- **CUDA / ROCm / CPU** variants live on the per-backend PEP 503 index at
  `lilbee.sh/<backend>/`; opt in with `pip install lilbee --extra-index-url
  https://lilbee.sh/<backend>/`.
- The **standalone executables** (brew / Docker / AUR) bundle this same
  self-contained engine via Nuitka, so those channels carry it too.

At runtime, `lilbee.providers.multi_gpu.binary.resolve_llama_server_binary()`
calls `lilbee_llama_server.get_binary_path()`; for a bring-your-own setup it
falls back to `LILBEE_LLAMA_SERVER_PATH` / `PATH`.
