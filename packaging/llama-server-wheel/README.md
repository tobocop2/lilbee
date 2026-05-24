# lilbee-llama-server

A per-platform wheel that bundles the compiled llama.cpp `llama-server` binary,
pulled by the `lilbee[multi-gpu]` extra. Kept separate from `lilbee` so the heavy
binary ships only to users who opt into the multi-GPU fleet.

## How it is built (CI)

1. `tools/wheel-build/build_llama_server.sh` compiles `llama-server` from the same
   pinned llama.cpp source `build_llama_cpp.sh` uses (matched to the
   `llama-cpp-python` version), with the per-backend flags from `cmake_args.sh`.
2. The binary is copied into `lilbee_llama_server/bin/`.
3. A platform-tagged wheel is built here and published with the existing wheel
   matrix (one wheel per OS/arch/backend).
4. The version is set to track the lilbee/llama.cpp pin.

Once published, `lilbee`'s `pyproject.toml` gains
`multi-gpu = ["lilbee-llama-server"]`, and CI test jobs switch from
`uv sync --all-extras` to enumerated extras so the heavy wheel is not pulled into
every run.

At runtime, `lilbee.providers.multi_gpu.binary.resolve_llama_server_binary()`
calls `lilbee_llama_server.get_binary_path()`; if the extra is not installed it
falls back to `LILBEE_LLAMA_SERVER_PATH` / `PATH`.
