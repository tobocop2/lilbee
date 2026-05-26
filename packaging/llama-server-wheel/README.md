# lilbee-llama-server

A per-platform wheel that bundles the compiled llama.cpp `llama-server` binary,
pulled by the `lilbee[multi-gpu]` extra. Kept separate from `lilbee` so the heavy
binary ships only to users who opt into the multi-GPU fleet.

## How it is built (CI)

`.github/workflows/build-multigpu.yml` is this artifact's own, supplementary build
path. It is fully decoupled from the default-wheel / extra-wheel / executable
release jobs and is `continue-on-error`, so a failure here never blocks a release.
Per backend cell (CUDA / Vulkan / ROCm / Metal / cpu):

1. `tools/wheel-build/build_llama_server.sh` compiles `llama-server` from the same
   pinned llama.cpp source `build_llama_cpp.sh` uses (matched to the
   `llama-cpp-python` version), with the per-backend flags from `cmake_args.sh`
   (SSL/CURL off, since the fleet only talks to localhost sidecars).
2. The binary is copied into `lilbee_llama_server/bin/`; the wheel version is set
   to the lilbee version, the wheel is built and retagged to the platform.
3. Two artifacts are uploaded on their own paths: the platform-tagged wheel
   (`wheel-multigpu-*`) and the standalone binary (`multigpu-exe-*`).
4. `attach-multigpu` in `release-candidate.yml` attaches both to the GH release
   (wheels get a backend build tag so same-platform variants coexist).

Once published, `lilbee`'s `pyproject.toml` gains
`multi-gpu = ["lilbee-llama-server"]`, and CI test jobs switch from
`uv sync --all-extras` to enumerated extras so the heavy wheel is not pulled into
every run.

The standalone binary path exists for channels that cannot `pip install` the
extra (frozen exe, Docker, BYO): download the matching `llama-server-*` asset and
point `LILBEE_LLAMA_SERVER_PATH` at it.

At runtime, `lilbee.providers.multi_gpu.binary.resolve_llama_server_binary()`
calls `lilbee_llama_server.get_binary_path()`; if the extra is not installed it
falls back to `LILBEE_LLAMA_SERVER_PATH` / `PATH`.
