# lilbee-engine-llama-cpp

A per-platform, self-contained wheel that bundles lilbee's local inference
engine binaries:

- `llama-server` (compiled llama.cpp, with its `ggml`/`llama`/`mtmd` shared
  libraries and a baked rpath so it carries everything it needs),
- `llama-swap` (the process supervisor + OpenAI proxy that fronts the role
  servers),
- `gguf-parser` (UMA-aware VRAM estimator used by the placement planner).

The standalone executables bundle this wheel. lilbee has no `engine` extra: a
`pip` or `uv` install downloads an engine pack with `lilbee engine install`.
To install this wheel by hand, run
`pip install lilbee-engine-llama-cpp --extra-index-url https://lilbee.sh/<backend>/`.
It is not on PyPI, since the CUDA and ROCm wheels run several times PyPI's
default 100 MiB per-file limit, so all backends are published from lilbee's own
PEP 503 index. CI fills `lilbee_engine_llama_cpp/bin/` with the three
binaries and retags the wheel to the platform before publishing; a source
checkout has an empty `bin/` and resolves via `PATH` or
`LILBEE_LLAMA_SERVER_PATH` instead.

`lilbee.providers.fleet.binary` resolves each tool via
`lilbee_engine_llama_cpp.get_llama_server_path()` / `get_llama_swap_path()` /
`get_gguf_parser_path()`. For a bring-your-own setup, point `LILBEE_LLAMA_SERVER_PATH`
at a `llama-server` binary and put `llama-swap` / `gguf-parser` on `PATH`.
