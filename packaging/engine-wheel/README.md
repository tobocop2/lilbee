# lilbee-engine

A per-platform, self-contained wheel that bundles lilbee's local inference
engine binaries:

- `llama-server` (compiled llama.cpp, with its `ggml`/`llama`/`mtmd` shared
  libraries and a baked rpath so it carries everything it needs),
- `llama-swap` (the process supervisor + OpenAI proxy that fronts the role
  servers),
- `gguf-parser` (UMA-aware VRAM estimator used by the placement planner).

This is lilbee's core engine dependency. `pip install lilbee` resolves the
matching per-platform wheel from the index; CI fills `lilbee_engine/bin/` with
the three binaries and retags the wheel to the platform before publishing.

`lilbee.providers.fleet.binary` resolves each tool via
`lilbee_engine.get_llama_server_path()` / `get_llama_swap_path()` /
`get_gguf_parser_path()`. For a bring-your-own setup, point `LILBEE_LLAMA_SERVER_PATH`
at a `llama-server` binary and put `llama-swap` / `gguf-parser` on `PATH`.
