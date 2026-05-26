# opencode per-model demo reel

Each `demos/opencode-<model>.gif` shows the real opencode TUI driving a lilbee-hosted
model through an agentic dev task: opencode talks to the model over an OpenAI-compatible
endpoint for inference and to lilbee over MCP for `lilbee_search`, and the model searches
lilbee's own source to do the work. Same dynamic as the godot demos, one model per gif.

- `giant_demo.sh` — stands up the model on `llama-server --jinja` (native tool calling)
  plus `lilbee serve` (the `lilbee_search` MCP tool) and writes opencode's config.
- `giant_demo.tape.tmpl` — the VHS template each `demos/opencode-<model>.tape` is rendered
  from (rose-pine, 1600x1000).

The QA side of this reel (per-model tool-call verdicts, the single- vs multi-GPU
placement findings) lives with the harness in the main repo under
`tools/qa/llama_server_toolcall/QA_MATRIX.md`.
