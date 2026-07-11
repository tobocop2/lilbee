# Local models for coding agents

`lilbee launch opencode` and `lilbee launch hermes` wire [opencode](https://opencode.ai) and [hermes](https://github.com/NousResearch/hermes-agent) to your local lilbee chat models, so an agent searches your library, reads the results, and answers with citations, all on your machine. The agent talks to lilbee over the OpenAI-compatible API and calls `lilbee_search` through MCP, so a model works here only if it emits tool calls in a format lilbee can read.

Launch with `--no-mcp` to keep lilbee as the model provider but drop its MCP block, leaving your own agent MCP config untouched. The default is the `agent_mcp_enabled` config field (env `LILBEE_AGENT_MCP_ENABLED`); `--mcp` / `--no-mcp` override it per launch.

## Verified families

Each family completes the loop end to end: the agent sends a prompt, the model calls `lilbee_search`, the tool runs against an indexed workspace, and the model answers from the results. How reliably a model reaches for a tool depends on the model and its size, not on lilbee.

| Family | Example model | Notes |
|--------|---------------|-------|
| Qwen3 | `Qwen/Qwen3-4B-GGUF` | |
| Qwen3-Coder | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` | sparse-MoE coder variant |
| Llama 3.1 | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | |
| Hermes 3 | `bartowski/Hermes-3-Llama-3.1-8B-GGUF` | |
| Mistral-Nemo | `bartowski/Mistral-Nemo-Instruct-2407-GGUF` | |
| Gemma | `unsloth/gemma-4-E2B-it-GGUF` | |
| SmolLM3 | `bartowski/HuggingFaceTB_SmolLM3-3B-GGUF` | |
| Cohere Command-R7B | `bartowski/c4ai-command-r7b-12-2024-GGUF` | renders from the HF tokenizer's template; needs the GGUF's wrong 8192 context corrected to the real 128K |
| gpt-oss | `ggml-org/gpt-oss-120b-GGUF` | Harmony tool-call format; the 120B is split across shards and needs a >80 GB GPU |
| GLM-4.5-Air | `unsloth/GLM-4.5-Air-GGUF` | renders from the HF tokenizer's template; split GGUF, needs a >80 GB GPU |

## Not yet supported

These don't work today, because of the model or the bundled runtime, not lilbee's tool plumbing.

| Family | Why |
|--------|-----|
| Functionary v3 | dispatches now that lilbee renders from the HF tokenizer's tool template, but multi-turn is still inconsistent |
| OLMo-3 | the tool-trained Olmo-3-7B-Instruct still describes the call in prose instead of emitting a structured one |
| InternLM2.5 | describes the search it would run instead of emitting a tool call |
| DeepSeek R1-Distill (Qwen, Llama) | reasoning distills that aren't tool-trained; they describe the search inside their reasoning instead of emitting a tool call |
| Phi-4-mini | the bundled llama.cpp aborts (SIGABRT) building the compute graph for this architecture |
| GLM-4-9B-chat | the bundled llama.cpp aborts (SIGABRT) loading this GGUF |
| ERNIE-4.5 0.3B, LFM2 1.2B | too small to call tools reliably |

## The QA harness

`tools/qa/opencode/` drives the real opencode binary against a real `lilbee serve`, one family at a time. Nothing is mocked: a pass means the whole path (serve, `/v1/chat/completions`, tool-call extraction, MCP `lilbee_search`, grounded answer) works for that model. Tool-calling is agent-agnostic, so a verified family behaves the same under hermes.

| File | Role |
|------|------|
| `matrix.py` | Driver. Defines the matrix (`models.toml`), runs each cell, writes `results/results.md`. |
| `models.toml` | One row per family with its GGUF ref and tier. |
| `prevalidate.py` | Optional pre-flight: assert each model answers well before an expensive run. |
| `stress.py` | Concurrency probe: many agents hitting one served model at once. |
| `results/` | Per-run output: `results.md` plus each cell's captured agent pane. |

### Per-cell lifecycle (`run_cell`)

1. **Setup**: pull the GGUF, write a per-cell workspace (a small indexed corpus plus an `AGENTS.md` directing the agent to use `lilbee_search`), scope the agent's built-in tools off, pin the model, boot `lilbee serve`, and index the corpus.
2. **Drive**: launch the agent in a dedicated tmux session, type the tier's scenario prompt, and poll the pane.
3. **Judge**: apply the PASS gate against the pane and the server log.
4. **Capture**: save the full agent pane to `results/<family>.pane.txt`.
5. **Teardown**: kill the session and the serve, scrape worker/dispatch errors from `launcher-serve.log`, delete the GGUF unless `--keep-models`.

### The PASS gate

A cell passes only when **both** hold, so a model that narrates a tool call as plain text cannot pass:

- the `⚙ lilbee_search` dispatch glyph appears in the agent pane, **and**
- a fresh delta of **>= 2** `POST /v1/chat/completions 200` lines in the server log (the tool turn plus its follow-up answer; 1 is prose-only).

A clean run also requires no worker/dispatch errors in `launcher-serve.log`.

### Memory and host notes

The harness is tuned for an 80GB+ GPU pod. On a memory-constrained host (e.g. a 32GB Apple Silicon laptop) a giant may not fit: the fleet refuses to load a model that exceeds free system RAM rather than freezing the machine (see `providers/fleet/planning.py`), so an oversize cell surfaces a clean error. `LILBEE_QA_NUM_CTX` pins a smaller context when needed.
