# Local models with opencode

`lilbee launch opencode` wires [opencode](https://opencode.ai) to your local
lilbee chat models so an agent can search your library, read the results, and
answer with citations, all on your machine. opencode talks to lilbee over the
OpenAI-compatible API and calls the `lilbee_search` tool through MCP, so a model
only works here if it can emit tool calls in a format lilbee can read back.

This page lists the model families we have driven end to end through opencode:
opencode sends a prompt, the model calls `lilbee_search`, opencode runs the tool
against an indexed workspace, and the model writes a final answer from the
results. Each family below completed that loop.

## Verified families

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

How reliably a model reaches for a tool depends on the model and its size, not
on lilbee: a smaller model may answer some prompts directly instead of
searching. When the model does call `lilbee_search`, lilbee reads the call and
opencode runs it.

## Not yet supported

A few families don't work through opencode today. The reason is either the
model itself or the bundled runtime, not the lilbee tool plumbing:

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

The harness lives in `tools/qa/opencode/` and drives the **real** opencode
binary against a **real** `lilbee serve`, one model family at a time. It is a
black-box integration test: nothing is mocked, so a pass means the whole path
(serve -> `/v1/chat/completions` -> tool-call extraction -> MCP `lilbee_search`
-> grounded answer) works for that model.

### Layout

| File | Role |
|------|------|
| `matrix.py` | The driver. Defines the model matrix (`models.toml`), runs each cell, writes `results/results.md`. |
| `models.toml` | The matrix: one row per family with its GGUF ref, tier, and `expected` support status. |
| `reelrun.sh` | Reel factory: records one validated demo reel per family, captures a multi-GPU snapshot, then gates the reel frame-by-frame. |
| `frame_qa.py` | OCRs a reel's frames and asserts the developer-visible arc (prompt -> dispatch -> grounded answer/code -> no error frame -> not a dead screen). |
| `gpu_snapshot.py` | Per-device VRAM snapshot while a model is resident; proves a giant's weights span multiple GPUs. |
| `stress.py` | Concurrency probe: many agents hitting one served model at once. |
| `results/` | Per-run output: `results.md` plus each cell's captured opencode pane. |

### Supported vs unsupported, and what a green run means

`models.toml` tags each family `expected = "supported"` (default) or
`"unsupported"` (documented below as not working). The report classifies every
cell as `PASS`, `REGRESSION` (a supported family that failed), `expected-fail`
(an unsupported family that failed for its known reason), or `newly-working`. The
run fails only on a regression, so a green matrix means **every supported family
passed and every unsupported one failed exactly as documented**.

### Reels and multi-GPU evidence

`reelrun.sh <family> <ref> <tier>` records the per-model demo reel and, in the
same run, produces two machine-checked artifacts: a `gpu.json`/`gpu.md` snapshot
(for a giant, proof the weights are tensor-split across >=2 cards) and a
`frame_qa.json`/`frame_qa.md` report (every unique frame OCR'd and graded). Both
gate the reel: a giant that lands on one card, or a reel whose timeline is a dead
boot screen or never shows the answer, is not published. The frame gate needs the
`tesseract` binary on the recording host (installed by `pod_bootstrap.sh`).

### Per-cell lifecycle (`run_cell`)

Each cell is one model, run in isolation:

1. **Setup**: pull the GGUF, write a per-cell workspace (a small indexed corpus
   plus an `AGENTS.md` that directs the agent to use `lilbee_search`), scope
   opencode's built-in tools off so the model must use the MCP search, pin the
   model as opencode's default, boot `lilbee serve`, and index the corpus.
2. **Drive**: `lilbee launch opencode` inside a dedicated tmux session, type the
   tier's scenario prompt, and poll the pane.
3. **Judge**: apply the PASS gate (below) by reading the pane and the server log.
4. **Capture**: save the full opencode pane to `results/<family>.pane.txt`.
5. **Teardown**: kill the tmux session and the serve; scrape worker/dispatch
   errors from `launcher-serve.log`; delete the GGUF unless `--keep-models`.

### The PASS gate

A cell passes only when **both** hold, so a model that narrates a tool call as
plain text (or rides a stale glyph) cannot pass:

- the `⚙ lilbee_search` dispatch glyph appears in the opencode pane, **and**
- a fresh delta of **>= 2** `POST /v1/chat/completions 200` lines in the server
  log (the tool turn plus its follow-up answer; 1 is prose-only).

A clean run also requires no worker/dispatch errors in `launcher-serve.log`.

### Memory and host notes

The harness is tuned for an 80GB+ GPU pod. On a memory-constrained host (e.g. a
32GB Apple Silicon laptop) a giant may not fit: the fleet now **refuses** to load
a model that exceeds free system RAM rather than freezing the machine (see
`providers/fleet/planning.py`), so an oversize cell surfaces a clean error.
`LILBEE_QA_NUM_CTX` pins a smaller context for the cell when needed.
