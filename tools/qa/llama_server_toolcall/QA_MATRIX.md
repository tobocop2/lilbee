# Per-model tool-call QA + demo reel

End-to-end QA of every model lilbee supports for agentic tool calling, recorded as a
demo at the same time. Where [`FINDINGS.md`](FINDINGS.md) answered "can `llama-server
--jinja` parse each family's native tool calls?" in isolation, this matrix answers the
product question: **does a real agent (opencode) talking to a lilbee-hosted model
actually call lilbee's tools and get useful work done?**

## What a run exercises

For each model, [`record_reel.py`](record_reel.py) stands up the full path and drives it
through the real opencode TUI:

```
opencode TUI
  ├─(model, OpenAI-compatible /v1)──> llama-server --jinja   the model, native tool parsing
  └─(MCP /mcp, bearer token)────────> lilbee serve            lilbee_search over src/lilbee/
```

opencode runs in an **empty** project directory whose `AGENTS.md` says the lilbee source
is not on disk and the only way to read it is the `lilbee_search` tool. The recorded
prompt is a real dev task ("add a new model family whose tool calls look like
`<tool_call name=search>{json}</tool_call>` ... wire it into the parser"), so the model
has to search lilbee's own code before it can answer. This is the same dynamic as the
godot demos: the agent must use the tool, but the prompt never says "use the tool."

### PASS gate

A run **PASSes** when lilbee serve logs a new `CallToolRequest` during the recording,
i.e. the model actually called `lilbee_search` over MCP. The verdict is mechanical (a
log delta), not a judgement of the answer. A model that records a demo but never calls
the tool is a **FAIL(no tool call)**; a model whose server never comes up is a
**SETUP_FAIL**.

## GPU placement

Small models run on a **single GPU**. Only the 200GB giants
(`multi_gpu_only` in [`models.py`](models.py)) span both, gated behind `MULTIGPU=1`.
This is load-bearing, not cosmetic: force-splitting a small model across both GPUs
(`-ngl 999`, two visible devices) crashed llama.cpp outright for gemma-4-E2B
(`GGML_SCHED_MAX_SPLIT_INPUTS`, bb-rpju) and silently suppressed tool-call emission for
mistral-nemo (bb-1cmj). The fleet must do capacity-based placement for the same reason
(bb-v6pk, blocks bb-n1bj).

## Results (reel run 2026-05-26, 2x H200)

| Family | Tier | GPU | Tool-call QA | Notes |
|--------|------|-----|--------------|-------|
| qwen3 | small | 1 | PASS | |
| llama3 | small | 1 | PASS | |
| mistral-nemo | small | 1 | PASS | FAILed on the 2-GPU split path; single-GPU fixed it (bb-1cmj) |
| gemma4 | small | 1 | PASS | crashed on the 2-GPU split path; single-GPU fixed it (bb-rpju) |
| qwen3-coder | small | 1 | _pending_ | |
| hermes | small | 1 | _pending_ | needs its real `tool_use` template (bartowski GGUF strips it) |
| functionary | small | 1 | _pending_ | |
| minimax-m2 | giant | 2 | _pending_ | |
| glm-4.6 | giant | 2 | _pending_ | ~200GB across both H200s |
| gpt-oss | large | 1 | _pending_ | Harmony format |
| glm-air | large | 1 | _pending_ | |

Native parser gaps (`granite`, `cohere`) and model-side declines (`phi4-mini`,
`internlm2`, `glm-4-9b`, `smollm`, `lfm2`) are catalogued in
[`FINDINGS.md`](FINDINGS.md); they are not in the demo reel because they either need
lilbee's fallback parser or the model itself refuses to call tools.

## Issues filed from this QA

- **bb-rpju** (closed) — llama-server crash splitting a small model across 2 GPUs; fixed
  in the harness (single-GPU default).
- **bb-1cmj** (closed) — mistral-nemo emitted no tool call on the split path; resolved by
  the same single-GPU fix.
- **bb-v6pk** (open, blocks bb-n1bj) — the multi-GPU fleet must not blindly layer-split a
  model that fits on one GPU; this is the general, shipped-code version of bb-rpju.

## Running it

On a 2-GPU pod with the standalone `llama-server` built (see
[`FINDINGS.md`](FINDINGS.md) for the build), lilbee serve installed with the CUDA
llama-cpp wheel (CPU embedding thrashes), opencode + vhs on PATH, and the rod chromium
wrapped with `--no-sandbox` for vhs-as-root:

```bash
HF_TOKEN=... HF_HUB_DISABLE_XET=1 PATH=$HOME/.opencode/bin:$PATH \
  python record_reel.py            # full reel
HF_TOKEN=... ... python record_reel.py gemma4 hermes minimax-m2   # resume subset
```

A resume run takes families on argv and **merges** into the existing
`/root/demos-out/reel_manifest.txt` instead of overwriting it, so already-recorded models
survive. Each model writes `reel-<fam>.gif` / `.webm` / `.png` and, on PASS, a published
`vhs.charm.sh` URL into the manifest (`family<TAB>verdict<TAB>url`).

The rendered gifs and their tapes are published to the site's `gh-pages` branch under
`demos/opencode-<model>.{gif,tape}` via a PR.
