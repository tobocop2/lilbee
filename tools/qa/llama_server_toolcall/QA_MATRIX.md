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

Each answer was read from the live `opencode run` transcript and the prompt iterated
until the answer was accurate and grounded, *before* recording. Verdict = the model
called `lilbee_search` and produced a correct, cited answer.

| Model (real HF name) | Tier | GPU | ctx | Demo | Topic |
|----------------------|------|-----|-----|------|-------|
| Qwen3-4B | small | 1 | 32K | PASS | chunking (`data/chunk.py`) |
| gemma-4-E2B-it | small | 1 | 32K | PASS | embedding (`retrieval/embedder.py`) |
| Meta-Llama-3.1-8B-Instruct | small | 1 | 32K | PASS | search step |
| Mistral-Nemo-Instruct-2407 | small | 1 | 32K | PASS | reranking (cross-encoder + BM25 pin) |
| Hermes-3-Llama-3.1-8B | small | 1 | 32K | PASS | query expansion |
| functionary-small-v3.2 | small | 1 | 32K | PASS | dedup (`prepare_results`/`diversify_sources`) |
| Qwen3-Coder-30B-A3B-Instruct | mid | 1 | 32K | PASS | code vs prose chunking (tree-sitter) |
| MiniMax-M2 | giant | 2 | 128K | PASS | full retrieval path + extension points |
| GLM-4.6 | giant | 2 | 128K | PASS | query expansion + reranking + caching |
| Qwen3-235B-A22B-Instruct-2507 | giant | 2 | 128K | PASS | end-to-end pipeline (used `lilbee_settings_get`) |
| GLM-4.5-Air | large | 1 | 64K | PASS | query→ranked-results pipeline |
| gpt-oss-120b | large | 1 | 64K | EXCLUDED | grounding-unreliable (bb-lks6) |

All 11 PASS models are in the demo reel (PR to gh-pages) with gif + mp4 + tape and the
real model name shown in opencode. gpt-oss-120b is excluded: it calls `lilbee_search`
but cannot ground its answer (hallucinates the embedding model/store, or falsely claims
the code is absent) across four prompt variants while every other model is accurate on
the identical index — a model limitation, not a lilbee/harness bug (bb-lks6).

Prompt-iteration lessons worth keeping: small models need one focused question (a broad
multi-stage prompt makes them flail); the coder/giants over-search and overflow context,
so giants get a large `-c` and prompts are scoped to "search then explain briefly"; a
"concept graph" query surfaced parser chunks for GLM-4.5-Air, so its prompt was moved to
the central retrieval pipeline.

Native parser gaps (`granite`, `cohere`) and model-side declines (`phi4-mini`,
`internlm2`, `glm-4-9b`, `smollm`, `lfm2`) are catalogued in
[`FINDINGS.md`](FINDINGS.md); they are not in the reel because they either need lilbee's
fallback parser or the model itself refuses to call tools.

## opencode integration gotchas (hard-won)

- **opencode resolves its project from `$PWD`, not the process cwd.** `subprocess(cwd=…)`
  leaves `$PWD` at the launcher's dir, so opencode loads the wrong project, never sees the
  lilbee provider, and 404s `lilbee/<model>` ("Model not found"). Set `PWD` explicitly, or
  launch via `cd <proj> && opencode`.
- **Force the model with `-m lilbee/<name>`**, else `opencode run`/TUI falls back to its
  built-in `build` agent + default model and reads files directly instead of using lilbee.
- **Warm llama-server completions before opencode** — it answers `/health` before it can
  serve completions; opencode's resolution probe 404s until then.
- **Disable the crawl/wiki MCP tools** in the demo (`tools: {lilbee_crawl: false, …}`) or a
  model wanders off to crawl GitHub instead of searching the index.

## Issues filed from this QA

- **bb-rpju** (closed) — llama-server crash splitting a small model across 2 GPUs; fixed
  in the harness (single-GPU default).
- **bb-1cmj** (closed) — mistral-nemo emitted no tool call on the split path; resolved by
  the same single-GPU fix.
- **bb-v6pk** (open, blocks bb-n1bj) — the multi-GPU fleet must not blindly layer-split a
  model that fits on one GPU; this is the general, shipped-code version of bb-rpju.
- **bb-lks6** (open) — gpt-oss-120b is grounding-unreliable in the agentic RAG demo
  (calls `lilbee_search` but hallucinates or falsely refuses); excluded from the reel.

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
