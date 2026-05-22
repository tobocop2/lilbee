# Opencode integration QA matrix

Local-only smoke matrix for the `lilbee launch opencode` happy path. Separate from the pytest-based pre-publish QA suite in `tools/qa/` (which gates releases in CI); this lives under `opencode/` to keep the two surfaces from sharing namespace.

## Run

```bash
# Full smoke matrix across every family in models.toml
uv run python tools/qa/opencode/matrix.py

# One family
uv run python tools/qa/opencode/matrix.py --families qwen3

# Skip the model-pull step (use cached GGUF only)
uv run python tools/qa/opencode/matrix.py --families qwen3 --no-pull
```

## What runs, per cell

1. `lilbee model pull` the target GGUF (unless `--no-pull`).
2. Seed a per-cell workspace with three markdown fixtures and index them via `lilbee add`.
3. Rewrite `~/.local/state/opencode/model.json` so opencode boots with the target model selected.
4. Boot `lilbee serve` on a free port.
5. `tmux new-session -d ... lilbee launch opencode` in a 200x50 pseudo-terminal.
6. Drive smoke scenarios via `tmux send-keys`; poll `tmux capture-pane` for expected substrings.
7. Tear down (the tmux session stays up on failure for manual inspection).

## Scenarios

| ID | Prompt | Pass criteria | Catches |
|----|--------|---------------|---------|
| S1 | "search the indexed docs for the chat worker file" | pane contains `lilbee_search` + `chat worker` | tool extraction wired end-to-end |
| S2 | "find the dispatch layer docs and then summarize how it routes models" | pane contains `lilbee_search` + `dispatch` + `KnownModelCache` | multi-tool turn + tool-result round-trip |
| S3 | "give me a verbose three-paragraph overview of how tool extraction works" | pane contains `tool`; no raw marker leak | streaming visible; Phi-4 / Functionary v3 marker class |

Forbidden in every scenario: `<tool_call>`, `[TOOL_CALLS]`, `functools[`, `Error:`, `Traceback`. A hit fails the cell.

## Output

- `tools/qa/opencode/results/results.md` — status table + pane excerpts for failing cells
- `tools/qa/opencode/logs/<family>.log` — `lilbee serve` and `lilbee add` stderr per cell

## Deferred

S4 (long-history windowing), S5 (mid-stream cancellation), S6 (backpressure 429 surfacing) are designed but not yet implemented. They will run only on the named happy-path family (qwen3) once added (bead bb-m8fi tracks follow-up).

## Findings to date

After the integration fixes on `fix/opencode-integration-findings`:

- **qwen3** (Qwen/Qwen3-4B-GGUF) — 3/3 PASS. The schema-slim fix freed enough context budget that S3 no longer hits overflow.
- **gemma4** (unsloth/gemma-4-E2B-it-GGUF) — S1 PASS. S2/S3 model-behavior: gemma4 paraphrases the rare-class quote and tries the built-in `webfetch` tool on open-ended prompts.
- **hermes** (bartowski/Hermes-3-Llama-3.1-8B-GGUF) — S1 PASS, newly enabled by `chat_format_override` + stream-downgrade. S2/S3 model-behavior: Hermes-3 8B drifts to opencode's built-in `read` tool on multi-turn prompts.
- **llama3** (bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) — S1 PASS, newly enabled by the bare-JSON parser fix in `schemas/llama3.json`. S2/S3 hit multi-turn context overflow.
- **Skipped:** mistral v0.3 (base model not trained for tool calling; prefer Mistral-Nemo / Mistral-Small), phi4mini (llama-cpp-python wheel CPU-feature SIGABRT on M1 Pro; environment fix), gemma2 (no tool template), qwen3-coder (17 GB MoE, opt-in only).

S1 validates the integration end-to-end across four chat templates: Qwen ChatML+tool_call, Gemma-4 native, ChatML-function-calling (via override), and Llama-3.1 bare-JSON. S2/S3 timeouts on non-qwen3 cells reflect model capability rather than integration breakage.
