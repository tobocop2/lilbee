# Opencode integration QA matrix

Local-only matrix that battle-tests `lilbee launch opencode` per supported model. Separate from the pytest-based pre-publish QA suite in `tools/qa/` (which gates releases in CI); this lives under `opencode/` so the two surfaces don't share namespace.

Each cell is **one model, one realistic prompt, one captured opencode pane** — the cell pane is both the pass/fail signal and the per-model demo evidence. Prompts are tiered to the model's capability:

- Smalls (≤4B) get a focused single-class lookup.
- Mids (5–10B) get a multi-class explanation with citations.
- Giants (17B+) get a codegen task that requires verifying the API surface against an indexed reference.

The reference corpus is the **Godot 4 class XML** (`/root/godot/doc/classes`) so the prompts are realistic dev questions, not internal-grep smoke.

## One-time pod setup

On a fresh from-source pod (no `pip install lilbee`), run the bootstrap first. It
installs the build toolchain + tmux + uv + opencode + VHS (the reel recorder),
syncs a local-disk venv, builds the engine once and caches it on `/workspace`, and
is idempotent across a stop/resume (only /workspace survives a resume, so the
engine isn't recompiled). VHS records on the pod (ttyd 1.7.7 + `VHS_NO_SANDBOX`
+ relative `Output` paths; the bootstrap smoke-tests it):

```bash
bash tools/qa/opencode/pod_bootstrap.sh        # BACKEND=cu124 by default
source /workspace/qa_env.sh                     # complete env the bootstrap wrote
export HF_TOKEN=...                              # for model pulls
```

Then the corpus:

```bash
# Clone Godot for the reference corpus (depth 1 is fine, classes/ is small)
git clone --depth 1 https://github.com/godotengine/godot /root/godot

# Pre-index the class reference once, reused by every cell
LILBEE_DATA=/root/godot_corpus lilbee add /root/godot/doc/classes
```

`/root/godot_corpus/data/lancedb` is the shared lancedb. Each matrix cell starts a fresh workspace and copies that lancedb in (a fast `cp -r` — no per-cell re-embed), so cells stay isolated but share the indexed corpus.

## Run

```bash
# Full matrix across every family in models.toml, largest-first
uv run python tools/qa/opencode/matrix.py

# One family
uv run python tools/qa/opencode/matrix.py --families qwen3

# Skip the model-pull step (use cached GGUF only)
uv run python tools/qa/opencode/matrix.py --families qwen3 --no-pull
```

On a pod (`RUNPOD_POD_ID` set), the matrix arms `pod_watchdog.sh` automatically:
if nothing is written under the logs/models/HF paths and the GPUs sit idle for
30 minutes, the watchdog stops the pod instead of letting a hung run bill for
nothing. It can also be run standalone against any log path:

```bash
IDLE_MIN=30 tools/qa/opencode/pod_watchdog.sh /workspace/qa_matrix.log
```

## Pod provisioning (infrastructure as code)

Provisioning is SkyPilot, reading the RunPod key from `~/.runpod/config.toml`. A
reusable network volume holds the from-source engine, the pulled GGUFs, and the
Godot corpus, so a torn-down pod loses nothing and a re-launch resumes. Pods are
on-demand secure-cloud (never interruptible), so spot eviction can't kill a run.

One-time:

```bash
pip install "skypilot[runpod]"
runpod config            # store the RunPod API key (key never enters the repo)
sky check runpod
make qa-pod-volume       # create/adopt the lilbee-qa network volume (once)
```

Run the matrix on a fresh pod:

```bash
make qa-pod-up                              # provision + bootstrap + run the full matrix
make qa-pod-up MATRIX_ARGS="--families qwen3 --keep-models"   # narrow it
sky logs lilbee-qa                          # follow the run (or: make qa-pod-logs)
ssh lilbee-qa                               # drive reels by hand (see below)
make qa-pod-down                            # tear down; the volume + its state survive
```

`qa-pod.sky.yaml` mounts the volume at `/workspace`, runs `pod_bootstrap.sh` in
`setup`, then in `run` builds the corpus (idempotent, `qa_corpus.sh`) and runs
`matrix.py`. The GPU ladder leads with `A100-80GB:3` and degrades the count
before dropping a tier; the volume pins the datacenter. `pod_watchdog.sh` still
arms automatically as the idle-billing backstop.

## Demo reels

`reelrun.sh <family> <ref> <small|mid|coder|giant>` records one reel per
supported model against a pre-warmed serve: it rebuilds the cell workspace,
records a `lilbee launch opencode` session with the tier prompt via VHS, and
extracts the full unique-frame set (1 fps, consecutive duplicates collapsed)
into `/workspace/reelfactory/<family>/` for frame-by-frame review. A reel is
accepted only after every unique frame has been reviewed.

## What runs, per cell

1. `lilbee model pull` the exact GGUF ref from models.toml (unless `--no-pull`), retried on transient failures; the cell hard-fails if the ref is not registered afterwards. The shards are then read once into page cache (the printed MB/s doubles as a volume health probe).
2. Make a fresh per-cell workspace; `cp -r` the shared Godot corpus lancedb into it (no re-embed). The workspace gets a project-level `opencode.json` (built-in tools off, autoupdate off) and the event-tap plugin under `.opencode/plugins/`.
3. `lilbee launch opencode` inside a pod-side tmux session (200×50 pseudo-terminal). The launcher spawns its own `lilbee serve` internally, wires opencode's provider + MCP to it, and pins the startup model via the injected config. The harness waits for the TUI (alternate-screen flag or first tap event) before the scenario clock starts.
4. Send the cell's tier prompt via `tmux send-keys`. The PASS gate is a fresh `lilbee_search` tool-dispatch event from the tap (`.lilbee/qa-events.jsonl`); the pane is still scanned for forbidden raw-leak substrings, and the gear-glyph + completions-delta gate remains as the no-tap fallback.
5. Capture the final pane into `results/<family>.pane.txt` and the cell's status into `results/results.md`.
6. Tear down. The tmux session **stays up on failure** for manual inspection; on success it's reaped.

## Tier prompts

Same prompt for every model in a tier — apples-to-apples comparison across the same yardstick.

### Tier 1 — Smalls (≤4B)

> Look up `Node._process(delta)` in the Godot 4 class reference and tell me what it does and when the engine calls it.

Slightly explicit ("look up") because the smallest models won't autonomously tool-call without a hint. The MCP tool name `lilbee_search` is never mentioned.

### Tier 2 — Mids (5–10B)

> In Godot 4, how do you connect a signal between two nodes? Walk me through `Object.connect`, `Signal.emit`, and the `CONNECT_*` flags with actual signatures.

Fully natural. A capable mid-sized model should discover the MCP tool and tool-call on its own — that's the integration test.

### Tier 3 — Giants (17B+)

> Write a Godot 4 GDScript scene script that procedurally generates a 2D dungeon using TileMap and TileSet. It should expose a `regenerate()` signal so the game can rebuild on demand.

Fully natural — what a developer would actually type. The agent should know to verify the classes/methods exist (lilbee_search is exposed via MCP). A model that hallucinates API names instead of looking them up is a real integration failure for that model.

## Pass criteria

A cell passes when **all** are true:

- `⚙ lilbee_search` appears in the pane (the tool was invoked).
- The answer cites at least one Godot class XML file (e.g. `Node.xml`, `TileMap.xml`).
- No forbidden substring appears (see table below).
- For **Giants**: the generated GDScript only references class / method / property / signal names that exist in the indexed Godot 4 reference (no hallucinated API). This is a manual review of the captured pane.

A genuine tool turn drives **at least 2 chat completions** (the model's tool-call turn + the follow-up answer once opencode feeds the tool result back in); a prose-only reply drives one and fails the tool-discovery test.

### Forbidden substrings

| Substring | What it catches |
|-----------|------------------|
| `<tool_call>` | model emitted the call as raw template text instead of through opencode's tool channel |
| `[TOOL_CALLS]` | Mistral-style raw marker leaked into the chat pane |
| `functools[` | Functionary marker leaked |
| `Error:` | opencode / serve surfaced an error in the pane |
| `Traceback` | a Python traceback rendered in the pane |

Any hit fails the cell.

## Models and tier assignment

Largest first. Cleanup-per-cell keeps disk flat at the peak single-model size.

| Family | GGUF size | Tier | Prompt |
|--------|-----------|------|--------|
| minimax-m2 | ~230 GB | giant | Tier 3 codegen |
| gpt-oss | 63 GB | giant | Tier 3 codegen |
| glm-air | 60 GB | giant | Tier 3 codegen |
| qwen3-coder | 17 GB | giant | Tier 3 codegen |
| functionary | 8.5 GB | mid | Tier 2 multi-class |
| mistral-nemo | 7 GB | mid | Tier 2 multi-class |
| glm-4-9b | 5.5 GB | mid | Tier 2 multi-class |
| deepseek-r1-llama | 4.9 GB | mid | Tier 2 multi-class |
| hermes | 4.9 GB | mid | Tier 2 multi-class |
| llama3 | 4.9 GB | mid | Tier 2 multi-class |
| granite | 4.8 GB | mid | Tier 2 multi-class |
| olmo3 | 4.8 GB | mid | Tier 2 multi-class |
| deepseek-r1-qwen | 4.7 GB | mid | Tier 2 multi-class |
| internlm2 | 4.7 GB | mid | Tier 2 multi-class |
| cohere | 4.4 GB | mid | Tier 2 multi-class |
| ~~mistral v0.3~~ | 4.4 GB | — | **skipped** — v0.3 base not trained for tool calling; prefer Mistral-Nemo |
| gemma4 | 3.0 GB | small | Tier 1 lookup |
| qwen3 | 2.5 GB | small | Tier 1 lookup |
| phi4mini | 2.4 GB | small | Tier 1 lookup |
| smollm | 2.0 GB | small | Tier 1 lookup |
| ~~gemma2~~ | 1.6 GB | — | **skipped** — no tool template |
| lfm2 | 0.9 GB | small | Tier 1 lookup |
| ernie | 0.4 GB | small | Tier 1 lookup |

Adding a model: add its `[[model]]` entry to `models.toml` with `tier = "small" | "mid" | "giant"`; the matrix picks the prompt from `_TIER_PROMPTS[tier]`.

## Output

- `tools/qa/opencode/results/results.md` — status table (family / tier / PASS|FAIL / scenario time / tool-completions count) with pane excerpts for failing cells.
- `tools/qa/opencode/results/<family>.pane.txt` — full final opencode pane per cell. **This is the per-model demo evidence**: the actual TUI interaction (prompt, tool call, answer, citations).
- `tools/qa/opencode/logs/<family>.log` — `lilbee serve` stderr per cell (worker / dispatch errors scraped from here).

## Deferred (designed but not yet implemented)

- **S4** long-history windowing (`bb-xdic`).
- **S5** mid-stream cancellation.
- **S6** backpressure / 429 surfacing.

These will run only on the qwen3 happy-path cell once added (bead `bb-m8fi` tracks the follow-up).

## Prior findings (historical, pre-tiered prompts)

Before the matrix was tiered, with the old internal-grep smoke prompts (preserved for context — the new tiered prompts will produce a different table):

- **qwen3** (Qwen3-4B) — 3/3 PASS on the old smoke. Reference happy path.
- **gemma4** (gemma-4-E2B-it) — S1 PASS. S2/S3 model-behavior: paraphrased the rare-class quote, tried opencode's built-in `webfetch`.
- **hermes** (Hermes-3-Llama-3.1-8B) — S1 PASS (newly enabled by `chat_format_override` + stream-downgrade). S2/S3 drifted to opencode's built-in `read` on multi-turn.
- **llama3** (Meta-Llama-3.1-8B-Instruct) — S1 PASS (newly enabled by the bare-JSON parser in `schemas/llama3.json`). S2/S3 hit multi-turn context overflow.

The new tiered prompts are realistic dev questions rather than internal grep, so behaviour will differ; the qwen3 happy path is still the reference cell to trust first.
