# Coding-agent stress scenarios and reels

The plan for the next session: run four coding agents concurrently against one
shared engine, on models people would actually code with, and record each run.
Every scenario is both a load test and a filmable demo.

This file is the contract for that session. Prompts here are the prompts that
get typed on camera.

## Models

No small models. A 4B or 8B model cannot carry a real coding task, and a demo
where the agent flails is worse than no demo.

| Tier | Candidate | Why it earns a row |
|---|---|---|
| Mid dense | Qwen3-32B, Q4_K_M (~20 GB) | The honest dense contrast. Makes the MoE number interpretable and fits comfortably beside an embed model. |
| Large dense | Llama-3.3-70B, Q4_K_M (~40 GB) | Where sharing stops being a convenience: two copies do not fit on one card, so the architecture is the only way four agents run at all. |
| Sparse MoE (already measured) | Qwen3.6-35B-A3B, Q8_0 (37 GB) | Reference point, carried over from the first run. |

Before pulling anything, run the real gguf-parser planner per candidate and
confirm slots, per-slot context, and KV budget on the target card. Substitute a
sibling model if a candidate does not fit four slots at a working context; do
not shrink context below what an agent needs to hold a session.

## The four agents

Four concurrent agents, chosen so they stress different parts of the engine at
once rather than four copies of one workload. Together they cover the two gaps
the first benchmark round left open: prefill-heavy traffic and concurrent
embed load.

| Agent | Work | Load profile it creates |
|---|---|---|
| 1 | Bug hunt in unfamiliar code | Heavy prefill: reads many files before generating much |
| 2 | Feature plus tests | Long sustained generation |
| 3 | Refactor across files | Bursty: many short turns, many tool calls |
| 4 | Architecture question answered from the indexed knowledge base | Retrieval: hits the embed model and the chat model concurrently |

Agent 4 is the important one. Nothing in the first benchmark round put embed
and chat under load at the same time, and that contention is the most plausible
place for the shared engine to misbehave.

All four run against the lilbee repo itself, indexed into a vault, so the work
is real and the retrieval has something true to find.

## The prompts

Written the way a person actually asks. No "create a file named X", no "then
stop", no restating the acceptance criteria back at the model. Ambiguity is
deliberate: a real question leaves the agent to figure out where to look.

**Agent 1, bug hunt**

```
something's off with catalog search when i filter by quant. models that
definitely have that quant are getting dropped from the results. can you dig
into why?
```

**Agent 2, feature plus tests**

```
i want lilbee model rm to warn me first if the model i'm deleting is one i've
got configured for a role. right now it just deletes it and then the next run
fails with a missing model. add the guard and a test for it?
```

**Agent 3, refactor**

```
the vram fits check is copy pasted in a few places in the placement code and
they've drifted apart. can you pull it into one helper and use it everywhere?
```

**Agent 4, architecture question**

```
how does lilbee decide it can share an engine with another process instead of
starting a second one? walk me through what it checks.
```

### Grading

Open prompts cannot be graded by an exit code, and forcing a checkable
criterion into the prompt is what made the old ones read like machine input.
Keep the prompts human and grade afterward against a rubric per scenario:

- Agent 1: names the actual filtering code path and a concrete cause, not a
  generic "add validation".
- Agent 2: the guard covers every configured role, and the test fails without
  the guard.
- Agent 3: one helper, all call sites moved, tests still green.
- Agent 4: names the engine pin and per-role model contract, and the
  bind-before-build order.

Grade with Fable 5 rather than the local model, so the judge is independent of
the system under test.

## Storyboard, per model tier

One reel per model tier, same four agents each time. The shape:

**Beat 0, the setup card (0:00 to 0:04).** One terminal, four panes tiled.
Each pane opens on its task in an editor view so the viewer can read all four
tasks at once before anything moves. Header line names the model and the card.

**Beat 1, launch (0:04 to 0:10).** Enter in each pane drops it into the coding
agent. Four agents come up against one engine. A fifth strip along the bottom
shows `nvidia-smi` memory and the engine process count, so the viewer sees one
engine serving four clients rather than four engines.

**Beat 2, the prompts get typed (0:10 to 0:20).** Each prompt types out
visibly, in the voice above. This is the beat that makes the reel credible:
the viewer reads a question they would have asked.

**Beat 3, concurrent work (0:20 to 0:50).** All four agents work at once. The
memory strip stays flat while four sessions stream. This is the proof shot.

**Beat 4, results (0:50 to 1:00).** Each pane lands its result: the bug named,
the test passing, the refactor green, the architecture answer written. Freeze
on the final frame.

### Length

There is a real tension here. A meaningful coding task takes minutes, and the
house reel style is roughly 27 seconds at 1x with no speedup. Two artifacts per
scenario resolves it:

- **Evidence capture**: full length, unedited, kept with the benchmark data.
  This is what backs the claims in the report.
- **Reel cut**: the beats above, with the working stretch cut to its most
  legible window rather than sped up, so the 1x text-quality rules still hold.

Which one leads on the site is a call to make after seeing the first evidence
capture.

## Session scope

That session runs stress tests and records them, nothing else. Per scenario:
run it, check the invariants, grade the output, record it. If a run surfaces a
failure, fix it, then re-record that scenario so every shipped reel reflects
the fixed build. No parallel feature work.
