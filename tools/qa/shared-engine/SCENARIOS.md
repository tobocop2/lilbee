# Coding-agent stress scenarios and reels

The plan for the next session: run four coding agents concurrently against one
shared engine, on models people actually reach for, and record each run. Every
scenario is both a load test and a filmable demo.

This file is the contract for that session. The prompts here are the prompts
that get typed on camera.

## Models and hardware

Consumer hardware wherever it is honestly possible. A reel shot entirely on
rented datacenter cards proves the architecture but sells nothing, because the
viewer cannot picture themselves doing it. Only the giant tier earns a
warehouse.

No small models either. A 4B or 8B cannot carry a real coding task, and an
agent flailing on camera is worse than no demo.

| Tier | Model | Size | Hardware | Role in the story |
|---|---|---|---|---|
| Consumer, single card | Qwen3-Coder-30B-A3B, Q4_K_M | ~18 GB | One RTX 4090 or 5090; M-series 32 GB | "Runs on the card you already own." Already a verified passing family in this repo's opencode matrix, including on M1 Metal. |
| Consumer, enthusiast desktop | Qwen3-Coder-Next, 80B-A3B, Q4_K_M | ~48.7 GB | Dual 5090 (64 GB), or Mac Studio M4 Max 64 GB | An 80B agentic coder at ~71% SWE-bench Verified, on a desktop. The Mac path is the least friction: no dual-card power budget, roughly 80 W at load. |
| Workstation | MiniMax-M2.1, 230B-A10B | ~130 GB | Mac Studio M3 Ultra (large unified memory) | Frontier coding agent on a machine a person can buy. On NVIDIA this tier needs expert offload, which lilbee does not support yet. |
| Warehouse | Kimi K2.6, 1T-A32B | ~340 GB at UD-Q2_K_XL | 4x H200 | The SWE-bench leader at ~80.2%. The one row that is honestly datacenter-only. |
| Reference | Qwen3.6-35B-A3B, Q8_0 | 37 GB | A100 80GB, already run | Free row, no new pod time. |

### Why the 80B needs more than one consumer card today

Qwen3-Coder-Next activates only ~3B of its 80B parameters per token, so in
principle a 24 GB card could hold the hot weights and park idle experts in
system RAM. llama.cpp supports exactly that. **lilbee does not**: there is no
`--n-cpu-moe`, no `--override-tensor`, nothing in the fleet launcher that
splits experts off the GPU, so every model must fit VRAM entirely. That is the
single lever between "rent an A100" and "runs on your 4090".

Expert offload is tracked in the issue tracker and **this plan is blocked on
it**, along with the wider audit of what else llama.cpp offers that lilbee
never emits. When those land, the enthusiast tier collapses onto a single 4090
and the workstation tier opens up on NVIDIA, so re-read this table before
booking any hardware: the whole shopping list changes.

### What four agents means on a small card

The planner already fits chat slots to the memory budget (`_fit_slots` steps
the count down until the footprint fits), so a 24 GB card will serve four
agents on fewer than four slots. That is the correct behaviour and it is worth
showing rather than hiding: the shared engine queues fairly instead of running
out of memory. The narration must say "four agents sharing one engine", not
"four in parallel", when the slot count is below four. Record the resolved slot
count on screen so the claim matches the config.

### Gates before pulling anything

- **Tool-call plumbing is not uniform.** Qwen3-Coder is already verified here,
  so both Qwen3-Coder models should inherit working dispatch. MiniMax and Kimi
  are new families and may need a response-parser schema before they dispatch
  at all. Verify with the existing opencode matrix harness first; fixing a
  parser is dev work and does not belong in a recording session.
- **Size against the real planner.** Run the gguf-parser planner per candidate
  on the actual target hardware and confirm slots, per-slot context, and KV
  budget. A 4-bit 80B gains roughly 7 GB going from 4k to 256k of context. Do
  not shrink context below what an agent needs to hold a session; change
  hardware or quant instead, and say which in the report.
- **Do not quietly drop quant to make something fit.** A 4090 can technically
  load the 80B at 2-bit, but that is a different model from the one the
  benchmarks describe. If a tier only fits at a degraded quant, either move it
  up a tier or state the quant plainly on the reel.

## The four agents

Four concurrent agents, chosen so they stress different parts of the engine at
once rather than running four copies of one workload. Together they close the
two gaps the first benchmark round left open: prefill-heavy traffic and
concurrent embed load.

| Agent | Shape of work | Load profile | What the pane looks like |
|---|---|---|---|
| 1 | Bug hunt in unfamiliar code | Heavy prefill: reads many files before generating much | Files scroll past, then the agent stops and names a cause. The payoff frame is the diagnosis sentence. |
| 2 | Feature plus a test | Long sustained generation | Code is written, then pytest runs. The payoff frame is green output. |
| 3 | Refactor across files | Bursty: many short turns, many tool calls | Rapid small edits in several files, then the suite runs. The payoff frame is the diff plus green. |
| 4 | Architecture question from the indexed vault | Retrieval: embed and chat under load together | Visible `lilbee_search` calls returning real hits, then a written answer. The payoff frame is the grounded explanation. |

Agent 4 matters most for the benchmark. Nothing in the first round put embed
and chat under load at the same time, and that contention is the most plausible
place for a shared engine to misbehave.

## Task tiering

The same four shapes at every tier, with difficulty scaled to the model. Asking
a 30B to solve a cross-thread lock deadlock on camera produces a flailing demo;
asking Kimi to extract a duplicated helper wastes the model.

**Every task is real.** Nothing here is invented. Each one is either a bug that
actually happened in this repo, where the real fix is known so grading is
exact, or an open issue somebody actually wants, so the run produces a usable
patch instead of throwaway motion. An agent hunting a bug that does not exist
finds nothing and films badly, which is the failure mode this rule prevents.

**Pin the checkout.** For every task drawn from a fixed bug, start the run at
the commit immediately before the fix landed, or the agent reads the answer out
of git history. Pins are listed per task.

Open-issue tasks should be re-checked against `bd ready` before the session; if
one has been closed in the meantime, swap in another from the same tier rather
than resurrecting dead work.

### Tier 1, Qwen3-Coder-30B on one consumer card: well-scoped, single subsystem

Bug hunt, real, pin `7a91a166` (fixed by `c62a0f28`, issue #537):

```
so the chat doesn't scroll down to the end when an answer comes in, you have to
drag the scrollbar yourself every time. can you work out why?
```

Feature, real, from the open HTTP chat surface issue:

```
i want the http chat surface to keep window history and compact a run when it
gets long, and stream something so the client knows compaction happened. can
you build that?
```

Refactor spanning several files, real, entry-point parity:

```
i think validation for the engine dir env vars only happens on some of the
entry points and not the others. can you find every place we take it and make
them agree?
```

Architecture question, answered from the vault:

```
how does lilbee decide it can share an engine with another process instead of
starting a second one? walk me through what it checks.
```

### Tier 2, Qwen3-Coder-Next on an enthusiast desktop: cross-file, needs judgment

Bug hunt, real, pin `06239a80` (fixed by `4a43e0c9`):

```
so after the engine restarts, a long running lilbee serve keeps erroring on
every chat, but the same query through the cli works fine every time. same box,
same engine. why would the resident process not recover when a fresh one does?
```

Feature, real, the expert-offload issue, and the one that matters most here:

```
i want lilbee to run big moe models on consumer gpus. only about 3b params are
active on an 80b so in theory we keep the hot weights on the card and push the
idle experts out to system ram, and llama.cpp already supports that. we don't
do any of it. can you work out what it would take and build it?
```

Refactor, real, pin `fc6347cc` (fixed by `af85823a`):

```
if someone asks for more output tokens than the window can give, we kill the
turn instead of just giving them what fits. that seems wrong to me. can you
make it degrade instead?
```

Architecture question:

```
if two people on the same box are running lilbee with different models
configured, what actually happens? i want to understand whether they collide.
```

### Tier 3, MiniMax-M2.1 on a workstation: real open P1 work

Bug hunt, real, the open event-loop blocking issue, currently unfixed:

```
i think chat retrieval is blocking the server event loop. if two requests come
in at once the second one just sits there until the first one finishes its
whole turn. can you confirm that's actually what's happening and fix it?
```

Feature, real, the recovery-window follow-up:

```
when the engine gets killed and rebuilt, requests that land during the rebuild
fail instead of waiting it out. i'd rather that window turned into latency than
errors, but i don't want something hanging forever if the rebuild is genuinely
broken. what's the right design here, and can you build it?
```

Refactor, real, the open retrieval-noise issue:

```
retrieval is pulling in tocs and cover pages and diluting the real context. i
think the title and table arms are injecting document structure noise. can you
dig into it?
```

Architecture question:

```
walk me through every way the engine acquisition ladder can fail when two
processes race each other, and tell me which of those we actually handle.
```

### Tier 4, Kimi K2.6 in the warehouse: the hardest real problem here

Bug hunt, real, pin `f36e1a8e` (fixed by `9a8f687d`). This one took a human
several hours and required reading filelock's source:

```
our integration tests hang, but only in ci and only sometimes. it's a file lock
that gets acquired on one thread and released on another, and after that a
later acquire never comes back. i've been staring at this a while. what's
actually going on?
```

Feature, real, the expert-offload issue at its harder end:

```
same expert offload question as before, but i want the planner to decide the
split itself based on what actually fits, not a flag i have to tune. is that
doable?
```

Design, real:

```
we hold a cross process build lock while the engine starts, and a membership
lock per process that the kernel drops if we get killed. i want to know if
there's a sequence of crashes that leaves the machine in a state nothing
recovers from.
```

Architecture question:

```
if you were going to break the shared engine on purpose, where would you
attack it? i want to know what we haven't defended.
```

### Voice

These follow the way the prompts actually get typed here: lowercase, opening on
"so" or "i want" or "i think", stating the problem and the preference rather
than the implementation, and leaving the agent to find its own way in. Several
ask for judgment ("that seems wrong to me", "what's the right design here")
rather than dictating an answer, which is both realistic and a better test of
the model.

No "create a file named X", no "then stop", no restating acceptance criteria
back at the model.

Deliberate typos are left out even though they would be authentic. A viewer
cannot tell an intentional typo from a broken take, so it reads as a recording
mistake rather than as personality.

### Grading

Open prompts cannot be graded by an exit code, and forcing a checkable
criterion into the prompt is what made the old fixtures read like machine
input. Keep the prompts human and grade afterward against a rubric, judged by
Fable 5 so the judge is independent of the system under test.

Where the task is a fixed bug, the rubric is the actual root cause. For the
tier-4 lock question, a passing answer reaches filelock's thread-local deadlock
registry and the entry orphaned by a cross-thread release, not merely "add a
timeout". A model that proposes the workaround without the mechanism scores
partial. Where the task is open work, the rubric is whether the patch is one a
reviewer here would merge.

## Storyboard

One reel per model tier, same four agents each time.

**Beat 0, setup card (0:00 to 0:04).** One terminal, four panes tiled. Each
pane opens on its task in an editor view so the viewer can read all four before
anything moves. Header names the model and the hardware.

**Beat 1, launch (0:04 to 0:10).** Enter in each pane drops it into the coding
agent. Four agents come up against one engine. A strip along the bottom shows
GPU memory and the engine process count, so the viewer sees one engine serving
four clients rather than four engines.

**Beat 2, prompts typed (0:10 to 0:20).** Each prompt types out visibly. This
is the beat that makes the reel credible: the viewer reads a question they
would have asked.

**Beat 3, concurrent work (0:20 to 0:50).** All four agents work at once, each
with its own visual signature from the table above. The memory strip stays flat
while four sessions stream. This is the proof shot.

**Beat 4, payoffs (0:50 to 1:00).** Each pane lands its result: the cause
named, the test green, the refactor clean, the architecture answer written.
Freeze on the final frame.

### Length

A meaningful coding task takes minutes, and the house reel style is roughly 27
seconds at 1x with no speedup. Two artifacts per scenario resolve it:

- **Evidence capture**: full length, unedited, kept with the benchmark data.
  This is what backs the claims in the report.
- **Reel cut**: the beats above, with the working stretch cut to its most
  legible window rather than sped up, so the 1x text-quality rules still hold.

Which one leads on the site is a call to make after seeing the first evidence
capture.

## Recording setup

`recording/rose-pine.json` in this directory is the opencode theme, matched to
the official rose-pine mapping (functions rose, types foam, parameters iris,
plain variables text). The v2 reel used a community theme that swapped function
and type and painted every variable rose, which is why the highlighting looked
wrong. Copy this file to `$HOME/.config/opencode/themes/` on the pod and set
`"theme": "rose-pine"` rather than re-deriving it by hand.

Render rules for the reel cut (1x, font 18 ExtraBold, ship the direct gif) are
unchanged and documented with the recording kit.

## Session scope

That session runs stress tests and records them, nothing else. Per scenario:
run it, check the invariants, grade the output, record it. If a run surfaces a
failure, fix it, then re-record that scenario so every shipped reel reflects
the fixed build. No parallel feature work.
