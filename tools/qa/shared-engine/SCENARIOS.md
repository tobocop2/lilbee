# Coding-agent stress scenarios and reels

The plan for the next session: run four coding agents concurrently against one
shared engine, on models people actually reach for, and record each run. Every
scenario is both a load test and a filmable demo.

This file is the contract for that session. The prompts here are the prompts
that get typed on camera.

## Models

No small models. A 4B or 8B model cannot carry a real coding task, and a demo
where the agent flails is worse than no demo.

| Tier | Model | Size at Q4 | Hardware | Role in the story |
|---|---|---|---|---|
| Workhorse | Qwen3-Coder-Next, 80B-A3B | ~49 GB | 1x A100 80GB or H200 | Purpose-built for agentic coding, ~70.6% SWE-bench Verified. 3B active means it streams fast enough to read on camera. Single GPU keeps the shared-engine result free of multi-GPU variables. |
| Frontier | MiniMax-M2.1, 230B-A10B | ~130 GB | 2 GPUs | Tuned for coding agents. Doubles as the multi-GPU flagship: four agents across two cards on one shared engine. |
| Giant | Kimi K2.6, 1T-A32B | ~340 GB at UD-Q2_K_XL | 4x H200 | The SWE-bench leader at ~80.2% verified. The "your own hardware runs the best open model there is" reel. |
| Reference | Qwen3.6-35B-A3B, Q8_0 | 37 GB | 1x A100 80GB | Already measured. Free row, no new pod time. |

Kimi is the most expensive row by a wide margin and puts a 2-bit quantization
on camera, so record it last, once the harness is proven on the cheaper tiers.
If its output quality visibly suffers at 2-bit, say so in the report rather
than shipping a flattering cut.

Two practical gates before pulling anything:

- **Tool-call plumbing is not uniform.** Qwen3-Coder is already a verified
  family in this repo's opencode matrix, so Qwen3-Coder-Next should inherit
  working tool dispatch. MiniMax and Kimi are new families and may need a
  response-parser schema before they dispatch at all. Verify dispatch with the
  existing matrix harness first; fixing a parser is dev work and does not
  belong in a recording session.
- **Size against the real planner.** Run the gguf-parser planner per candidate
  and confirm slots, per-slot context, and KV budget on the target hardware. A
  4-bit 80B gains roughly 7 GB going from 4k to 256k of context, and this
  harness wants four slots at a context an agent can hold a session in. Do not
  shrink context to make a model fit; pick different hardware.

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
Qwen3-Coder-Next to solve a cross-thread lock deadlock on camera produces a
flailing demo; asking Kimi to extract a duplicated helper wastes the model.

The bug-hunt and refactor tasks are drawn from problems that actually happened
in this repo, which is what makes them good stories and also makes grading
exact, since the real fix is known. **Pin each run's checkout to the commit
immediately before the real fix landed**, or the agent will read the answer out
of git history.

### Tier 1, Qwen3-Coder-Next: well-scoped, single subsystem

```
something's off with catalog search when i filter by quant. models that
definitely have that quant are getting dropped from the results. can you dig
into why?
```

```
i want lilbee model rm to warn me first if the model i'm deleting is one i've
got configured for a role. right now it just deletes it and the next run fails
with a missing model. add the guard and a test for it?
```

```
the vram fits check is copy pasted in a few places in the placement code and
they've drifted apart. can you pull it into one helper and use it everywhere?
```

```
how does lilbee decide it can share an engine with another process instead of
starting a second one? walk me through what it checks.
```

### Tier 2, MiniMax-M2.1: cross-file, needs judgment

```
here's a weird one. after the engine restarts, a long running lilbee serve
keeps returning errors for every chat, but if i run the same query through the
cli it works fine every time. same box, same engine. why would the resident
process not recover when a fresh one does?
```

```
when someone launches an agent session we bump the context up to a floor, but
if the engine was already warmed at something smaller the user just gets a
warning and a worse experience. i'd rather it either warm to the right size or
tell them clearly what to do. can you work out the right behaviour and build
it?
```

```
we look up the engine directory in a few different places and i think the rules
have drifted between them. can you find all the spots and make them agree?
```

```
if two people on the same box are running lilbee with different models
configured, what actually happens? i want to understand whether they collide.
```

### Tier 3, Kimi K2.6: genuinely hard, concurrency and design

```
our integration tests hang, but only in ci and only sometimes. it's a file lock
that gets acquired on one thread and released on another, and after that a
later acquire never returns. i've been staring at this for a while. can you
work out what's actually happening?
```

```
when the engine gets killed and rebuilt in place, requests that land during the
rebuild fail instead of waiting. i'd rather that window turned into latency
than errors, but i don't want a request hanging forever either if the rebuild
is genuinely broken. what's the right design here, and can you build it?
```

```
there's a rule in the engine acquisition path about when we replace an existing
engine versus starting a second one alongside it. i think the rule is wrong in
at least one case and it costs us a duplicate model load. can you reason
through the cases and fix it?
```

```
walk me through every way the engine acquisition ladder can fail when two
processes race each other, and tell me which of those we actually handle.
```

### Voice

These are written the way a person asks: lowercase, conversational, stating a
problem rather than issuing instructions, and leaving the agent to work out
where to look. No "create a file named X", no "then stop", no restating the
acceptance criteria back at the model.

Deliberate typos are left out even though they would be authentic. A viewer
cannot tell an intentional typo from a broken take, so it reads as a recording
mistake rather than as personality.

### Grading

Open prompts cannot be graded by an exit code, and forcing a checkable
criterion into the prompt is what made the old fixtures read like machine
input. Keep the prompts human and grade afterward against a rubric, judged by
Fable 5 so the judge is independent of the system under test.

Because the harder tasks come from real fixed bugs, the rubric is the actual
root cause. For the tier-3 lock question, for example, a passing answer reaches
the thread-local deadlock registry and the orphaned entry, not merely "add a
timeout". A model that proposes the workaround without the mechanism scores
partial.

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
