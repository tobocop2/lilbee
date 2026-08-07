# Reel pipeline: architecture and hard-won rules

This is the working knowledge behind `demos-src/asciinema/`. It exists because the same
mistakes were made more than once, and the cost each time was hours of recording, a pod,
or a set of demos that shipped looking wrong. Read the rules before changing the pipeline.

## What a reel is

A reel is one recorded terminal session, rendered to a gif and an mp4, that demonstrates
one claim. It has four artefacts:

| Artefact | What it is | Why it is kept |
|---|---|---|
| `.cast` | asciinema recording, plain text | The source. A reel with a cast can be re-rendered without a machine, a model or a pod. A reel without one is dead the moment the pipeline changes. |
| `.gif` | what the README embeds | GitHub renders it inline |
| `.mp4` | what the marketing site embeds | The site uses `<video>`, not `<img>` |
| `.png` | poster frame | Video poster, and a still for a card |

**Always ship the cast.** Two pod reels were stuck for weeks because their casts were
lost and re-recording meant renting two GPUs again.

## The shape of the pipeline

    record  ->  render  ->  deseam  ->  trim/compress  ->  verify  ->  gate  ->  publish
    (tmux)      (agg)                    (frametrim)      (2nd pass)  (gates)

Recording is a **timing measurement**: the reel's frame rate is the application's own
repaint cadence. Everything after it is a deterministic function of the cast.

That split is the single most useful fact about this pipeline:

* **Recording must be serial.** Two reels recording at once contend for CPU and memory and
  depress the repaint rate being measured. Concurrent recordings have also taken a laptop
  down with a kernel panic.
* **Rendering can be parallel.** Measured on a full set: recording is 26% of wall time,
  rendering 74%. Use `--record-only` on a pod, pull the cast back, render locally, and run
  the render phase 3-way parallel.

## Rules that were learned the hard way

### A fixer and a checker separated by a transform will disagree forever

This cost most of a night, three times over. The wait compressor scored frames *before*
deseaming while the gate scored them *after*; repairing seams erases small pixel
differences, so waits the compressor handled separately arrived at the gate welded into
one long one. Raising the compressor's threshold twice did not fix it. Reordering deseam
ahead of the trim fixed that instance and exposed the next: saving a gif quantises it to
a 256-colour palette, which does the same thing again.

**The rule:** a stage that fixes something must measure what it *wrote*, not what it held
in memory. `compress_waits` exists solely to re-check the written file. Where two stages
must share a threshold, the fixer's margin points inward -- compress from 4.5s against a
6s limit, never from 6s.

### Overlap means clip, never discard

Written four times in this file's history, wrong three of them. When a detected span
overlaps a protected or already-planned span, trim the covered part and keep the rest.
Discarding the whole span dropped a genuine 2.6s slow stretch because it began 0.1s
before an adjacent window ended, and later let 10.9s of dead air ship.

### Do not compress what the reel exists to prove

`PROTECT_WINDOWS` names spans that neither the timelapse nor the hold clamp may touch. A
launch reel exists so a viewer can see how long starting takes; compressing it answers the
question the reel was recorded to ask. cold-start protects ~16s, later-start ~1.6s, and the
two sit side by side in the README precisely so those numbers can be compared.

The gate reports protected spans rather than failing them, and prints what it skipped, so
a protected span cannot quietly hide an unrelated problem.

### Protect driver motion from *slowness* compression, not from *wait* detection

Driver motion (typing, scroll bursts) is what `motion_fps` measures, so thinning it
destroys the measurement. But a reel pressing `j` sixty times against a log with nothing
left to scroll produces one-second frames that barely change -- 10.9s of that shipped while
being counted as protected motion. If the screen is not moving, it does not matter that a
key is being pressed.

### An untested row is not a pass, and an absent subject is not a failed measurement

`UNTESTED` blocks a reel. But distinguish two cases:

* The driver moved and the sample was too small to judge -> genuinely untested, fix the reel.
* The reel contains no stretch of that kind at all -> the subject is absent, and the row
  passes with the reason stated.

Where a rate genuinely cannot be met, the reel declares it in its own source with a written
reason and a named control. `first-start` declares `COLD_BY_DESIGN`: it deletes the unpack
cache, so the app repaints slowly enough that most driver frames exceed the hold cap. Its
control is `later-start` -- same binary, same question, warm cache -- which measures 20fps
and is held to the floor normally. Never waive a row without naming what proves the waiver.

### Gates must be falsifiable

Two gates in the first draft could not fail. `selftest()` asserts every threshold against a
deliberately broken input: a choppy gif, dimmed text, spliced dead air, injected seams, a
spinner-only stretch, a slow app section. If a row cannot go red on demand it is
decoration, and worse than nothing, because it launders a bad take as verified.

### Content beats catch what property checks cannot

A reel can pass colour, frame rate, seams and size while showing the wrong thing. Shipped
examples: a sessions reel listing one conversation instead of three, a palette reel whose
`/add` silently did nothing, a placement reel that toggled nothing. `BEATS` assert that
named things appeared **in order**.

Beats are not enough either. Both review rounds found defects in reels that passed every
gate, and every one was a defect in the reel's own declared expectations. Grading has to
include a pass that ignores the script: watch it as someone who has never seen the product,
say what it claims, and say whether it delivered.

## Traps in the application under test

These are properties of lilbee's TUI, not of the pipeline. Each cost a take or several.

* **It boots in INSERT.** Pressing `i` inserts a literal character rather than switching
  mode, which is where a stray `i`, typed and deleted, appeared before every question in a
  whole shipped set. Establish the mode before pressing anything. From NORMAL with focus in
  a drawer, `i` is swallowed and the mode never changes, so pressing and hoping is wrong in
  both directions.
* **`personal encyclopedia` is the empty-state hint,** not "chat is up". It disappears once
  a data root has history, so reels reusing a staged root pass on the first take and fail on
  every one after. Wait on `Slash commands`, which is in the footer either way.
* **A fresh data root opens the first-run wizard,** which no chat marker matches. Dismiss it.
* **The mode chip is not always visible.** The placement drawer takes the space the tab
  strip renders it in. Track the mode instead of requiring the chip, and reconcile whenever
  one is on screen.
* **"all caught up" is what the Task Center shows when it holds nothing.** Accepting it as a
  completion signal matches *before* the task registers. Wait for the task to appear first.
  A crawl reel read it 88 seconds in, asked its question against an empty index, and shipped
  an answer with no citation.
* **The provider label is rendered text, not the ref prefix.** The catalog shows
  `lm studio`, lowercase with a space -- not `lm_studio` and not the display constant
  `LM Studio`. Waiting on either timed out against a row that was on screen the whole time.
* **Wait for a row before scrolling past it.** LM Studio's models are the last entries in
  Library and were already visible when the tab opened; scrolling first carried the viewport
  past them.

## Recording on a pod

Two reels need two GPUs. What that takes:

1. **Provision** a 2x4090 (48GB total holds a 70B at Q4_K_M with context headroom).
2. **Arm a deadman timer immediately.** A pod once idled 9.3 hours after a session died.
3. **SSH.** The direct port is often unreachable; the RunPod proxy needs a PTY and ignores
   command arguments, so commands go in on stdin. `podrun.sh` does this. Its host id must
   come from the API per pod -- a stale id from a previous pod silently talks to nothing.
4. **Download with Xet.** `HF_HUB_ENABLE_HF_TRANSFER` is deprecated and does nothing; the
   hub says so in its own warning. `HF_XET_HIGH_PERFORMANCE=1` took a 42GB pull from
   ~0.4 GB/min to ~4 GB/min.
5. **Install lilbee.** Easiest is the standalone binary --
   `lilbee-linux-x86_64-cu125` from GitHub releases -- which bundles the Python runtime,
   llama.cpp and the extras in one file. If you use pip instead, note that the extras are
   two halves of one thing and each fails late while blaming something else:
   `lilbee[engine]` installs `llama-server` without the CUDA runtime, `lilbee[cuda]`
   installs the runtime without the engine. You want
   `pip install --pre 'lilbee[engine,cuda]' --extra-index-url https://lilbee.sh/cu125/`;
   the engine wheel is published on lilbee.sh rather than PyPI, which is why the index is
   needed. Missing engine surfaces as "No embed model server is running" during ingest;
   missing runtime surfaces as "enumerated no CUDA-capable device".

6. **Preflight CUDA before anything else.** `python3 -c "import torch;
   print(torch.cuda.is_available(), torch.cuda.device_count())"` plus one allocation. A
   community-cloud pod once reported two 4090s to nvidia-smi while CUDA init failed with
   "unknown error", and twenty minutes went into diagnosing lilbee on a broken host. Ten
   seconds up front tells you whether the machine is worth installing on.
7. **Stage the data root by hand.** Pod reels do not write their own `config.toml` the way
   local reels do. `tui-manual-placement` additionally needs its corpus indexed *before* the
   reel runs, because it deliberately never ingests.
8. **Record only.** Pull the cast back and render locally.
9. **Power the pod off.**

10. **One deadman, and cancel the old one first.** Re-arming without cancelling leaves the
    original timer live: it fired on its old schedule and killed two takes mid-run. Size it
    against the real work, not optimism -- a 42GB pull plus an ingest eats most of an hour
    before a reel starts.

11. **Move files with `runpodctl send` / `receive`.** The proxy's PTY mangles long lines;
    two hand-rolled base64 transfers came back corrupt.

12. **Run takes under `nohup setsid`, never inside tmux.** When the download session exited
    it took the tmux server with it, and a queued take went nowhere. On pod *resume* only
    `/workspace` survives, so system packages and symlinks need reinstalling.

## Cost discipline

Check the environment before recording, not fifteen minutes into a take. Every long failure
chain in this project traced to something absent rather than something wrong, and each
surfaced as a misleading timeout deep inside a reel:

* A missing chat model opens the install wizard, and the reel dies reporting a missing chat
  marker.
* A missing model that still boots waits the full generation timeout for a citation no model
  will produce.
* Processes orphaned from a killed take hold the engine port; one pair had been up 43 hours.
* A full disk fails writes with no useful error anywhere near the cause.

Audit configured models against installed ones, and clear reel-root orphans, before the
first take.
