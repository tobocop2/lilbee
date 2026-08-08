# Placement matrix

End-to-end validation of model placement against real hardware. Unit tests cover
the planner's arithmetic; this covers whether the plan is *true* — whether what
the planner promised actually loads, serves, and stays inside the memory it was
charged.

It exists because placement bugs are invisible to static review. A refusal that
should have been a launch, and a launch that should have been a refusal, both
look like correct code.

## What it checks

Per cell (one model on one hardware shape):

| Rule | Invariant |
|---|---|
| `plan-loads` | a plan is a promise the launch loads |
| `plan-sustains` | the published window survives consecutive full-window requests |
| `refusal-is-real` | a refusal must not be contradicted by forcing `num_ctx` |
| `estimate-not-under` | no device allocates materially more than it was charged |
| `estimate-not-wildly-over` | over-charging costs context elsewhere |
| `estimate-covers-devices` | every device the engine used was charged for |
| `oversize-spills` | a model larger than the GPUs loads by spilling, not by dying |
| `tight-group-has-no-ratio` | a best-effort placement leaves the split to the engine |
| `readback-missing` | a load whose per-device report was unreadable, so nothing was checked |
| `load-measured-alone` | another process held VRAM at launch, so the numbers describe a contended box |
| `refusal-is-testable` | a refusal a `num_ctx` pin could not contradict, which proves nothing |
| `cell-errored` | the cell raised; a crash is a failure, never an absent result |
| `result-unreadable` | a merged result this harness cannot read, rather than one silently dropped |

Across pairs of cells differing in exactly one knob, with no expected value
needed — only an order that must hold:

| Rule | Invariant |
|---|---|
| `monotonic-ctx` | more cards or more free VRAM never serves a smaller window |
| `monotonic-service` | the roomier side is never the one that gets refused |

`refusal-is-real` and `tight-group-has-no-ratio` each correspond to a real bug
this repo has shipped or nearly shipped. A rule that never caught anything is a
candidate for deletion, not a badge.

## Running it

Serial, on whatever the box has:

```bash
python -m tools.qa.placement_matrix run --out results/
```

Across pods, one shard each, then merge. Shards are disjoint and jointly cover
the matrix (asserted in `tests/test_placement_matrix.py`):

```bash
# pod 0                              # pod 1
... run --out results/ --shard 0/4   ... run --out results/ --shard 1/4
```

Copy every pod's `results/` into one directory and judge the union:

```bash
python -m tools.qa.placement_matrix report --out results/
```

Exit code is non-zero on any violation, and also when nothing ran — a matrix
that produced no results must not read as a pass. A cell that raises is written
out as a failed result rather than dropped, so a run whose cells all crashed
cannot merge into a clean report.

Results carry a schema version. Shards are merged across pods and possibly
across commits, so a file written by a different version is reported rather than
loaded with today's defaults, which would show fields nobody measured.

"Sustained" means the token counts came back, not that the request returned 200:
a round has to generate what it asked for and occupy most of the window, so a
silently truncated prompt or a one-token answer fails.

Useful flags: `--resume` (skip cells already recorded, so a reclaimed pod picks
up where it stopped), `--models tiny tight-split` (subset by key), `--max-cards`.

## Hardware

Cells declare how many GPUs they need; a host runs only what it can and records
the rest as skipped, so a heterogeneous set of pods can share one matrix. Card
counts are produced by masking `CUDA_VISIBLE_DEVICES`, so a single 4-GPU box
covers the 1-, 2-, 3- and 4-card layouts rather than needing four machines.

Cells that ask for a resident tenant (uneven capacity) are skipped unless the
host supports allocating ballast; they are declared in the matrix so the gap is
visible rather than absent.

## Models

One per decision boundary. Adding a model that reaches a branch already covered
costs a download and proves nothing.

| Key | Boundary |
|---|---|
| `tiny` | single card with room to spare; multi-slot |
| `kv-starved` | weights fit one 24 GiB card, a usable context does not: forces a split |
| `tight-split` | needs two 24 GiB cards, lands in the tight group, serves anyway |
| `spill` | exceeds the box: must load by spilling to system memory; also MoE |
| `embed` | co-tenant whose reservation is held back from chat |

`spill` is a 133 GiB download and is the only cell that exercises
`oversize-spills`. Run at least once per release on a box it genuinely exceeds;
`--models` skips it the rest of the time.

Pull the models before running; the harness plans and launches, it does not
download.
