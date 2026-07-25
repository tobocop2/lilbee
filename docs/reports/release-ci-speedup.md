# Release pipeline speedup

Audit of `release-candidate.yml` and its called workflows against measured run data
(runs 30003020138 and 29960035783). Every number below is either measured from `gh`
job/step timestamps or explicitly labelled unverified.

**Baseline correction first.** Run 30003020138's headline "214.6 min" is `run_attempt: 2`.
Attempt 1 ran 11:24:42 to 16:29:35 = **305 min**, matching reference run 29960035783 (305.2 min).
The "second wave started 16:31" reading is a re-run artifact, not queueing. Any analysis priced
off a 20:03 finish is priced off a manual re-run of two failed jobs. Treat **~305 min** as the
clean-run baseline.

---

## 1. Where the hours actually go

The release is four Windows Nuitka compiles. Nothing else is close.

The terminus of a clean run is `build-binaries (windows-latest, compat, vulkan)` at 240.1 min or
`build-cuda (windows-2022, cu125)` at 293.6 min, whichever tails. Both are ~95% one step:
Nuitka compiling ~9056 C files with MSVC at `-j4`, zero cache hits. Two more Windows legs sit
within 5 min of them. Everything else (all Linux CUDA, macOS, manylinux, multigpu, wheels,
smoke) finishes 2 to 4 hours earlier and is off the path.

Critical-path job, `build-cuda (windows-2022, cu125)`, 293.6 min:

| step | min | cacheable? | today |
|---|---|---|---|
| Install CUDA Toolkit (Windows) | 4.9 | skippable on engine cache hit | runs unconditionally |
| Bundle llama-server | 62.5 | yes, engine cache | miss, built from source |
| Build executable: Nuitka Python phase | 19.3 | partly | cold |
| Build executable: Scons C compile (9056 files) | 176.4 | yes, clcache | **0 hits / 9056 misses** |
| Build executable: link + onefile compression | 27.4 | onefile cache only | cold |
| smoke / upload / attach | 1.7 | no | fine |

The sibling `build-binaries (windows-latest, compat, vulkan)` at 240.1 min has the same shape
minus the CUDA engine build: 234.4 min of `Build executable`, also 0/9059 clcache hits.

Three numbers ARE the release:

1. **176 min of MSVC C compilation, four times over, never once cached.** The clcache mechanism
   exists in `release.yml` and has produced zero hits in every run sampled (07-14, 07-16, 07-18,
   07-19, 07-20, 07-22, 07-23). Not a key bug: the cache entry is LRU-evicted from the 10 GB
   repo cap before the next release reads it.
2. **62.5 min of `Bundle llama-server`** on each CUDA leg, because `build-cuda-executables.yml`
   is never run on a non-tag ref, so its engine cache is never written to a restorable scope.
3. **27 min of onefile zstd-22 compression** per leg, content-addressed and cacheable, cache dir
   never persisted on Linux/macOS.

The repo is at **9.58 GiB of GitHub's 10 GiB per-repo cache cap**, 5.03 GiB of which is
tag-scoped entries no future run can ever read. That is the mechanism behind (1) and (2).
Caching is not "not working" because it is misconfigured; it is working and then being evicted
by garbage.

---

## 2. Course of action

Ranked by minutes off the critical path per unit of risk and effort, not by minutes alone.

| # | change | file | min off critical path | risk | verified? |
|---|---|---|---|---|---|
| 1 | Reclaim cache budget: purge tag-scoped entries, fix Playwright/HF key churn, drop Windows/macOS ollama caches | `ci.yml`, one-off `gh cache delete` | 0 direct, precondition for 2-6 | low | yes (cap + entry list measured) |
| 2 | Stop saving engine caches on tag refs (restore/save split + `github.ref_type != 'tag'`) | `.github/actions/bundle-llama-server/action.yml:35`, `build-multigpu.yml:84` | **12** | low | **yes, verified; original 62.5 claim was ~5x too high** |
| 3 | Warm `build-cuda-executables.yml` on main | `warm-executables-cache.yml:28` | **48** on a Linux-tail run, ~12 on a Windows-tail run | low-med | **yes, verified; 50.4 corrected to 48.2** |
| 4 | Export `CCACHE_DIR` so Nuitka stops redirecting to `~/.cache/Nuitka/ccache` | `build-cuda-executables.yml:135`, `release.yml:218` | **70** on the Linux tail leg, 0 on Windows | low | **yes, verified; only as a 3-part package with #1 and #3** |
| 5 | Drop `litellm.proxy` from the compiled closure (581 dead modules, slowest TU in the build) | `tools/wheel-build/build_lilbee_binary.sh:133` | ~18 per leg, lands on the Windows terminus, cold-build | med | unverified |
| 6 | Add the Windows Nuitka cache to `build-cuda-executables.yml` + persist `NUITKA_CACHE_DIR` on all OSes | `build-cuda-executables.yml:181`, `release.yml:225` | 0 today, ~100-150/leg if clcache warms | med | unverified, and the mechanism has **never** hit in this repo |
| 7 | Persist the onefile-compression cache | `release.yml:225` extended to Linux/macOS | ~20 | med | unverified |
| 8 | Add a cu121 cell to `build-multigpu` so its engine key is warmable | `build-multigpu.yml:38-52,63,89` | ~47 when cu121 tails | low | unverified |
| 9 | Gate the CUDA toolkit install on an engine-cache probe | `build-cuda-executables.yml:112` | 3-5, only after #3 | med | unverified |
| 10 | Weekly `schedule:` on `warm-engine-cache.yml` | `warm-engine-cache.yml:9` | 0, insurance against 7-day idle eviction | low | unverified |
| 11 | `compression-level: 0` on executable artifact uploads | `build-cuda-executables.yml:253` | 0.2 | low | unverified |
| 12 | Make the CI gate wait for an in-flight run instead of duplicating lint+test | `release.yml:57` | 0 | low | unverified |

Items 2, 3 and 4 are the only verified ones. Item 4 is worth 70 min but **only** as a package
with 1 and 3; as a standalone one-liner it is worth exactly 0, forever.

---

## 3. Tier 1: do these first

Apply in this order. 1 before 2, 2 before 3, 3 before 4.

### 1.1 Reclaim the cache budget

Measured: 35 entries, 10,283,123,387 bytes against a 10 GiB hard cap. 5.03 GiB is on
`refs/heads/refs/tags/v0.6.90b420.dev726|dev727` and is unrestorable by construction.

```bash
gh cache list --limit 100 --json id,ref --jq '.[]|select(.ref|test("refs/tags/"))|.id' \
  | xargs -n1 gh cache delete
gh cache delete ollama-bin-Windows-0.31.2
gh cache delete ollama-models-Windows-qwen3-nomic-v1
gh cache delete ollama-models-macOS-qwen3-nomic-v1
```

The ollama Windows/macOS entries (2.7 GiB) cache a path that does not run: `qa-matrix.yml:155`
documents those lanes skipping via "daemon not reachable". Gate those cache steps in `ci.yml`
on `runner.os == 'Linux'`.

Two version-bump-in-the-key antipatterns mint a new cache generation on every release:

```yaml
# ci.yml:316-322 - key on the browser revision, not the lockfile. Drop restore-keys.
-         key: playwright-${{ runner.os }}-${{ hashFiles('uv.lock') }}
-         restore-keys: playwright-${{ runner.os }}-
+         key: playwright-${{ runner.os }}-chromium-v1
```

```yaml
# ci.yml:339-344 - delete outright. Never populated (models come from the ci-models
# release via restore-ci-models), and keyed on pyproject.toml so it churns every bump.
-      - name: Cache HuggingFace models
-        uses: actions/cache@v6
-        with:
-          path: ~/.cache/huggingface/hub
-          key: hf-models-${{ runner.os }}-${{ hashFiles('pyproject.toml') }}
-          restore-keys: hf-models-${{ runner.os }}-
```

Verify headroom before proceeding:

```bash
gh api repos/:owner/:repo/actions/cache/usage --jq .active_caches_size_in_bytes
```

Target: under 5 GiB before adding any new cache consumer.

### 1.2 Stop saving engine caches on tag refs (verified, 12 min)

`actions/cache@v4` has no save condition, so every tag run writes ~2.45 GiB of engine caches
scoped to a tag ref that no future run can read. The ccache step at
`build-cuda-executables.yml:130` already carries the correct guard; the engine cache never got it.

In `.github/actions/bundle-llama-server/action.yml`:

```yaml
     - name: Restore the built engine
       id: engine-cache
-      uses: actions/cache@v4
+      uses: actions/cache/restore@v4
       with:
         path: packaging/engine-wheel/lilbee_engine/bin
         key: engine-${{ inputs.cache-env }}-${{ inputs.backend }}-tk${{ inputs.toolkit-version }}-llcp${{ inputs.llama-cpp-version }}-${{ hashFiles('engine-versions.env', 'tools/wheel-build/build_llama_server.sh', 'tools/wheel-build/cmake_args.sh', 'tools/wheel-build/install_gpu_toolkit.sh') }}
```

and after `Build the self-contained engine`:

```yaml
    # Tag-scoped saves are unrestorable by future runs and evict the warm ones.
    - name: Save the built engine
      if: steps.engine-cache.outputs.cache-hit != 'true' && github.ref_type != 'tag'
      uses: actions/cache/save@v4
      with:
        path: packaging/engine-wheel/lilbee_engine/bin
        key: ${{ steps.engine-cache.outputs.cache-primary-key }}
```

Apply the identical split to `build-multigpu.yml:84-89`, save step after line 139.
Also fix the comment at `action.yml:34`, which currently claims "tag-scoped saves are still
restorable across a moved tag". That is false and is the origin of the bug.

Three notes verified during review:

- `actions/cache/restore@v4` sets the same `cache-hit` output, so the four downstream
  `if: steps.engine-cache.outputs.cache-hit != 'true'` guards are unaffected.
- The tag ref really is a tag: run 30003020138 is `event=push, head_branch=v0.6.90b420.dev726`,
  and no branch named `refs/tags/v...` exists. The guard will fire.
- Merging this fires `warm-engine-cache.yml` (it triggers on pushes touching
  `build-multigpu.yml`), so it re-primes itself. Without that it would sit inert.

Add a weekly schedule to `warm-engine-cache.yml` at the same time, since GitHub evicts caches
unaccessed for 7 days and the workflow has no cron today:

```yaml
  schedule:
    - cron: '0 5 * * 1'
```

**Expected effect:** frees 3.82 GiB. Run-level 12 min. The value is durability, not the 12 min.

### 1.3 Warm `build-cuda-executables.yml` on main (verified, 48 min)

`warm-executables-cache.yml` calls `release.yml` only. Nothing anywhere runs the CUDA executable
matrix on a non-tag ref, and `build-cuda-executables.yml:130` correctly disables its ccache save
on tags. Net: those keys are written never and read never. Confirmed: zero
`ccache-nuitka-ubuntu-22.04-cu12*` entries exist, and every CUDA leg logs `No cache found.`

```yaml
jobs:
  warm:
    uses: ./.github/workflows/release.yml
    secrets: inherit

  warm-cuda:
    uses: ./.github/workflows/build-cuda-executables.yml
    secrets: inherit
```

No `with:` block. All three `workflow_call` inputs are optional; an empty `release_tag` keeps the
run in artifacts-only mode and both attach steps skip. Do **not** pass `skip_tests: true`: the
smoke step costs 14 s, the cells are already `continue-on-error`, and `actions/cache` saves in
its post step even after a failed step, so keeping it is free and preserves the pre-tag
validation the workflow's own header claims to provide.

**Expected effect, measured warm-vs-cold on a CUDA cell in the same run:** `Bundle llama-server`
50.4 min cold (job 89268750496) vs 2.2 min warm (job 89268750737, which hit the 686 MB cu124
entry a sibling had just saved) = **48.2 min**. On a Windows CUDA cell the cold bundle is 62.5 min
against a 440-465 MB cache, so that leg saves ~59 min.

**Truncation caveat, stated plainly:** on a clean Windows-tail run, shortening both build-cuda
Windows legs by ~59 min moves them to ~15:27 and ~15:15, at which point
`build-binaries (windows-latest, compat, vulkan)` at 16:14:55 becomes the terminus and is
untouched by this change. Run-level realisation is then ~12 min, not 48. The full 48-59 min is
realised only when a build-cuda cell tails, which it did in run 29960035783 and in attempt 2 of
30003020138. Which cell tails is not stable across runs.

Note the cu121 cell can never be warmed by `warm-engine-cache.yml` at all: `build-multigpu.yml`
has no cu121 row and threads an empty `inputs.llama_cpp_version` into the key, while
`build-cuda-executables.yml:57` pins `llama-cpp-version: "0.3.30"`. Fix with item 8 below.

Risk correction: the repo is **public** (`gh repo view` visibility PUBLIC), so runner minutes are
free. The real cost is ~22 runner-hours of concurrency slots on the weekly warm run, and ~3.0 GiB
of new main-scoped engine entries. Steady state is roughly cap-neutral (those bytes are written
at tag scope today and thrown away), but the transition will trigger LRU eviction. Do 1.1 first.

### 1.4 Export `CCACHE_DIR` (verified, 70 min on the Linux tail)

Root cause, read from the Nuitka 4.1.3 sdist that CI actually runs
(`nuitka/build/SconsCaching.py:167-172`):

```python
# Unless asked to do otherwise, store ccache files in our own directory.
if "CCACHE_DIR" not in os.environ:
    ccache_dir = getCacheDir("ccache", create=True)
    setEnvironmentVariable(env, "CCACHE_DIR", ccache_dir)
```

`hendrikmuhs/ccache-action` writes a config file only (`ccache --set-config=cache_dir=...`);
`grep CCACHE_DIR` over a full job log returns nothing. ccache env vars outrank the config file,
so Nuitka's 9031 objects land in `~/.cache/Nuitka/ccache` and die with the runner. Proof it is
inert: the manylinux vulkan leg restored a warm ccache, then reported `Hits: 0 / 0, Misses: 0`
from `ccache -s` while Nuitka in the same job logged
`Cached C files (using ccache) with result 'cache miss': 9031`. The 411 hits the cu125 leg does
record are the llama.cpp/ggml cmake build, which honors the config file.

Add after the ccache step in `build-cuda-executables.yml` (~line 135) and `release.yml` (~line 218):

```yaml
      # Nuitka redirects CCACHE_DIR to its own cache dir unless it is already set
      # (nuitka/build/SconsCaching.py), so ccache-action's config-file-only setup
      # never sees Nuitka's objects. Pin it to the directory the action caches.
      - name: Point ccache at the cached directory
        if: runner.os != 'Windows'
        shell: bash
        run: echo "CCACHE_DIR=${GITHUB_WORKSPACE}/.ccache" >> "$GITHUB_ENV"
```

The path is correct on both bare runners (`/home/runner/work/lilbee/lilbee/.ccache`) and inside
the manylinux container (`/__w/lilbee/lilbee/.ccache`); both are `$GITHUB_WORKSPACE/.ccache`.
Safe against `Reclaim build-intermediate disk`, which removes only `dist/*` and `nuitka-cache`
and already documents "ccache stays".

**Do not bump `max-size` to 5G as originally proposed.** 9031 Nuitka objects extrapolate to
~2.5-3 GB compressed per leg; 5G x 4 Linux CUDA legs would evict the `engine-ubuntu-22.04-cu12x-*`
entries whose restore is what makes `Bundle llama-server` 2 min instead of 50. Raise it only for
the two tail legs, and only after 1.1 has freed real headroom.

**Expected effect on the Linux tail leg (cu121, 211.3 min):** the only addressable window is the
96.1 min Scons C phase. Warm ceiling ~12 min; at a realistic ~90% hit rate against a main-branch
cache built at a different commit, ~15-20 min. Discounted for ccache's internal LRU: **70 min**.
Combined with 1.3 the cu121 job goes 211.3 to ~93 min.

Zero on Windows: MSVC does not use ccache. Do not fan the export out to macOS and manylinux in
the same change; those legs are 2 to 4 hours off the path and would only consume cap.

### 1.5 Drop `litellm.proxy` from the closure (unverified, ~18 min, hits Windows)

The only Tier 1 item that shortens the Windows terminus without depending on any cache.

`build_lilbee_binary.sh:133` passes `--include-package=litellm`, which pulls in all 2084 modules.
581 of them are `litellm/proxy`, 28% of the package, and nothing in lilbee imports it
(`grep -rn 'litellm.proxy' src/ tools/` hits only that build line; every call site does a bare
`import litellm`). Nuitka itself objects: `anti-bloat: Undesirable import of 'pytest' in
'litellm.proxy.guardrails...'`. The single slowest translation unit in the entire pipeline is one
of them: `module.litellm.proxy._types.c took 377.56 seconds`, which is also what triggers
`Slow C compilation detected ... scalability problem`. At `-j4` a 6.3-minute file is a serial tail.

```diff
     --nofollow-import-to=*.tests.* \
+    --nofollow-import-to=litellm.proxy \
     --nofollow-import-to=tkinter --nofollow-import-to=_tkinter \
```

This genuinely overrides `--include-package`: `Recursion.py` checks the `no_case` patterns and
returns "instructed by user to not follow" at line 300-303, before the `any_case:` branch at 306
that `--include-package` feeds.

580 of 9056 modules = 6.4% of the closure, plus a disproportionate share of compile time given
the slowest TU is one of them, plus a share of the 15.2 min Python optimization phase. ~18 min is
a mid estimate and is **unverified**.

Risk: medium. litellm does dynamic provider dispatch, so a runtime path could conceivably reach
it. Land it on a `warm-executables-cache` dispatch and read the smoke output before tagging. The
Vulkan/Metal legs run the full `tools/qa/artifact_smoke.sh` sweep including the ask path.

### 1.6 Small, free, do them while you are in there

```yaml
# build-cuda-executables.yml:253 and release.yml:458 - onefile payloads are
# already compressed; deflating them again buys nothing.
+         compression-level: 0
```

Measured at 21 s for 635 MiB, so this is ~10 s. Reported because it was asked about explicitly:
the logs do not support the premise that artifact compression costs minutes.

`release.yml:57`'s CI gate counts only completed successes, so a tag pushed alongside a main push
sees CI still running, concludes `ci_passed=false`, and re-runs the 9-cell lint + test matrix on
the identical SHA. Poll instead of duplicating. Zero critical path, two runner slots.

---

## 4. Tier 2: structural

### 2.1 Compile the Windows closure once instead of four times (largest available win)

The compiled C closure is backend-independent. Comparing full Nuitka command lines between the
cu125 and vulkan Windows legs, the only differences are `--output-filename` and the
`--include-data-files=...lilbee_engine/bin/*` list. Both are data and link-stage inputs, not
translation units. File counts confirm it: 9056 (win cu125), 9059 (win vulkan), 9034 (linux
cu125), 9031 (manylinux vulkan). The only real compile-flag difference in the whole matrix is the
Linux compat cells' `-march=x86-64-v2`; Windows compat sets no CFLAGS at all
(`release.yml:146-150`: "MSVC's default baseline is already conservative").

So the pipeline compiles the same ~9057 C files four times on Windows, at ~230 min each, to
produce four executables that differ only in which engine DLLs are embedded.

**Payoff:** removes three of four Windows compiles. The terminus becomes a single Windows job, and
the pipeline stops being gated by a 4x-duplicated 176-minute compile.

**Effort and feasibility:** real work, and the key question is unverified. Nuitka has no documented
"package this prebuilt standalone dist into a onefile" entry point, so this likely requires either
(a) building `--standalone` once, uploading the dist as an artifact, then invoking Nuitka's onefile
packaging step directly per backend, or (b) shipping the engine DLLs beside the executable instead
of inside it, which changes the single-file distribution promise. Investigate (a) against
`nuitka/build/OnefileBootstrap` and `OnefileCompressor.py` before committing to it.

Interim, cheap, and safe: give the Windows compat and non-compat legs one cache namespace (they
compile byte-identical C), and split the Linux ones (they do not, and today they share a key).

```yaml
# release.yml:230-231, 301 - key on compile inputs, not asset name
          key: nuitka-${{ matrix.os }}-${{ matrix.backend }}-${{ github.run_id }}
          restore-keys: nuitka-${{ matrix.os }}-${{ matrix.backend }}-
# release.yml:215, build-cuda-executables.yml:133 - add the baseline so the
# compat and non-compat Linux legs stop colliding
          key: nuitka-${{ matrix.os }}-${{ matrix.backend }}${{ matrix.march && format('-{0}', matrix.march) || '' }}
```

This halves the Windows Nuitka cache footprint, which is what makes 2.3 affordable inside the cap.

### 2.2 Move engine binaries off the Actions cache onto GH release assets

The engine binaries are perfectly immutable: `engine-versions.env` pins llama-cpp-python 0.3.30,
llama-swap v223, gguf-parser v0.25.0, and `action.yml:31-34` already asserts "pinned sources, so
identical keys mean byte-identical binaries". They are ~3.5 GiB of a 10 GiB cap that has no
business being there.

Publish them to a content-addressed `engine-<pin-hash>` release on push to main, using the same
`hashFiles(...)` expression the cache key uses today. Then in `bundle-llama-server`, try
`gh release download` first, fall back to the cache, fall back to the source build. Keeping the
source build as the last fallback means a missing asset degrades to today's timing.

**Payoff:** permanently removes ~3.5 GiB from the cap, which is what keeps the Nuitka caches
resident run to run; and covers legs the Actions cache structurally cannot warm (cu121, and any
new variant, without needing a matching warm cell). It also dissolves the intra-run duplicate
build problem: ubuntu cu124 x3, ubuntu cu125 x2, windows cu124 x2, windows cu125 x2, windows
vulkan x3, macos metal x2 all build the same bytes concurrently because `actions/cache` gives no
cross-job mutual exclusion.

**Effort:** new workflow, `contents: write`, a prune step for old `engine-<hash>` releases, and a
decision on whether those releases are visible in the release list.

**Do not** solve the duplicate-build problem by adding a serial `build-engines` job that downstream
legs consume via `download-artifact`. That puts the 80.7 min engine build in front of the 223.7 min
Nuitka step and makes the critical path worse: 81 + 224 = 305 vs today's 293.6. Same for adding
`needs: build-multigpu` to `build-cuda`.

### 2.3 Persist the onefile-compression cache

`OnefileCompressor.py:33-34` hardcodes zstd level 22. Measured cost: 25.3 min on Windows,
36.5 min on Linux. But Nuitka caches each compressed file individually
(`OnefileCompressor.py:170-181`), keyed on file **content** plus Python/Nuitka/zstd version and
level, under `getCacheDir("onefile-compression")`, which routes through `NUITKA_CACHE_DIR`.
`NUITKA_CACHE_DIR` is set nowhere on Linux and nowhere in `build-cuda-executables.yml`.

The cache is content-addressed, so it is portable across matrix legs whose 1.25 GB payloads are
identical apart from ~15 engine DLLs. A warm cache reduces the phase to hashing plus file copies.
Estimated 25.3 to ~5 min, **unverified**.

```yaml
      - name: Point Nuitka at the cache dir
        shell: bash
        run: echo "NUITKA_CACHE_DIR=${GITHUB_WORKSPACE}/nuitka-cache" >> "$GITHUB_ENV"
```

plus `nuitka-cache` in the `actions/cache` restore/save pair on every OS (`release.yml` does this
for Windows only today, lines 225-236 and 296-301). Keep the existing ordering: `release.yml:324`
`rm -rf ... nuitka-cache` runs after the save.

This is the single largest new cache consumer (~666 MB Windows, ~1.25 GB Linux). It cannot land
before 1.1 and 2.2. Measure the cache upload/download step on the first warm run: if it exceeds
~20 min the saving evaporates.

Do **not** reach for `--onefile-no-compression`. The per-file zstd is doing real work: the shipped
`lilbee-linux-x86_64-cu125` is 1,248,930,008 bytes and matches the compressed payload. Disabling
it roughly doubles a 1.25 GB user download.

### 2.4 Minor structural cleanups

- `smoke-wheels` declares `needs: build-multigpu` but consumes only the three `cpu` cells. In
  30003020138 it waited ~83 min for CUDA cells whose artifacts it never reads. Currently worth
  0 min (it has ~3.5 h of slack behind `attach-prerelease`), so defer until Tier 1 lands.
- An opt-in `backends:` subset input on `build-cuda-executables.yml`, defaulting to empty (all),
  would let a fast RC dispatch build one CUDA leg. Zero default behaviour change. Gate every step
  after the subset check, not just the build, so a skipped leg attaches nothing.
- Gate the CUDA toolkit install on a `lookup-only: true` engine cache probe, as
  `build-multigpu.yml:100-110` already does. 4.9 min Windows / 3.3 min Linux, but only once 1.3
  makes the engine cache actually hit. The key expression gets duplicated between the probe and
  the composite action; drift means the toolkit is skipped when needed, which fails loudly at
  `CUDA_PATH must be set` rather than shipping a bad artifact.

---

## 5. Tier 3: costs money

**Larger GitHub-hosted runners.** Not addressable today: `gh api repos/:owner/:repo --jq
'.owner.type'` returns `User`, and `gh api /orgs/:owner/actions/runner-groups` returns 404.
Larger runners are configured through org or enterprise runner groups. Moving the repo under an
organization on a Team plan ($4/user/month) unlocks `windows-latest-8-core` and `-16-core`.

The trade, since the C phase is embarrassingly parallel and currently pinned at `-j4`
(`getJobLimit()` falls back to `getCPUCoreCount()`, and both runners report 4):

| runner | C phase (from 176.4 min at 4 cores) | billing multiplier | approx cost per Windows leg |
|---|---|---|---|
| windows 4-core (today) | 176 min | 2x, free on public repo | $0 |
| windows 8-core | ~90 min, unverified | 8x-16x on private; larger runners are **not free on public repos** | ~$5-8 |
| windows 16-core | ~50 min, unverified | ~2x the 8-core rate | ~$8-13 |

Note the asterisk: GitHub-hosted larger runners are billed even on public repos, unlike standard
runners. Four Windows legs per release at 16-core is roughly $35-50 per release. Scaling is
assumed near-linear for compilation and is **unverified** for this workload.

**Self-hosted Windows runner.** A single persistent box gives a genuinely warm clcache and
`ccache` on local disk, no 10 GiB cap, no cold `Bundle llama-server`, and as many cores as the
hardware has. This is the only option that makes the caching story reliable rather than
LRU-dependent. Cost is a machine plus maintenance plus the security posture of a self-hosted
runner on a public repo (never run untrusted PR workflows on it; restrict to `release-*`
workflows and tag pushes).

**Not worth doing:** upstream llama.cpp prebuilt binaries. Verified against the live asset list on
tag b10107: upstream ships exactly two CUDA binaries, both Windows (12.4 and 13.3). There is no
Linux CUDA asset at all, and no Windows cu125. Every variant upstream does publish maps onto a leg
that is already under 10 min or already cache-warm. Two further blockers: the build clones
`abetlen/llama-cpp-python` at a pinned tag and builds that release's vendored llama.cpp commit,
which has no stable mapping to an upstream `bNNNN`; and the build deliberately diverges from stock
(`BUILD_SHARED_LIBS=ON`, baked `$ORIGIN`/`@loader_path` rpath, CURL/SSL off).

---

## 6. What is already fine

Stop worrying about these.

- **`--jobs` is not missing.** Nuitka's `getJobLimit()` defaults to `getCPUCoreCount()`, and both
  runner types log `build_jobs=4` after `nproc`. It already runs `-j4`. Do **not** set
  `--low-memory`: it forces `jobs=1` and would triple the C phase.
- **LTO is already off.** No `--lto` in any logged Nuitka command line, and the actual gcc line is
  `-O3` with no `-flto`. There is no link-time win sitting on the table.
- **Onefile compression is not waste.** The `100.00%` ratio in the log is a reporting artifact of
  `--onefile-as-archive`, which makes the outer compressor a passthrough and compresses each file
  individually. Real compression is happening; the shipped asset size proves it.
- **`actions/cache` key expressions are correct.** The engine keys hash `engine-versions.env` plus
  the three build scripts. The 07-22 warm save and the 07-23 release miss used byte-identical keys.
  This was never a key bug, it was eviction and scope.
- **Ref scoping works.** The tag run successfully restored two main-scoped caches (macos metal,
  manylinux vulkan). Main-scope to tag-scope restore is fine. The problem is that the large entries
  had been evicted, not that they were unreachable.
- **The Vulkan/Metal warm path works.** `build-binaries (windows-latest, vulkan)` restores its
  engine cache and pays 12 s for `Bundle llama-server` instead of 62 min. That is the mechanism the
  CUDA legs need, and it is proof the design is right.
- **`continue-on-error` + `timeout-minutes: 360`** on the build cells means a warm-run failure
  cannot break main, which is what makes the Tier 1 warming changes safe to land.
- **The `restore-ci-models` pattern** (pull from a `ci-models` GitHub release rather than the
  Actions cache) is the right shape and is exactly the model 2.2 generalises.
- **Artifact upload is not a bottleneck.** 21 s for 635 MiB.

---

## 7. Realistic end state

Baseline: **~305 min** clean run, terminus `build-cuda (windows-2022, cu125)` at 293.6 min, with
`build-binaries (windows-latest, compat, vulkan)` 11 min behind it at 240.1 min.

**Tier 1 applied (1.1 through 1.6): ~275 min.**

Working: build-cuda Windows legs lose ~59 min of `Bundle llama-server` and ~14 min of litellm,
landing around 15:15 and 15:00. The Linux tail (cu121) goes 211 to ~93 min. But
`build-binaries (windows-latest, compat, vulkan)` only loses the ~14 min of litellm and becomes
the terminus at ~16:01. Run-level saving is roughly **30 min, not hours**.

That is the honest answer and it is worth stating why it is disappointing: Tier 1 is entirely
cache plumbing, and there is no cache in this pipeline that touches the 176 minutes of MSVC
compilation that actually gates the release. Tier 1's real product is that the caching becomes
*reliable* rather than evicted, which is a precondition for everything else, plus ~30 min.

**Upside case, unverified: ~160-180 min.** If item 6 (Windows Nuitka cache in `build-cuda`, plus
whatever makes the existing `release.yml` mechanism actually restore) produces real clcache hits,
each Windows `Build executable` goes from ~230 min to somewhere around 90-110 min and the run
lands near 160-180 min. Flag: **no warm Windows Nuitka compile has ever been observed in this
repo's history.** Every sampled run shows `0 cache hits`. The 60-75% hit rate this depends on is
invented, not measured. Do not plan around it until one warm run proves it.

**Tier 1 + Tier 2: ~120-140 min**, if 2.1 (compile the Windows closure once) is feasible. The
terminus becomes a single Windows compile at ~210 min cold or ~95 min warm, plus a ~2 min engine
restore, plus ~20 min of packaging, plus the ~15 min of pre-build gates and the ~3 min attach.

**The floor.** Given Nuitka on a 4-core GitHub Windows runner, the shortest this pipeline can be:

- With everything in Tier 1+2 working and clcache warm: **~90-110 min**. That is one Windows
  Nuitka compile at ~60-90 min (unverified), plus ~27 min of link and onefile compression that no
  cache removes on a first-of-its-kind payload, plus fixed overhead.
- With everything in Tier 1+2 working and clcache cold (a dependency bump, a lilbee source change
  touching a widely-imported module, or a Nuitka version bump): **~230-250 min**. Cold is the
  common case for a release that actually changed code.
- With a 16-core runner or a self-hosted Windows box: **~60-75 min**, unverified.

So: no, this does not get under an hour on free runners, and it will not reliably get under two
hours either, because the cold case is the normal case for a release. Realistically you are
choosing between "~5 hours, unreliable caching" and "~2 to 2.5 hours typical, ~1.5 to 2 hours when
the cache is warm". Getting genuinely fast requires either more cores (Tier 3) or removing three
of the four Windows compiles (2.1).

---

## 8. Uncertainties

Things that need a measurement run before they can be trusted.

1. **Windows clcache hit rate: completely unmeasured.** Zero observed hits across 07-14, 07-16,
   07-18, 07-19, 07-20, 07-22 and 07-23. Every projection involving item 6 rests on this. First
   measurement: after 1.1 frees cap headroom, run `warm-executables-cache` twice in a row and grep
   the second run for `Compiled NNNN C files using clcache with N cache hits`. If N is still 0, the
   mechanism is broken for a reason nobody has found and Tier 2's 2.1 becomes the only path.
2. **Linux ccache warm hit rate after the `CCACHE_DIR` fix.** The 70 min assumes ~90% direct hits
   against a main-branch cache built at a different commit and version string. Verify by grepping
   for `result 'cache hit'` in the first post-fix warm run, and check `ccache -s` reports nonzero
   totals.
3. **Nuitka cache sizes vs the cap.** 9031 objects extrapolated to ~2.5-3 GB compressed per leg
   from cu121's 411 ggml objects at 0.18 GB. If the real number is higher, `max-size: 2G` silently
   degrades the hit rate run over run and the whole caching plan needs 2.2 first.
4. **Onefile compression cache upload/download time.** ~666 MB Windows, ~1.25 GB Linux. If the
   `actions/cache` step exceeds ~20 min the 20 min saving is net zero.
5. **`litellm.proxy` runtime reachability.** Static analysis says nothing imports it; litellm's
   dynamic provider dispatch means static analysis is not conclusive. Gate on the full
   `artifact_smoke.sh` ask path.
6. **Which cell tails.** Not stable across runs: 30003020138 attempt 1 tails on Windows cu125,
   29960035783 also on Windows cu125, attempt 2 of 30003020138 on Linux cu121. All Tier 1
   projections assume a Windows tail; a Linux tail makes items 3 and 4 worth substantially more.
7. **Larger-runner scaling.** ~50 min at 16 cores is a linear extrapolation from 176 min at 4.
   Nuitka's Scons phase has a documented serial tail (the "Slow C compilation detected,
   scalability problem" warning fires today), so real scaling will be sublinear. Unknown by how
   much.
8. **2.1 feasibility.** Whether Nuitka can package a prebuilt standalone dist into a onefile
   without recompiling is unknown and is the single highest-value open question in this report.

### Incidental defects found while measuring

- `action.yml:34` comment claims tag-scoped saves are "still restorable across a moved tag". False.
- `build-cuda-executables.yml` matrix comment says cu121 pins llama-cpp-python to 0.3.20; the cell
  sets 0.3.30.
- Windows engine cache keys hash to `8ba7a308...` while Linux/macOS hash to `e066ee62...` for the
  same four files: `hashFiles` seeing CRLF from the Windows checkout. Consistent, so it causes no
  misses today, but any future line-ending normalization silently invalidates every Windows engine
  cache at once.
- `engine-ubuntu-22.04-cu121` (753.8 MB, the largest engine cache in the repo) is pure ballast: its
  key can never be produced by anything on main, so it is written on every release and read never.
