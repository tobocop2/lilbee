# The lilbee npm launcher, audited against popular binary shims

Reference points: esbuild, @biomejs/biome, turbo, @swc/core (platform packages);
playwright, cypress (managed downloads); sentry-cli, prisma engines
(postinstall downloads); node-pre-gyp (prebuild fetch).

## Delivery pattern

| Shim | Pattern | Why |
| --- | --- | --- |
| esbuild, biome, turbo, swc | Per-platform npm packages via `optionalDependencies` | Binaries are 10–40MB, fit the registry |
| playwright, cypress | Managed download into a shared cache | Assets are 100MB+ |
| sentry-cli, prisma | `postinstall` download | Medium binaries, install-time network |
| **lilbee** | **Lazy first-run bootstrap into a shared, versioned cache** | Binaries are 370–520MB — over the registry cap; lazy beats `postinstall` because `npm install` stays offline-safe and `npx -y lilbee` still works |

The platform-package pattern is not available to lilbee at these sizes.
Among download patterns, first-use is the least intrusive: no install-time
code execution (a `postinstall` script is also a supply-chain flag many
audits reject), no wasted download in CI paths that never run the binary.

## Selection

- Platform/arch mapping at run time, like every shim above.
- **Hardware detection on top** (new): NVIDIA driver's CUDA level →
  `cu121/cu124/cu125`, ROCm userland → `rocm`, missing AVX2 → `-compat`
  builds, composed (`compat-cu124`). No mainstream npm shim does this —
  they don't need to; lilbee's builds differ by GPU stack the way
  distro packages do, so the launcher behaves like `brew`/`flatpak`
  and picks for you. `LILBEE_VARIANT` remains the explicit override,
  and detection failing means the universal build, which runs anywhere.

## Integrity

- sha256 verified against the GitHub release asset digest before the
  binary is adopted; mismatch aborts. Parity with prisma's checksum
  verification; stronger than a bare `curl | sh`.
- Same-origin trust model (checksum and asset from the same release),
  which matches esbuild/playwright/prisma. Detached signatures would be
  the next step up; noted as future work, not a regression vs peers.

## Resilience

- Per-pid temp file + adopt-rival-download: two concurrent first runs
  cannot corrupt the cache (esbuild had this class of install race).
- One retry on failed transfer; multi-hundred-MB downloads reset sometimes.
- No HTTP Range resume (a full restart on retry). Playwright resumes;
  worth adding if first-run reports show flaky networks. Minor.

## Environment behavior

- `HTTPS_PROXY`/`HTTP_PROXY` honored (new): Node's fetch ignores proxy
  env vars, so the launcher now routes through undici's env proxy agent —
  sentry-cli and playwright both support this; before this change the
  bootstrap was broken behind corporate proxies.
- `LILBEE_BIN` (exact binary) and PATH take precedence over any download —
  the analogue of `ESBUILD_BINARY_PATH` / `PLAYWRIGHT_BROWSERS_PATH`.
- Version-pinned: the npm package's version maps to one release tag
  (`package.json » lilbee.release`), so installs are reproducible like
  esbuild's exact-version platform packages.

## Process behavior

- Signals forwarded; exit code is 128+signum on signal death, matching
  the conventions esbuild's shim follows.
- No `postinstall`, no install-time network, no scripts at all — the
  package is inert until executed.

## Gaps kept open (deliberately)

- No download resume (retry restarts). Revisit on evidence.
- No detached signature verification (peers don't either).
- Windows has no CPU-baseline probe; `-compat` there is explicit-only.
