/**
 * Download a release asset with sha256 verification: stream to a temp file,
 * check the hash against the release digest, land by atomic rename.
 */

import fs from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable, Transform } from "node:stream";
import { createHash } from "node:crypto";

import { LauncherError } from "./errors.mjs";
import { launcherFetch } from "./fetch.mjs";
import { DEFAULT_REPO, USER_AGENT } from "./releases.mjs";

const DOWNLOAD_ATTEMPTS = 2;
const DOWNLOAD_IDLE_TIMEOUT_MS = 60_000;
const DISK_SPACE_FACTOR = 1.1;
const DOWNLOAD_STALLED = "The lilbee download stalled: no data arrived for 60 seconds. Check your connection and try again.";
const NO_DIGEST = (assetName) => `Release asset ${assetName} has no published digest, so the download cannot be verified.`;
const DOWNLOAD_CANCELED = "The lilbee download was cancelled.";

const MB = 1048576;
const GB = 1073741824;
const BAR_FULL = "━";
const BAR_HALF = "╸";
const BAR_REST = "─";
const BAR_WIDTH = 30;
const ANSI_ROSE = "\x1b[38;5;211m";
const ANSI_DIM = "\x1b[2m";
const ANSI_RESET = "\x1b[0m";
const CLEAR_LINE = "\r\x1b[2K";
const DRAW_INTERVAL_MS = 100;
const LOG_STEP_PCT = 10;

const mbCount = (bytes) => Math.floor(bytes / MB);
const formatMB = (bytes) => `${mbCount(bytes)}MB`;
const formatSize = (bytes) => (bytes >= GB ? `${(bytes / GB).toFixed(1)}GB` : formatMB(bytes));

/** Thrown when the caller's signal aborts a download. */
export class DownloadCanceledError extends Error {
  constructor() {
    super(DOWNLOAD_CANCELED);
    this.name = "AbortError";
  }
}

/** True for a DownloadCanceledError, or any error whose name is "AbortError". */
export function isDownloadCanceled(err) {
  return err instanceof DownloadCanceledError || (typeof err === "object" && err !== null && err.name === "AbortError");
}

/** "m:ss" for a duration in seconds. */
function formatEta(seconds) {
  const s = Math.round(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** The bar glyphs for a 0..1 fraction at `width` cells; `color` tints the filled part. */
function barGlyphs(frac, width, color) {
  const cells = frac * width;
  const full = Math.floor(cells);
  const half = full < width && cells - full >= 0.5 ? BAR_HALF : "";
  const filled = BAR_FULL.repeat(full) + half;
  const rest = BAR_REST.repeat(width - full - half.length);
  return color ? `${ANSI_ROSE}${filled}${ANSI_RESET}${ANSI_DIM}${rest}${ANSI_RESET}` : filled + rest;
}

/**
 * One rich-style progress line: `lilbee │━━━━━╸────│  55% 523/1246MB 87.0MB/s eta 0:08`.
 * Pure; rate and eta are omitted when unknown.
 */
export function progressLine({ done, total, bytesPerSec, barWidth = BAR_WIDTH, color = false }) {
  if (!total) return `lilbee: downloading… ${formatMB(done)}`;
  const frac = Math.min(done / total, 1);
  const pct = `${Math.floor(frac * 100)}%`.padStart(5);
  const line = `lilbee │${barGlyphs(frac, barWidth, color)}│${pct} ${mbCount(done)}/${formatMB(total)}`;
  if (!(bytesPerSec > 0) || done >= total) return line;
  return `${line} ${(bytesPerSec / MB).toFixed(1)}MB/s eta ${formatEta((total - done) / bytesPerSec)}`;
}

/** A progress consumer that redraws one bar line in place and ends it with a newline. */
function ttyProgress(stream) {
  const started = Date.now();
  let lastDraw = 0;
  let open = false;
  const draw = ({ done, total }) => {
    const elapsed = (Date.now() - started) / 1000;
    stream.write(CLEAR_LINE + progressLine({ done, total, bytesPerSec: elapsed ? done / elapsed : 0, color: true }));
    open = true;
  };
  const finish = () => {
    if (!open) return;
    stream.write("\n");
    open = false;
  };
  const report = (progress) => {
    const complete = progress.total !== null && progress.done >= progress.total;
    if (complete || Date.now() - lastDraw >= DRAW_INTERVAL_MS) {
      lastDraw = Date.now();
      draw(progress);
    }
    if (complete) finish();
  };
  report.finish = finish;
  return report;
}

/** A progress consumer that logs one line per LOG_STEP_PCT of progress. */
function lineProgress(log) {
  let lastPct = -LOG_STEP_PCT;
  const report = ({ done, total }) => {
    const pct = total ? Math.floor((done / total) * 100) : 0;
    if (pct < lastPct) lastPct = -LOG_STEP_PCT;
    if (pct >= lastPct + LOG_STEP_PCT) {
      lastPct = pct;
      log(`lilbee: downloading… ${pct}% (${formatMB(done)})`);
    }
  };
  report.finish = () => {};
  return report;
}

/** An onProgress consumer for the CLI: an in-place bar on a TTY, log lines elsewhere. Call `finish()` when the download ends. */
export function progressReporter(log, stream = process.stderr) {
  return stream.isTTY ? ttyProgress(stream) : lineProgress(log);
}

/** Refuse a download the filesystem holding `dir` cannot fit; silent when free space is unknown. */
async function assertFreeSpace(dir, size) {
  if (typeof fs.promises.statfs !== "function") return;
  let stats;
  try {
    stats = await fs.promises.statfs(dir);
  } catch {
    return;
  }
  const free = Number(stats.bavail) * Number(stats.bsize);
  const needed = Math.ceil(size * DISK_SPACE_FACTOR);
  if (free < needed) {
    throw new LauncherError(
      "no-space",
      `Not enough disk space for the lilbee binary: it needs about ${formatSize(needed)} free in ${dir}, ` +
        `and ${formatSize(free)} is available.`,
      { neededBytes: needed, freeBytes: free }
    );
  }
}

/** The total a transfer reports: Content-Length, else the release's asset size, else null. */
function totalBytes(res, release) {
  const header = Number(res.headers.get("content-length"));
  if (Number.isFinite(header) && header > 0) return header;
  return release.size > 0 ? release.size : null;
}

/** A Node stream over a response body that is either a web ReadableStream or a Node Readable. */
function toReadable(body) {
  return typeof body.pipe === "function" ? body : Readable.fromWeb(body);
}

/** Remove every entry of the release dir but the landed binary. */
function removeSiblings(dest) {
  const dir = path.dirname(dest);
  const keep = path.basename(dest);
  for (const entry of fs.readdirSync(dir)) {
    if (entry === keep) continue;
    try {
      fs.rmSync(path.join(dir, entry), { recursive: true, force: true });
    } catch {
      // a busy file on Windows: a stale build is a nuisance, not an error
    }
  }
}

/**
 * One transfer of `release.url` to `tmp`. Resolves with the hex sha256 of what
 * landed; rejects with DownloadCanceledError on `signal`, or a stall error when
 * no bytes arrive for `idleTimeoutMs`.
 */
async function transfer({ release, tmp, fetchImpl, onProgress, signal, idleTimeoutMs }) {
  const controller = new AbortController();
  let source = null;
  let failure = null;
  let timer = null;
  const fail = (err) => {
    if (!failure) failure = err;
    clearTimeout(timer);
    controller.abort(err);
    source?.destroy(err);
  };
  const restartClock = () => {
    clearTimeout(timer);
    timer = setTimeout(() => fail(new LauncherError("stalled", DOWNLOAD_STALLED)), idleTimeoutMs);
  };
  const onAbort = () => fail(new DownloadCanceledError());
  signal?.addEventListener("abort", onAbort, { once: true });

  try {
    if (signal?.aborted) throw new DownloadCanceledError();
    restartClock();
    const res = await fetchImpl(release.url, { headers: { "user-agent": USER_AGENT }, signal: controller.signal });
    if (!res.ok || !res.body) throw new LauncherError("http", `GET ${release.url} -> HTTP ${res.status}`, { status: res.status });
    const total = totalBytes(res, release);
    const hash = createHash("sha256");
    let done = 0;
    const meter = new Transform({
      transform(chunk, _enc, cb) {
        restartClock();
        hash.update(chunk);
        done += chunk.length;
        onProgress?.({ done, total });
        cb(null, chunk);
      },
    });
    source = toReadable(res.body);
    await pipeline(source, meter, fs.createWriteStream(tmp));
    return hash.digest("hex");
  } catch (err) {
    throw failure ?? err;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

/**
 * Download `release` to `dest`: verified, landed by rename, with every other
 * file of the release dir removed afterwards. One retry on a failed transfer;
 * none after a cancel. A rival process that lands `dest` first is adopted.
 */
export async function download({
  release,
  dest,
  repo = DEFAULT_REPO,
  fetch,
  env = process.env,
  onProgress,
  signal,
  log = () => {},
  requireDigest = false,
  idleTimeoutMs = DOWNLOAD_IDLE_TIMEOUT_MS,
}) {
  if (signal?.aborted) throw new DownloadCanceledError();
  if (!release.digest && requireDigest) throw new LauncherError("no-digest", NO_DIGEST(release.assetName));
  const dir = path.dirname(dest);
  fs.mkdirSync(dir, { recursive: true });
  await assertFreeSpace(dir, release.size);
  const fetchImpl = fetch ?? (await launcherFetch(env, log));

  log(
    `lilbee: downloading ${release.assetName} (${formatMB(release.size)}) ` +
      `from ${repo}@${release.tag}. One-time per release; cached after that.`
  );
  if (!release.digest) {
    log(`lilbee: release asset ${release.assetName} has no published digest; skipping sha256 verification.`);
  }
  // Per-process temp file: concurrent first runs must not interleave writes into one path.
  const tmp = `${dest}.download.${process.pid}`;

  for (let attempt = 1; ; attempt += 1) {
    try {
      const got = await transfer({ release, tmp, fetchImpl, onProgress, signal, idleTimeoutMs });
      if (release.digest && got !== release.digest) {
        throw new LauncherError("digest-mismatch", `sha256 mismatch for ${release.assetName}: expected ${release.digest}, got ${got}`);
      }
      break;
    } catch (err) {
      fs.rmSync(tmp, { force: true });
      if (isDownloadCanceled(err)) throw err;
      if (fs.existsSync(dest)) {
        log("lilbee: another process finished this download first; using it.");
        return;
      }
      if (attempt >= DOWNLOAD_ATTEMPTS) throw err;
      log(`lilbee: download failed (${err.message}), retrying once…`);
    }
  }

  fs.chmodSync(tmp, 0o755);
  fs.renameSync(tmp, dest);
  removeSiblings(dest);
  log(`lilbee: installed ${dest}`);
}
