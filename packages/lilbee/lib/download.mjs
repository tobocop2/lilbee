/**
 * Download a standalone lilbee release asset with sha256 verification.
 *
 * The GitHub API reports a `digest` ("sha256:<hex>") per asset; the download
 * streams to a temp file, the hash is checked against that digest, and the
 * file lands at its final path via atomic rename. All progress goes to
 * stderr — stdout belongs to the MCP wire.
 */

import fs from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable, Transform } from "node:stream";
import { createHash } from "node:crypto";

/**
 * Corporate proxies: Node's built-in fetch ignores HTTP(S)_PROXY env vars
 * (unlike curl and npm itself). Route through undici's env-aware agent when
 * a proxy is configured, so the bootstrap works behind the same proxy npm
 * installed the package through.
 */
async function configureProxyFromEnv() {
  const proxy =
    process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
  if (!proxy) return;
  try {
    const { EnvHttpProxyAgent, setGlobalDispatcher } = await import("undici");
    setGlobalDispatcher(new EnvHttpProxyAgent());
  } catch {
    console.error("lilbee: HTTPS_PROXY is set but the proxy agent is unavailable; downloading directly.");
  }
}
const proxyReady = configureProxyFromEnv();

const USER_AGENT = "lilbee-npm-launcher";
const DOWNLOAD_ATTEMPTS = 2;

async function fetchJson(url) {
  await proxyReady;
  const res = await fetch(url, {
    headers: { "user-agent": USER_AGENT, accept: "application/vnd.github+json" },
  });
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
  return res.json();
}

/** The tag of the newest published release, e.g. "v0.6.90b425". */
export async function latestReleaseTag(repo) {
  const info = await fetchJson(`https://api.github.com/repos/${repo}/releases/latest`);
  if (!info.tag_name) throw new Error(`releases/latest for ${repo} has no tag_name`);
  return info.tag_name;
}

/** Look up the asset's browser_download_url and sha256 digest for a release tag. */
export async function releaseAsset(repo, release, assetName) {
  const info = await fetchJson(`https://api.github.com/repos/${repo}/releases/tags/${release}`);
  const asset = (info.assets ?? []).find((a) => a.name === assetName);
  if (!asset) {
    const names = (info.assets ?? []).map((a) => a.name).join(", ");
    throw new Error(`Release ${release} has no asset "${assetName}". Available: ${names}`);
  }
  const digest =
    typeof asset.digest === "string" && asset.digest.startsWith("sha256:")
      ? asset.digest.slice("sha256:".length)
      : null;
  return { url: asset.browser_download_url, size: asset.size, digest };
}

const MB = 1048576;
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

/** Stream stage that redraws one bar line in place and ends it with a newline. */
function ttyProgress(total, stream) {
  const started = Date.now();
  let done = 0;
  let lastDraw = 0;
  const draw = () => {
    const elapsed = (Date.now() - started) / 1000;
    stream.write(CLEAR_LINE + progressLine({ done, total, bytesPerSec: elapsed ? done / elapsed : 0, color: true }));
  };
  return new Transform({
    transform(chunk, _enc, cb) {
      done += chunk.length;
      if (Date.now() - lastDraw >= DRAW_INTERVAL_MS) {
        lastDraw = Date.now();
        draw();
      }
      cb(null, chunk);
    },
    flush(cb) {
      draw();
      stream.write("\n");
      cb();
    },
  });
}

/** Stream stage that logs one line per LOG_STEP_PCT of progress. */
function lineProgress(total, log) {
  let done = 0;
  let lastPct = -LOG_STEP_PCT;
  return new Transform({
    transform(chunk, _enc, cb) {
      done += chunk.length;
      const pct = total ? Math.floor((done / total) * 100) : 0;
      if (pct >= lastPct + LOG_STEP_PCT) {
        lastPct = pct;
        log(`lilbee: downloading… ${pct}% (${formatMB(done)})`);
      }
      cb(null, chunk);
    },
  });
}

/**
 * Progress for the download pipeline: an in-place bar on a TTY, log lines elsewhere.
 * Local, not a library: keeps the zero-runtime-dependency publish.
 */
export function progressReporter(total, log, stream = process.stderr) {
  return stream.isTTY ? ttyProgress(total, stream) : lineProgress(total, log);
}

/**
 * Download the asset for `release` to `dest`. One retry on a failed attempt —
 * these are multi-hundred-MB files and transient resets happen.
 */
export async function download({ env, release, assetName, dest, log = console.error }) {
  const repo = env.LILBEE_REPO || "tobocop2/lilbee";
  const { url, size, digest } = await releaseAsset(repo, release, assetName);
  log(
    `lilbee: downloading ${assetName} (${formatMB(size)}) ` +
      `from ${repo}@${release}. One-time per release; cached after that.`
  );

  if (!digest) {
    log(`lilbee: release asset ${assetName} has no published digest; skipping sha256 verification.`);
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  // Per-process temp file: concurrent first runs (e.g. two agents spawning at
  // once) must not interleave writes into one temp path.
  const tmp = `${dest}.download.${process.pid}`;

  let attempt = 0;
  for (;;) {
    attempt += 1;
    try {
      await proxyReady;
      const res = await fetch(url, { headers: { "user-agent": USER_AGENT } });
      if (!res.ok || !res.body) throw new Error(`GET ${url} -> HTTP ${res.status}`);
      const hash = createHash("sha256");
      const hasher = new Transform({
        transform(chunk, _enc, cb) {
          hash.update(chunk);
          cb(null, chunk);
        },
      });
      await pipeline(
        Readable.fromWeb(res.body),
        hasher,
        progressReporter(size, log),
        fs.createWriteStream(tmp)
      );
      const got = hash.digest("hex");
      if (digest && got !== digest) {
        throw new Error(`sha256 mismatch for ${assetName}: expected ${digest}, got ${got}`);
      }
      break;
    } catch (err) {
      fs.rmSync(tmp, { force: true });
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
  log(`lilbee: installed ${dest}`);
}
