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

function progressReporter(total, log) {
  let done = 0;
  let lastPct = -10;
  return new Transform({
    transform(chunk, _enc, cb) {
      done += chunk.length;
      const pct = total ? Math.floor((done / total) * 100) : 0;
      if (pct >= lastPct + 10) {
        lastPct = pct;
        log(`lilbee: downloading… ${pct}% (${Math.floor(done / 1048576)}MB)`);
      }
      cb(null, chunk);
    },
  });
}

/**
 * Download the asset for `release` to `dest`. One retry on a failed attempt —
 * these are multi-hundred-MB files and transient resets happen.
 */
export async function download({ env, release, assetName, dest, log = console.error }) {
  const repo = env.LILBEE_REPO || "tobocop2/lilbee";
  const { url, size, digest } = await releaseAsset(repo, release, assetName);
  log(
    `lilbee: first run — downloading ${assetName} (${Math.ceil(size / 1048576)}MB) ` +
      `from ${repo}@${release}. Cached for next time; run "lilbee prepare" to do this ahead of time.`
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
