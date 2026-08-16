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

const USER_AGENT = "lilbee-mcp";

async function fetchJson(url) {
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
        log(`lilbee-mcp: downloading… ${pct}% (${Math.floor(done / 1048576)}MB)`);
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
    `lilbee-mcp: first run — downloading ${assetName} (${Math.ceil(size / 1048576)}MB) ` +
      `from ${repo}@${release}. Cached for next time; run "lilbee-mcp prepare" to do this ahead of time.`
  );

  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const tmp = `${dest}.download`;

  let attempt = 0;
  for (;;) {
    attempt += 1;
    try {
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
      if (attempt >= 2) throw err;
      log(`lilbee-mcp: download failed (${err.message}), retrying once…`);
    }
  }

  fs.chmodSync(tmp, 0o755);
  fs.renameSync(tmp, dest);
  log(`lilbee-mcp: installed ${dest}`);
}
