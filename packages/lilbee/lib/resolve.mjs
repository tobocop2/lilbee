/**
 * Binary resolution for the CLI: LILBEE_BIN, else an exact release tag, else
 * the newest cached binary, else the latest release on the channel with the
 * package's pinned release as the fallback when that lookup fails.
 */

import fs from "node:fs";

import { cachedBinaryPath, installedBinary } from "./cache.mjs";
import { latestRelease, releaseByTag } from "./releases.mjs";

/**
 * Plan the binary to run: `{ path, source, release, download }` where `source`
 * is "env" | "cache" | "download" and `download` is the ResolvedRelease to
 * install when `source` is "download". Throws when LILBEE_BIN is set but missing.
 */
export async function resolveBinary({ env, cacheDir, host, pinned, tag = null, refresh = false, includeDev = false, repo, fetch, log = () => {} }) {
  if (env.LILBEE_BIN) {
    if (!fs.existsSync(env.LILBEE_BIN)) {
      throw new Error(`LILBEE_BIN points at "${env.LILBEE_BIN}" but nothing is there.`);
    }
    return { path: env.LILBEE_BIN, source: "env", release: null, download: null };
  }

  const query = { repo, env, host, includeDev, fetch };
  const wanted = tag || env.LILBEE_RELEASE;
  let resolved;
  if (wanted) {
    resolved = await releaseByTag(wanted, query);
  } else {
    const installed = refresh ? null : installedBinary({ cacheDir, host });
    if (installed) return { path: installed.path, source: "cache", release: installed.release, download: null };
    try {
      resolved = await latestRelease(query);
    } catch (err) {
      log(`lilbee: could not look up the latest release (${err.message}); using the tested ${pinned}.`);
      resolved = await releaseByTag(pinned, query);
    }
  }

  const dest = cachedBinaryPath(cacheDir, resolved.tag, resolved.assetName);
  if (fs.existsSync(dest)) return { path: dest, source: "cache", release: resolved.tag, download: null };
  return { path: dest, source: "download", release: resolved.tag, download: resolved };
}
