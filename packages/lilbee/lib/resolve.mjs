/**
 * Binary resolution for the CLI: LILBEE_BIN, else the release the package pins
 * (or the one LILBEE_RELEASE / an explicit tag names), from the cache when that
 * exact release is cached and downloaded otherwise.
 */

import fs from "node:fs";

import { cachedBinaryPath, releaseBinary } from "./cache.mjs";
import { releaseByTag } from "./releases.mjs";

/**
 * Plan the binary to run: `{ path, source, release, download }` where `source`
 * is "env" | "cache" | "download" and `download` is the ResolvedRelease to
 * install when `source` is "download". `refresh` skips the cache lookup and
 * resolves the release again. Throws when LILBEE_BIN is set but missing.
 */
export async function resolveBinary({ env, cacheDir, host, pinned, tag = null, refresh = false, repo, fetch }) {
  if (env.LILBEE_BIN) {
    if (!fs.existsSync(env.LILBEE_BIN)) {
      throw new Error(`LILBEE_BIN points at "${env.LILBEE_BIN}" but nothing is there.`);
    }
    return { path: env.LILBEE_BIN, source: "env", release: null, download: null };
  }

  const wanted = tag || env.LILBEE_RELEASE || pinned;
  const cached = refresh ? null : releaseBinary({ cacheDir, release: wanted, host });
  if (cached) return { path: cached.path, source: "cache", release: cached.release, download: null };

  const resolved = await releaseByTag(wanted, { repo, env, host, fetch });
  const dest = cachedBinaryPath(cacheDir, resolved.tag, resolved.assetName);
  if (fs.existsSync(dest)) return { path: dest, source: "cache", release: resolved.tag, download: null };
  return { path: dest, source: "download", release: resolved.tag, download: resolved };
}
