/**
 * Local-mode binary resolution, in order:
 *   1. LILBEE_BIN (explicit override; must exist)
 *   2. previously bootstrapped binary in the launcher's versioned cache
 *   3. download the release asset into the cache (see download.mjs)
 *
 * The launcher deliberately ignores lilbee installs from other package
 * managers (PATH, the shared data root): the npm package always runs the
 * exact release it pins, so `npm install` means a fresh, known binary.
 * LILBEE_BIN is the one escape hatch.
 *
 * Dependencies are injected so tests never touch the real fs/network.
 */

import path from "node:path";
import os from "node:os";

const CACHE_DIR_NAME = "lilbee-npm";

export function cacheDir(env, platform = process.platform) {
  if (env.LILBEE_MCP_CACHE) return env.LILBEE_MCP_CACHE;
  const home = os.homedir();
  if (platform === "darwin") return path.join(home, "Library", "Caches", CACHE_DIR_NAME);
  if (platform === "win32") {
    return path.join(env.LOCALAPPDATA || path.join(home, "AppData", "Local"), CACHE_DIR_NAME);
  }
  return path.join(env.XDG_CACHE_HOME || path.join(home, ".cache"), CACHE_DIR_NAME);
}

export function cachedBinaryPath(env, release, assetName, platform = process.platform) {
  return path.join(cacheDir(env, platform), release, assetName);
}

/**
 * Resolve the binary to run. Returns { path, source } where source is one of
 * "env" | "cache" | "download". Throws when LILBEE_BIN is set but missing
 * (an explicit override silently falling through would be worse).
 *
 * @param {object} opts
 * @param {object} opts.env
 * @param {string} opts.release   release tag to bootstrap when needed
 * @param {string} opts.assetName release asset for this host
 * @param {object} opts.deps      { existsSync, download }
 */
export async function resolveBinary({ env, release, assetName, deps }) {
  if (env.LILBEE_BIN) {
    if (!deps.existsSync(env.LILBEE_BIN)) {
      throw new Error(`LILBEE_BIN points at "${env.LILBEE_BIN}" but nothing is there.`);
    }
    return { path: env.LILBEE_BIN, source: "env" };
  }

  const cached = cachedBinaryPath(env, release, assetName);
  if (deps.existsSync(cached)) return { path: cached, source: "cache" };

  await deps.download({ env, release, assetName, dest: cached });
  return { path: cached, source: "download" };
}
