/**
 * Local-mode binary resolution, in order:
 *   1. LILBEE_BIN (explicit override; must exist)
 *   2. `lilbee` on PATH
 *   3. the shared-root binary other lilbee installers manage (read-only here)
 *   4. previously bootstrapped binary in the launcher's versioned cache
 *   5. download the release asset into the cache (see download.mjs)
 *
 * Dependencies are injected so tests never touch the real fs/network.
 */

import path from "node:path";
import os from "node:os";

const CACHE_DIR_NAME = "lilbee-npm";
const SHARED_BIN_DIR = "bin";

/** lilbee's platform data root (mirrors the server's resolution defaults). */
export function dataRoot(env, platform = process.platform) {
  if (env.LILBEE_DATA) return env.LILBEE_DATA;
  const home = os.homedir();
  if (platform === "darwin") return path.join(home, "Library", "Application Support", "lilbee");
  if (platform === "win32") {
    return path.join(env.LOCALAPPDATA || path.join(home, "AppData", "Local"), "lilbee");
  }
  return path.join(env.XDG_DATA_HOME || path.join(home, ".local", "share"), "lilbee");
}

/** The unversioned binary that lilbee's other installers manage. The launcher
 * only ever reads it; writes stay in the launcher's own versioned cache so the
 * two managers cannot fight over one file. */
export function sharedRootBinary(env, platform = process.platform) {
  const name = platform === "win32" ? "lilbee.exe" : "lilbee";
  return path.join(dataRoot(env, platform), SHARED_BIN_DIR, name);
}

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
 * "env" | "path" | "cache" | "download". Throws when LILBEE_BIN is set but
 * missing (an explicit override silently falling through would be worse).
 *
 * @param {object} opts
 * @param {object} opts.env
 * @param {string} opts.release   release tag to bootstrap when needed
 * @param {string} opts.assetName release asset for this host
 * @param {object} opts.deps      { existsSync, whichSync, download }
 */
function safeRealpath(realpathSync, p) {
  try {
    return realpathSync(p);
  } catch {
    return p;
  }
}

export async function resolveBinary({ env, release, assetName, deps }) {
  if (env.LILBEE_BIN) {
    if (!deps.existsSync(env.LILBEE_BIN)) {
      throw new Error(`LILBEE_BIN points at "${env.LILBEE_BIN}" but nothing is there.`);
    }
    return { path: env.LILBEE_BIN, source: "env" };
  }

  // A PATH hit must be a real installed lilbee, not an npm bin shim. Under
  // npx (and global npm installs) node_modules/.bin leads the PATH and the
  // entry there is THIS launcher — accepting it made `prepare` a silent
  // no-op and every passthrough command spawn itself without bound. Scan
  // every PATH candidate so a real install behind the shim still wins.
  for (const onPath of deps.whichAllSync("lilbee")) {
    const real = deps.realpathSync ? safeRealpath(deps.realpathSync, onPath) : onPath;
    const isNpmShim = /[\\/]node_modules[\\/]/.test(real);
    const isSelf = deps.selfPath ? real === deps.selfPath : false;
    if (!isNpmShim && !isSelf) return { path: onPath, source: "path" };
  }

  const shared = sharedRootBinary(env);
  if (deps.existsSync(shared)) return { path: shared, source: "shared-root" };

  const cached = cachedBinaryPath(env, release, assetName);
  if (deps.existsSync(cached)) return { path: cached, source: "cache" };

  await deps.download({ env, release, assetName, dest: cached });
  return { path: cached, source: "download" };
}
