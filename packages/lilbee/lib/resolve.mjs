/**
 * Local-mode binary resolution.
 *
 * The launcher tracks the LATEST lilbee release: a fresh install (and
 * `lilbee prepare`) asks GitHub for the newest release and downloads its
 * binary. Every later run execs the newest cached binary with no network.
 * The pinned release in package.json is the tested fallback when the
 * latest-release lookup fails (offline, rate limit, release mid-upload).
 *
 * Order:
 *   1. LILBEE_BIN (explicit override; must exist)
 *   2. LILBEE_RELEASE (explicit release tag: cache, then download)
 *   3. newest already-cached release (skipped when refresh is set)
 *   4. the latest GitHub release, falling back to the pin — cache, then
 *      download (old cached releases are pruned after a download lands)
 *
 * The launcher deliberately ignores lilbee installs from other package
 * managers (PATH, the shared data root): npm always runs its own verified
 * download, so `npm install` means a fresh, known binary.
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

/** Order release tags like v0.6.90b423 newest-first (numeric segments compared numerically). */
export function compareReleaseTags(a, b) {
  const nums = (t) => (String(t).match(/\d+/g) || []).map(Number);
  const na = nums(a);
  const nb = nums(b);
  for (let i = 0; i < Math.max(na.length, nb.length); i += 1) {
    const d = (nb[i] ?? -1) - (na[i] ?? -1);
    if (d) return d;
  }
  return 0;
}

/** The releases already cached for this asset, newest first. */
function cachedReleases(env, assetName, deps) {
  let entries;
  try {
    entries = deps.readdirSync(cacheDir(env));
  } catch {
    return [];
  }
  return entries.filter((r) => deps.existsSync(cachedBinaryPath(env, r, assetName))).sort(compareReleaseTags);
}

/** Delete cached release dirs other than `keep` — an upgrade replaces, not accumulates. */
function pruneOtherReleases(env, keep, deps) {
  if (!deps.rmSync) return;
  let entries;
  try {
    entries = deps.readdirSync(cacheDir(env));
  } catch {
    return;
  }
  for (const r of entries) {
    if (r === keep) continue;
    try {
      deps.rmSync(path.join(cacheDir(env), r), { recursive: true, force: true });
    } catch {
      // a busy file on Windows: stale cache is a nuisance, not an error
    }
  }
}

/**
 * Resolve the binary to run. Returns { path, source, release } where source
 * is "env" | "cache" | "download". Throws when LILBEE_BIN is set but missing
 * (an explicit override silently falling through would be worse).
 *
 * @param {object} opts
 * @param {object} opts.env
 * @param {string} opts.release   pinned fallback release tag
 * @param {string} opts.assetName release asset for this host
 * @param {boolean} [opts.refresh] re-resolve latest even when a binary is cached
 * @param {object} opts.deps      { existsSync, readdirSync, rmSync, latestTag, log, download }
 */
export async function resolveBinary({ env, release, assetName, refresh = false, deps }) {
  if (env.LILBEE_BIN) {
    if (!deps.existsSync(env.LILBEE_BIN)) {
      throw new Error(`LILBEE_BIN points at "${env.LILBEE_BIN}" but nothing is there.`);
    }
    return { path: env.LILBEE_BIN, source: "env", release: null };
  }

  const log = deps.log ?? (() => {});
  const cached = cachedReleases(env, assetName, deps);

  let tag;
  if (env.LILBEE_RELEASE) {
    tag = env.LILBEE_RELEASE;
  } else {
    // Runs with a usable binary stay off the network; prepare re-resolves.
    if (!refresh && cached.length) {
      return { path: cachedBinaryPath(env, cached[0], assetName), source: "cache", release: cached[0] };
    }
    try {
      tag = await deps.latestTag();
    } catch {
      tag = release;
      log(`lilbee: could not look up the latest release; using the tested ${release}.`);
    }
  }

  if (cached.includes(tag)) {
    return { path: cachedBinaryPath(env, tag, assetName), source: "cache", release: tag };
  }

  const dest = cachedBinaryPath(env, tag, assetName);
  await deps.download({ env, release: tag, assetName, dest });
  pruneOtherReleases(env, tag, deps);
  return { path: dest, source: "download", release: tag };
}
