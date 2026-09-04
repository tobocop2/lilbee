/**
 * The launcher's binary cache: `<cacheDir>/<release>/<assetName>`, one
 * release at a time, with lookup and install on top of the download.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { parseAssetName } from "./assets.mjs";
import { download } from "./download.mjs";
import { compareReleaseTags } from "./releases.mjs";

const CACHE_DIR_NAME = "lilbee-npm";

/** Where the launcher caches binaries: LILBEE_MCP_CACHE, else the platform cache dir. */
export function cacheDir(env = process.env, platform = process.platform) {
  if (env.LILBEE_MCP_CACHE) return env.LILBEE_MCP_CACHE;
  const home = os.homedir();
  if (platform === "darwin") return path.join(home, "Library", "Caches", CACHE_DIR_NAME);
  if (platform === "win32") {
    return path.join(env.LOCALAPPDATA || path.join(home, "AppData", "Local"), CACHE_DIR_NAME);
  }
  return path.join(env.XDG_CACHE_HOME || path.join(home, ".cache"), CACHE_DIR_NAME);
}

/** Path of a cached binary: `<cacheDir>/<release>/<assetName>`. */
export function cachedBinaryPath(cacheDir, release, assetName) {
  return path.join(cacheDir, release, assetName);
}

function entriesOf(dir) {
  try {
    return fs.readdirSync(dir);
  } catch {
    return [];
  }
}

/** The binaries of one release dir that run on the host, the host's own variant first. */
function hostBinaries(cacheDir, release, host) {
  const found = [];
  for (const name of entriesOf(path.join(cacheDir, release)).sort()) {
    const parsed = parseAssetName(name);
    if (!parsed || parsed.platform !== host.platform || parsed.arch !== host.arch) continue;
    found.push({ path: cachedBinaryPath(cacheDir, release, name), release, assetName: name, variant: parsed.variant });
  }
  return found.sort((a, b) => Number(b.variant === host.variant) - Number(a.variant === host.variant));
}

/**
 * The newest cached binary for this host, or null when none is cached. Within a
 * release, a binary matching `host.variant` wins over any other build of the host's platform.
 */
export function installedBinary({ cacheDir, host = { platform: process.platform, arch: process.arch, variant: null } }) {
  for (const release of entriesOf(cacheDir).sort(compareReleaseTags)) {
    const [best] = hostBinaries(cacheDir, release, host);
    if (best) return best;
  }
  return null;
}

/** Delete every cached release dir but `keep`. */
export function pruneOtherReleases(cacheDir, keep) {
  for (const release of entriesOf(cacheDir)) {
    if (release === keep) continue;
    try {
      fs.rmSync(path.join(cacheDir, release), { recursive: true, force: true });
    } catch {
      // a busy file on Windows: a stale release is a nuisance, not an error
    }
  }
}

/**
 * Put a resolved release's binary in the cache and return it with its `source`:
 * "cache" when it is already there (unless `force`), else "download".
 */
export async function installRelease(release, { cacheDir, force = false, ...transfer }) {
  const dest = cachedBinaryPath(cacheDir, release.tag, release.assetName);
  const installed = { path: dest, release: release.tag, assetName: release.assetName, variant: release.variant };
  if (!force && fs.existsSync(dest)) return { ...installed, source: "cache" };
  await download({ ...transfer, release, dest });
  pruneOtherReleases(cacheDir, release.tag);
  return { ...installed, source: "download" };
}
