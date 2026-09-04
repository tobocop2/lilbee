/**
 * The lilbee npm launcher's programmatic API: resolve, download, and list
 * lilbee release binaries with the CLI's detection, verification, and cache
 * layout. Silent, and nothing blocks the event loop.
 */

import { cacheDir, cachedBinaryPath, installedBinary, installRelease } from "./cache.mjs";
import { detectHost } from "./detect.mjs";
import { compareReleaseTags, isDevBuild, latestRelease, listReleases, releaseByTag } from "./releases.mjs";

export { assetNameFor, parseAssetName } from "./assets.mjs";
export { DownloadCanceledError, isDownloadCanceled } from "./download.mjs";
export { cacheDir, cachedBinaryPath, compareReleaseTags, detectHost, installedBinary, isDevBuild, latestRelease, listReleases, releaseByTag };

/**
 * Make a lilbee binary available in `cacheDir` and return it. Without `release`
 * the cached binary is used as is; `refresh` re-resolves the latest release and
 * `force` downloads again even when the resolved release is cached.
 */
export async function ensureBinary({ cacheDir, release, refresh = false, force = false, onProgress, signal, log = () => {}, ...query }) {
  const env = query.env ?? process.env;
  const host = query.host ?? (await detectHost(env));
  const q = { ...query, env, host };
  let resolved;
  if (release) {
    resolved = await releaseByTag(release, q);
  } else {
    const installed = refresh ? null : installedBinary({ cacheDir, host });
    if (installed && !force) return { ...installed, source: "cache" };
    resolved = installed ? await releaseByTag(installed.release, q) : await latestRelease(q);
  }
  return installRelease(resolved, { cacheDir, force, fetch: q.fetch, env, onProgress, signal, log });
}
