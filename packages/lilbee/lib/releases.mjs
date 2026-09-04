/**
 * GitHub release listing with the build each release ships for a host:
 * the CUDA build the driver supports, the ROCm build when its gfx manifest
 * covers every host GPU, else the baseline build of the host's CPU family.
 */

import { assetNameFor } from "./assets.mjs";
import { detectHost } from "./detect.mjs";
import { launcherFetch } from "./fetch.mjs";

export const DEFAULT_REPO = "tobocop2/lilbee";
export const USER_AGENT = "lilbee-npm-launcher";

const GITHUB_API = "https://api.github.com";
const RELEASE_HISTORY_LIMIT = 10;
const RELEASE_PAGE_SIZE = 100;
const RELEASE_PAGE_BUDGET = 3;
const RATE_LIMIT_STATUSES = new Set([403, 429]);
const GITHUB_RATE_LIMITED = "GitHub's rate limit was reached; release checks reset within the hour.";
const GFX_MANIFEST_SUFFIX = ".gfx.txt";

/** An in-development build, tagged with a trailing `.dev<n>` (e.g. v0.6.90b420.dev711). */
export function isDevBuild(tag) {
  return /\.dev\d*$/i.test(tag);
}

/** Order release tags like v0.6.90b423 newest-first; numeric segments compare numerically. */
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

function githubHeaders(env) {
  const headers = { "user-agent": USER_AGENT, accept: "application/vnd.github+json" };
  const token = env.GITHUB_TOKEN || env.GH_TOKEN;
  if (token) headers.authorization = `Bearer ${token}`;
  return headers;
}

async function fetchGitHub(fetchImpl, url, env) {
  const res = await fetchImpl(url, { headers: githubHeaders(env) });
  if (RATE_LIMIT_STATUSES.has(res.status)) throw new Error(GITHUB_RATE_LIMITED);
  return res;
}

/** The hex sha256 GitHub reports for an asset, or null when the release carries none. */
function digestOf(asset) {
  const digest = asset.digest;
  return typeof digest === "string" && digest.startsWith("sha256:") ? digest.slice("sha256:".length) : null;
}

/** The gfx targets a manifest lists, or null when it is missing, unreachable, or empty. */
async function shippedGfxTargets(fetchImpl, manifest) {
  if (!manifest) return null;
  try {
    const res = await fetchImpl(manifest.browser_download_url, { headers: { "user-agent": USER_AGENT } });
    if (!res.ok) return null;
    const targets = (await res.text()).split("\n").map((line) => line.trim()).filter(Boolean);
    return targets.length ? new Set(targets) : null;
  } catch {
    return null;
  }
}

/** The GPU build the host's variant asks for, and the baseline build it falls back to. */
function buildsFor(host) {
  const compat = host.variant === "compat" || host.variant.startsWith("compat-");
  const baseline = compat ? "compat" : "default";
  const gpu = host.variant === baseline ? null : host.variant;
  return { baseline, gpu };
}

/** The baseline asset of `data` for the host, or null when the release ships none for it. */
function baselineAsset(data, host) {
  const { baseline } = buildsFor(host);
  let name;
  try {
    name = assetNameFor(host.platform, host.arch, baseline);
  } catch {
    return null;
  }
  return (data.assets ?? []).find((a) => a.name === name) ?? null;
}

/** The build of `data` this host should run: its GPU build when the release ships one it can run, else `fallback`. */
async function resolveBuild(data, fallback, host, fetchImpl) {
  const { baseline, gpu } = buildsFor(host);
  const find = (name) => (data.assets ?? []).find((a) => a.name === name);
  let pick = { variant: baseline, asset: fallback };
  if (gpu) {
    const gpuName = assetNameFor(host.platform, host.arch, gpu);
    const gpuAsset = find(gpuName);
    if (gpuAsset && (await hostCanRun(gpu, host, find(gpuName + GFX_MANIFEST_SUFFIX), fetchImpl))) {
      pick = { variant: gpu, asset: gpuAsset };
    }
  }
  return {
    tag: data.tag_name,
    dev: isDevBuild(data.tag_name),
    assetName: pick.asset.name,
    variant: pick.variant,
    size: pick.asset.size,
    digest: digestOf(pick.asset),
    url: pick.asset.browser_download_url,
  };
}

/** A ROCm build must ship kernels for every host GPU it is known to have; other builds always qualify. */
async function hostCanRun(variant, host, manifest, fetchImpl) {
  if (!variant.endsWith("rocm") || host.amdGfxTargets.length === 0) return true;
  const shipped = await shippedGfxTargets(fetchImpl, manifest);
  return shipped !== null && host.amdGfxTargets.every((target) => shipped.has(target));
}

async function resolveQuery(query) {
  const env = query.env ?? process.env;
  return {
    repo: query.repo ?? DEFAULT_REPO,
    includeDev: query.includeDev ?? false,
    env,
    host: query.host ?? (await detectHost(env)),
    fetchImpl: query.fetch ?? (await launcherFetch(env)),
  };
}

/** Releases that ship a baseline build for the host, newest first, within the per-kind quotas. */
async function collectInstallable(q, limit) {
  const found = [];
  let stable = 0;
  let dev = 0;
  for (let page = 1; page <= RELEASE_PAGE_BUDGET; page += 1) {
    const url = `${GITHUB_API}/repos/${q.repo}/releases?per_page=${RELEASE_PAGE_SIZE}&page=${page}`;
    const res = await fetchGitHub(q.fetchImpl, url, q.env);
    if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
    const releases = await res.json();
    for (const data of releases) {
      if (data.draft || data.prerelease) continue;
      const isDev = isDevBuild(data.tag_name);
      if (isDev && !q.includeDev) continue;
      if (isDev ? dev >= limit : stable >= limit) continue;
      const fallback = baselineAsset(data, q.host);
      if (!fallback) continue;
      found.push({ data, fallback });
      if (isDev) dev += 1;
      else stable += 1;
      if (stable >= limit && (!q.includeDev || dev >= limit)) return found;
    }
    if (releases.length < RELEASE_PAGE_SIZE) break;
  }
  return found;
}

/**
 * Published releases that ship a build for the host, newest first: up to `limit`
 * stable releases plus up to `limit` dev builds when `includeDev` is set.
 */
export async function listReleases(query = {}) {
  const q = await resolveQuery(query);
  const limit = query.limit ?? RELEASE_HISTORY_LIMIT;
  const candidates = await collectInstallable(q, limit);
  return Promise.all(candidates.map(({ data, fallback }) => resolveBuild(data, fallback, q.host, q.fetchImpl)));
}

/** The newest installable release on the chosen channel. Throws when none exists. */
export async function latestRelease(query = {}) {
  const [newest] = await listReleases({ ...query, limit: 1 });
  if (!newest) throw new Error("No installable lilbee release was found.");
  return newest;
}

/** One release by tag, resolved for the host. Throws when the tag or the host's build is missing. */
export async function releaseByTag(tag, query = {}) {
  const q = await resolveQuery(query);
  const url = `${GITHUB_API}/repos/${q.repo}/releases/tags/${tag}`;
  const res = await fetchGitHub(q.fetchImpl, url, q.env);
  if (res.status === 404) throw new Error(`Release ${tag} was not found in ${q.repo}.`);
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
  const data = await res.json();
  const fallback = baselineAsset(data, q.host);
  if (!fallback) throw new Error(`Release ${tag} has no build for ${q.host.platform}/${q.host.arch}.`);
  return resolveBuild(data, fallback, q.host, q.fetchImpl);
}
