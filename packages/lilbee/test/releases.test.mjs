import { test } from "node:test";
import assert from "node:assert/strict";
import { isDevBuild, latestRelease, listReleases, releaseByTag } from "../lib/releases.mjs";

const LINUX = { platform: "linux", arch: "x64", variant: "default", amdGfxTargets: [] };
const CUDA = { ...LINUX, variant: "cu125" };
const ROCM = { ...LINUX, variant: "rocm", amdGfxTargets: ["gfx942", "gfx1100"] };

const asset = (name, extra = {}) => ({
  name,
  size: 100,
  digest: `sha256:${name}`,
  browser_download_url: `https://dl/${name}`,
  ...extra,
});

const release = (tag, assets, extra = {}) => ({ tag_name: tag, assets: assets.map((a) => (typeof a === "string" ? asset(a) : a)), ...extra });

const LINUX_ASSETS = ["lilbee-linux-x86_64", "lilbee-linux-x86_64-cu125", "lilbee-linux-x86_64-rocm", "lilbee-macos-arm64"];

/**
 * A GitHub stand-in: `pages` are the release list pages, `tags` the per-tag
 * lookups, `manifests` the gfx.txt bodies by asset name. Records every request.
 */
function github({ pages = [], tags = {}, manifests = {}, status = null } = {}) {
  const calls = [];
  const fetch = async (url, init = {}) => {
    calls.push({ url, headers: init.headers ?? {} });
    const respond = (code, json, text = "") => ({
      ok: code < 400,
      status: code,
      headers: { get: () => null },
      body: null,
      json: async () => json,
      text: async () => text,
    });
    if (status) return respond(status, {});
    const page = /\/releases\?per_page=(\d+)&page=(\d+)$/.exec(url);
    if (page) return respond(200, pages[Number(page[2]) - 1] ?? []);
    const tag = /\/releases\/tags\/([^/]+)$/.exec(url);
    if (tag) return tags[tag[1]] ? respond(200, tags[tag[1]]) : respond(404, { message: "Not Found" });
    const manifest = /\/dl\/(.+\.gfx\.txt)$/.exec(url);
    if (manifest) return manifest[1] in manifests ? respond(200, null, manifests[manifest[1]]) : respond(404, null);
    throw new Error(`unexpected request ${url}`);
  };
  return { fetch, calls };
}

test("isDevBuild recognises the .dev suffix", () => {
  assert.equal(isDevBuild("v0.6.90b420.dev711"), true);
  assert.equal(isDevBuild("v0.6.90b420"), false);
});

test("stable and dev builds fill separate quotas, so a run of dev builds cannot hide stable ones", async () => {
  const devs = Array.from({ length: 100 }, (_, i) => release(`v0.6.90b430.dev${900 - i}`, LINUX_ASSETS));
  const stables = [release("v0.6.90b430", LINUX_ASSETS), release("v0.6.90b429", LINUX_ASSETS), release("v0.6.90b428", LINUX_ASSETS)];
  const gh = github({ pages: [devs, stables] });

  const stable = await listReleases({ host: LINUX, fetch: gh.fetch, limit: 2 });
  assert.deepEqual(stable.map((r) => r.tag), ["v0.6.90b430", "v0.6.90b429"]);
  assert.ok(stable.every((r) => r.dev === false));

  const both = await listReleases({ host: LINUX, fetch: gh.fetch, limit: 2, includeDev: true });
  assert.deepEqual(both.map((r) => r.tag), ["v0.6.90b430.dev900", "v0.6.90b430.dev899", "v0.6.90b430", "v0.6.90b429"]);
  assert.deepEqual(both.map((r) => r.dev), [true, true, false, false]);
});

test("paging stops at a short page and never reads past the page budget", async () => {
  const gh = github({ pages: [[release("v1", LINUX_ASSETS)]] });
  assert.equal((await listReleases({ host: LINUX, fetch: gh.fetch, limit: 5 })).length, 1);
  assert.equal(gh.calls.length, 1);

  const full = Array.from({ length: 100 }, (_, i) => release(`v0.1.${999 - i}`, ["lilbee-macos-arm64"]));
  const busy = github({ pages: [full, full, full, full] });
  assert.deepEqual(await listReleases({ host: LINUX, fetch: busy.fetch, limit: 5 }), []);
  assert.equal(busy.calls.length, 3);
});

test("drafts, prereleases, and releases without the host's baseline build are left out", async () => {
  const gh = github({
    pages: [[
      release("v5", LINUX_ASSETS, { draft: true }),
      release("v4", LINUX_ASSETS, { prerelease: true }),
      release("v3", ["lilbee-macos-arm64", "lilbee-linux-x86_64-cu125"]),
      release("v2", LINUX_ASSETS),
    ]],
  });
  assert.deepEqual((await listReleases({ host: CUDA, fetch: gh.fetch })).map((r) => r.tag), ["v2"]);
});

test("a CUDA host takes the matching CUDA build and falls back to the default build", async () => {
  const gh = github({ pages: [[release("v2", LINUX_ASSETS), release("v1", ["lilbee-linux-x86_64"])]] });
  const [v2, v1] = await listReleases({ host: CUDA, fetch: gh.fetch });
  assert.deepEqual(v2, {
    tag: "v2",
    dev: false,
    assetName: "lilbee-linux-x86_64-cu125",
    variant: "cu125",
    size: 100,
    digest: "lilbee-linux-x86_64-cu125",
    url: "https://dl/lilbee-linux-x86_64-cu125",
  });
  assert.equal(v1.variant, "default");
  assert.equal(v1.assetName, "lilbee-linux-x86_64");
});

test("the ROCm build needs a gfx manifest that covers every host GPU, not just one", async () => {
  const manifest = "lilbee-linux-x86_64-rocm.gfx.txt";
  const withManifest = [...LINUX_ASSETS, manifest];
  const covering = { [manifest]: "gfx942\ngfx1100\ngfx1201\n" };
  const partial = { [manifest]: "gfx942\n" };

  const all = await listReleases({ host: ROCM, fetch: github({ pages: [[release("v3", withManifest)]], manifests: covering }).fetch });
  assert.equal(all[0].variant, "rocm");
  assert.equal(all[0].assetName, "lilbee-linux-x86_64-rocm");

  const some = await listReleases({ host: ROCM, fetch: github({ pages: [[release("v3", withManifest)]], manifests: partial }).fetch });
  assert.equal(some[0].variant, "default");

  const none = await listReleases({ host: ROCM, fetch: github({ pages: [[release("v1", LINUX_ASSETS)]] }).fetch });
  assert.equal(none[0].variant, "default");

  const empty = await listReleases({ host: ROCM, fetch: github({ pages: [[release("v3", withManifest)]], manifests: { [manifest]: "\n" } }).fetch });
  assert.equal(empty[0].variant, "default");
});

test("a ROCm host with no readable gfx targets takes the rocm build the release ships", async () => {
  const blind = { ...ROCM, amdGfxTargets: [] };
  const gh = github({ pages: [[release("v1", LINUX_ASSETS)]] });
  assert.equal((await listReleases({ host: blind, fetch: gh.fetch }))[0].variant, "rocm");
});

test("compat hosts use the compat build as their baseline and compose the GPU flavor the same way", async () => {
  const compatAssets = ["lilbee-linux-x86_64", "lilbee-compat-linux-x86_64", "lilbee-compat-linux-x86_64-rocm", "lilbee-compat-linux-x86_64-rocm.gfx.txt"];
  const manifests = { "lilbee-compat-linux-x86_64-rocm.gfx.txt": "gfx942\ngfx1100\n" };
  const gh = github({ pages: [[release("v2", compatAssets), release("v1", ["lilbee-linux-x86_64"])]], manifests });

  const compatRocm = await listReleases({ host: { ...ROCM, variant: "compat-rocm" }, fetch: gh.fetch });
  assert.deepEqual(compatRocm.map((r) => [r.tag, r.assetName]), [["v2", "lilbee-compat-linux-x86_64-rocm"]]);

  const compatCuda = await listReleases({ host: { ...LINUX, variant: "compat-cu124" }, fetch: gh.fetch });
  assert.deepEqual(compatCuda.map((r) => [r.tag, r.assetName, r.variant]), [["v2", "lilbee-compat-linux-x86_64", "compat"]]);
});

test("a host lilbee publishes no build for sees no releases", async () => {
  const gh = github({ pages: [[release("v1", LINUX_ASSETS)]] });
  assert.deepEqual(await listReleases({ host: { ...LINUX, arch: "arm64" }, fetch: gh.fetch }), []);
});

test("GitHub's rate limit surfaces as one plain message", async () => {
  for (const status of [403, 429]) {
    await assert.rejects(listReleases({ host: LINUX, fetch: github({ status }).fetch }), /rate limit was reached; release checks reset within the hour/);
  }
  await assert.rejects(listReleases({ host: LINUX, fetch: github({ status: 500 }).fetch }), /HTTP 500/);
});

test("a GitHub token from the environment rides along as a bearer header; none is sent otherwise", async () => {
  const gh = github({ pages: [[release("v1", LINUX_ASSETS)]] });
  await listReleases({ host: LINUX, fetch: gh.fetch, env: { GITHUB_TOKEN: "ghp_x" } });
  assert.equal(gh.calls[0].headers.authorization, "Bearer ghp_x");
  await listReleases({ host: LINUX, fetch: gh.fetch, env: { GH_TOKEN: "gho_y" } });
  assert.equal(gh.calls[1].headers.authorization, "Bearer gho_y");
  await listReleases({ host: LINUX, fetch: gh.fetch, env: {} });
  assert.equal("authorization" in gh.calls[2].headers, false);
  assert.match(gh.calls[2].headers["user-agent"], /lilbee/);
});

test("latestRelease is the newest entry on the channel and fails plainly when there is none", async () => {
  const gh = github({ pages: [[release("v2.dev5", LINUX_ASSETS), release("v2", LINUX_ASSETS), release("v1", LINUX_ASSETS)]] });
  assert.equal((await latestRelease({ host: LINUX, fetch: gh.fetch })).tag, "v2");
  assert.equal((await latestRelease({ host: LINUX, fetch: gh.fetch, includeDev: true })).tag, "v2.dev5");
  await assert.rejects(latestRelease({ host: LINUX, fetch: github({ pages: [[]] }).fetch }), /No installable lilbee release/);
});

test("releaseByTag resolves one release the same way and names a missing tag or build", async () => {
  const gh = github({ tags: { "v0.6.90b432": release("v0.6.90b432", LINUX_ASSETS), "v0.6.90b432.dev3": release("v0.6.90b432.dev3", ["lilbee-macos-arm64"]) } });
  const r = await releaseByTag("v0.6.90b432", { host: CUDA, fetch: gh.fetch });
  assert.equal(r.assetName, "lilbee-linux-x86_64-cu125");
  assert.equal(r.dev, false);
  assert.match(gh.calls[0].url, /\/repos\/tobocop2\/lilbee\/releases\/tags\/v0\.6\.90b432$/);
  await assert.rejects(releaseByTag("v0.0.0", { host: CUDA, fetch: gh.fetch }), /Release v0\.0\.0 was not found/);
  await assert.rejects(releaseByTag("v0.6.90b432.dev3", { host: CUDA, fetch: gh.fetch }), /no build for linux\/x64/);
});

test("the repo is configurable", async () => {
  const gh = github({ pages: [[release("v1", LINUX_ASSETS)]] });
  await listReleases({ host: LINUX, fetch: gh.fetch, repo: "acme/fork" });
  assert.match(gh.calls[0].url, /\/repos\/acme\/fork\/releases\?per_page=100&page=1$/);
});

import { LauncherError } from "../lib/errors.mjs";

test("release query failures carry LauncherError codes: rate-limited, http, no-release", async () => {
  const limited = github({ status: 403 });
  await assert.rejects(listReleases({ host: LINUX, fetch: limited.fetch }), (err) => err instanceof LauncherError && err.code === "rate-limited" && err.status === 403);
  const broken = github({ status: 500 });
  await assert.rejects(listReleases({ host: LINUX, fetch: broken.fetch }), (err) => err instanceof LauncherError && err.code === "http" && err.status === 500);
  const empty = github({ pages: [[]] });
  await assert.rejects(latestRelease({ host: LINUX, fetch: empty.fetch }), (err) => err instanceof LauncherError && err.code === "no-release");
  await assert.rejects(releaseByTag("v404", { host: LINUX, fetch: empty.fetch }), (err) => err instanceof LauncherError && err.code === "no-release" && err.status === 404);
});
