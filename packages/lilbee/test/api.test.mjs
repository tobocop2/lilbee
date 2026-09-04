import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { Readable } from "node:stream";
import * as api from "../lib/api.mjs";

const PAYLOAD = Buffer.from("binary ".repeat(32));
const DIGEST = createHash("sha256").update(PAYLOAD).digest("hex");
const HOST = { platform: "linux", arch: "x64", variant: "cu125", amdGfxTargets: [] };
const ASSETS = ["lilbee-linux-x86_64", "lilbee-linux-x86_64-cu125"];

const tmpCache = () => fs.mkdtempSync(path.join(os.tmpdir(), "lilbee-api-"));

/** GitHub plus the asset host: release pages by tag list, binaries served from PAYLOAD. */
function stub(tags) {
  const calls = [];
  const releases = tags.map((tag) => ({
    tag_name: tag,
    assets: ASSETS.map((name) => ({ name, size: PAYLOAD.length, digest: `sha256:${DIGEST}`, browser_download_url: `https://dl/${tag}/${name}` })),
  }));
  const fetch = async (url, init = {}) => {
    calls.push(url);
    const respond = (status, json = null, body = null) => ({
      ok: status < 400,
      status,
      headers: { get: () => null },
      body,
      json: async () => json,
      text: async () => "",
    });
    if (/\/releases\?/.test(url)) return respond(200, releases);
    const byTag = /\/releases\/tags\/(.+)$/.exec(url);
    if (byTag) {
      const found = releases.find((r) => r.tag_name === byTag[1]);
      return found ? respond(200, found) : respond(404, {});
    }
    if (url.startsWith("https://dl/")) return respond(200, null, Readable.from([PAYLOAD]));
    throw new Error(`unexpected ${url}`);
  };
  return { fetch, calls };
}

const seed = (cache, release, assetName) => {
  fs.mkdirSync(path.join(cache, release), { recursive: true });
  fs.writeFileSync(path.join(cache, release, assetName), "cached");
};

test("the package entry exports the whole contract", () => {
  for (const name of [
    "detectHost", "assetNameFor", "parseAssetName", "isDevBuild", "compareReleaseTags",
    "listReleases", "latestRelease", "releaseByTag", "cacheDir", "cachedBinaryPath",
    "installedBinary", "ensureBinary", "DownloadCanceledError", "isDownloadCanceled",
  ]) {
    assert.equal(typeof api[name], "function", name);
  }
});

test("a fresh cache downloads the latest release for the host", async () => {
  const cache = tmpCache();
  const gh = stub(["v2", "v1"]);
  const progress = [];
  const got = await api.ensureBinary({ cacheDir: cache, host: HOST, fetch: gh.fetch, onProgress: (p) => progress.push(p) });
  assert.deepEqual(got, {
    path: path.join(cache, "v2", "lilbee-linux-x86_64-cu125"),
    release: "v2",
    assetName: "lilbee-linux-x86_64-cu125",
    variant: "cu125",
    source: "download",
  });
  assert.ok(Buffer.from(fs.readFileSync(got.path)).equals(PAYLOAD));
  assert.ok(progress.length > 0);
  assert.deepEqual(gh.calls.filter((u) => u.includes("api.github.com")).length, 1);
  fs.rmSync(cache, { recursive: true, force: true });
});

test("a cached binary of any variant is returned without touching the network", async () => {
  const cache = tmpCache();
  seed(cache, "v1", "lilbee-linux-x86_64");
  const gh = stub(["v2", "v1"]);
  const got = await api.ensureBinary({ cacheDir: cache, host: HOST, fetch: gh.fetch });
  assert.equal(got.source, "cache");
  assert.equal(got.variant, "default");
  assert.equal(gh.calls.length, 0);
  fs.rmSync(cache, { recursive: true, force: true });
});

test("refresh re-resolves latest: a matching cache is kept, an older one is replaced and pruned", async () => {
  const cache = tmpCache();
  seed(cache, "v2", "lilbee-linux-x86_64-cu125");
  const same = await api.ensureBinary({ cacheDir: cache, host: HOST, fetch: stub(["v2"]).fetch, refresh: true });
  assert.equal(same.source, "cache");
  assert.equal(fs.readFileSync(same.path, "utf8"), "cached");

  seed(cache, "v1", "lilbee-linux-x86_64-cu125");
  const newer = await api.ensureBinary({ cacheDir: cache, host: HOST, fetch: stub(["v3", "v2"]).fetch, refresh: true });
  assert.equal(newer.source, "download");
  assert.equal(newer.release, "v3");
  assert.deepEqual(fs.readdirSync(cache), ["v3"]);
  fs.rmSync(cache, { recursive: true, force: true });
});

test("after a hardware change, refresh switches builds within the same release and drops the old one", async () => {
  const cache = tmpCache();
  seed(cache, "v2", "lilbee-linux-x86_64");
  const got = await api.ensureBinary({ cacheDir: cache, host: HOST, fetch: stub(["v2"]).fetch, refresh: true });
  assert.equal(got.source, "download");
  assert.equal(got.assetName, "lilbee-linux-x86_64-cu125");
  assert.deepEqual(fs.readdirSync(path.join(cache, "v2")), ["lilbee-linux-x86_64-cu125"]);
  fs.rmSync(cache, { recursive: true, force: true });
});

test("an explicit release tag installs that release, cached or not", async () => {
  const cache = tmpCache();
  const gh = stub(["v3", "v2", "v1"]);
  const got = await api.ensureBinary({ cacheDir: cache, host: HOST, fetch: gh.fetch, release: "v1" });
  assert.equal(got.release, "v1");
  assert.equal(got.source, "download");
  assert.match(gh.calls[0], /releases\/tags\/v1$/);
  const again = await api.ensureBinary({ cacheDir: cache, host: HOST, fetch: gh.fetch, release: "v1" });
  assert.equal(again.source, "cache");
  await assert.rejects(api.ensureBinary({ cacheDir: cache, host: HOST, fetch: gh.fetch, release: "v0" }), /was not found/);
  fs.rmSync(cache, { recursive: true, force: true });
});

test("force reinstalls the resolved release over the cached copy", async () => {
  const cache = tmpCache();
  seed(cache, "v2", "lilbee-linux-x86_64-cu125");
  const got = await api.ensureBinary({ cacheDir: cache, host: HOST, fetch: stub(["v2"]).fetch, force: true });
  assert.equal(got.source, "download");
  assert.equal(got.release, "v2");
  assert.ok(Buffer.from(fs.readFileSync(got.path)).equals(PAYLOAD));
  fs.rmSync(cache, { recursive: true, force: true });
});

test("a cancelled download rejects and leaves the previously cached release in place", async () => {
  const cache = tmpCache();
  seed(cache, "v1", "lilbee-linux-x86_64-cu125");
  const controller = new AbortController();
  const gh = stub(["v2", "v1"]);
  const fetch = async (url, init) => {
    const res = await gh.fetch(url, init);
    if (url.startsWith("https://dl/")) {
      const body = new Readable({ read() {} });
      body.push(PAYLOAD.subarray(0, 8));
      setTimeout(() => controller.abort(), 10);
      return { ...res, body };
    }
    return res;
  };
  await assert.rejects(api.ensureBinary({ cacheDir: cache, host: HOST, fetch, refresh: true, signal: controller.signal }), (e) => api.isDownloadCanceled(e));
  assert.deepEqual(fs.readdirSync(cache).sort(), ["v1", "v2"]);
  assert.deepEqual(fs.readdirSync(path.join(cache, "v2")), []);
  assert.equal(fs.readFileSync(path.join(cache, "v1", "lilbee-linux-x86_64-cu125"), "utf8"), "cached");
  fs.rmSync(cache, { recursive: true, force: true });
});
