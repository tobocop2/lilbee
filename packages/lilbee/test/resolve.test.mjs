import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { resolveBinary } from "../lib/resolve.mjs";

const DETECTION = { nvidia: { status: "skipped" }, amd: { status: "skipped" }, cpu: { status: "skipped" }, detectedAt: "2026-09-04T12:00:00.000Z" };
const HOST = { platform: "darwin", arch: "arm64", variant: "default", amdGfxTargets: [], detection: DETECTION };
const ASSET = "lilbee-macos-arm64";

const tmpCache = () => fs.mkdtempSync(path.join(os.tmpdir(), "lilbee-resolve-"));
const seed = (cache, release) => {
  fs.mkdirSync(path.join(cache, release), { recursive: true });
  fs.writeFileSync(path.join(cache, release, ASSET), "");
};

/** GitHub with the given tags (newest first); `down` makes every call fail. */
function github(tags, { down = false } = {}) {
  const calls = [];
  const releases = tags.map((tag) => ({
    tag_name: tag,
    assets: [{ name: ASSET, size: 1, digest: null, browser_download_url: `https://dl/${tag}` }],
  }));
  const fetch = async (url) => {
    calls.push(url);
    if (down) throw new Error("network down");
    const respond = (status, json) => ({ ok: status < 400, status, headers: { get: () => null }, body: null, json: async () => json, text: async () => "" });
    if (/\/releases\?/.test(url)) return respond(200, releases);
    const tag = /\/releases\/tags\/(.+)$/.exec(url)?.[1];
    const found = releases.find((r) => r.tag_name === tag);
    return found ? respond(200, found) : respond(404, {});
  };
  return { fetch, calls };
}

const resolve = (cache, gh, extra = {}) =>
  resolveBinary({ env: {}, cacheDir: cache, host: HOST, pinned: "vPin", fetch: gh.fetch, ...extra });

test("LILBEE_BIN wins, and a missing LILBEE_BIN is an error not a fallthrough", async () => {
  const cache = tmpCache();
  const bin = path.join(cache, "custom-lilbee");
  fs.writeFileSync(bin, "");
  const found = await resolve(cache, github(["v1"]), { env: { LILBEE_BIN: bin } });
  assert.deepEqual(found, { path: bin, source: "env", release: null, download: null });
  await assert.rejects(resolve(cache, github(["v1"]), { env: { LILBEE_BIN: "/gone" } }), /nothing is there/);
  fs.rmSync(cache, { recursive: true, force: true });
});

test("a run with any cached release uses the newest one, no network", async () => {
  const cache = tmpCache();
  seed(cache, "v0.6.90b423");
  seed(cache, "v0.6.90b425");
  const gh = github(["v0.6.90b426"]);
  const r = await resolve(cache, gh);
  assert.deepEqual(r, { path: path.join(cache, "v0.6.90b425", ASSET), source: "cache", release: "v0.6.90b425", download: null });
  assert.equal(gh.calls.length, 0);
  fs.rmSync(cache, { recursive: true, force: true });
});

test("a fresh install plans a download of the latest release", async () => {
  const cache = tmpCache();
  const r = await resolve(cache, github(["v0.6.90b425", "v0.6.90b423"]));
  assert.equal(r.source, "download");
  assert.equal(r.release, "v0.6.90b425");
  assert.equal(r.path, path.join(cache, "v0.6.90b425", ASSET));
  assert.equal(r.download.url, "https://dl/v0.6.90b425");
  fs.rmSync(cache, { recursive: true, force: true });
});

test("when the latest lookup fails, the pinned release is the fallback", async () => {
  const cache = tmpCache();
  const logs = [];
  const gh = github(["v0.6.90b423"]);
  const flaky = { fetch: (url, init) => (/\/releases\?/.test(url) ? Promise.reject(new Error("network down")) : gh.fetch(url, init)) };
  const r = await resolve(cache, flaky, { pinned: "v0.6.90b423", log: (m) => logs.push(m) });
  assert.equal(r.release, "v0.6.90b423");
  assert.equal(r.source, "download");
  assert.ok(logs.includes("lilbee: could not look up the latest release; using the tested v0.6.90b423."));
  fs.rmSync(cache, { recursive: true, force: true });
});

test("refresh re-resolves latest past a cached binary, and cache satisfies it without a download", async () => {
  const cache = tmpCache();
  seed(cache, "v0.6.90b425");
  const same = await resolve(cache, github(["v0.6.90b425"]), { refresh: true });
  assert.deepEqual(same, { path: path.join(cache, "v0.6.90b425", ASSET), source: "cache", release: "v0.6.90b425", download: null });

  const upgraded = await resolve(cache, github(["v0.6.90b426", "v0.6.90b425"]), { refresh: true });
  assert.equal(upgraded.source, "download");
  assert.equal(upgraded.release, "v0.6.90b426");
  fs.rmSync(cache, { recursive: true, force: true });
});

test("LILBEE_RELEASE and the prepare tag pin one release; the tag wins over the variable", async () => {
  const cache = tmpCache();
  seed(cache, "v0.6.90b425");
  const gh = github(["v0.6.90b426", "v0.6.90b425", "v0.6.90b410", "v0.6.90b400"]);
  const fromEnv = await resolve(cache, gh, { env: { LILBEE_RELEASE: "v0.6.90b410" } });
  assert.equal(fromEnv.release, "v0.6.90b410");
  assert.equal(fromEnv.source, "download");
  assert.match(gh.calls.at(-1), /releases\/tags\/v0\.6\.90b410$/);

  const fromArg = await resolve(cache, gh, { env: { LILBEE_RELEASE: "v0.6.90b410" }, tag: "v0.6.90b400" });
  assert.equal(fromArg.release, "v0.6.90b400");

  const cached = await resolve(cache, gh, { tag: "v0.6.90b425" });
  assert.equal(cached.source, "cache");
  fs.rmSync(cache, { recursive: true, force: true });
});

test("the dev channel makes the newest dev build the latest", async () => {
  const cache = tmpCache();
  const gh = github(["v0.6.90b426.dev3", "v0.6.90b425"]);
  assert.equal((await resolve(cache, gh)).release, "v0.6.90b425");
  assert.equal((await resolve(cache, gh, { includeDev: true })).release, "v0.6.90b426.dev3");
  fs.rmSync(cache, { recursive: true, force: true });
});
