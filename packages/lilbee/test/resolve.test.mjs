import { test } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { cacheDir, cachedBinaryPath, compareReleaseTags, resolveBinary } from "../lib/resolve.mjs";

const ENV = { LILBEE_MCP_CACHE: "/cache" };
const ASSET = "lilbee-macos-arm64";
const at = (release) => cachedBinaryPath(ENV, release, ASSET);

/** deps with a cache holding `releases` and a network reporting `latest`. */
function deps({ releases = [], latest = null, downloads = [], removed = [] } = {}) {
  return {
    existsSync: (p) => releases.some((r) => p === at(r)),
    readdirSync: (p) => {
      if (p !== "/cache") throw new Error("no dir");
      return [...releases];
    },
    rmSync: (p) => removed.push(p),
    latestTag: async () => {
      if (!latest) throw new Error("network down");
      return latest;
    },
    download: async (o) => downloads.push(o),
  };
}

test("cacheDir honors LILBEE_MCP_CACHE and platform conventions", () => {
  assert.equal(cacheDir({ LILBEE_MCP_CACHE: "/tmp/c" }), "/tmp/c");
  assert.ok(cacheDir({}, "darwin").endsWith(path.join("Library", "Caches", "lilbee-npm")));
  assert.ok(cacheDir({ XDG_CACHE_HOME: "/xdg" }, "linux").startsWith("/xdg"));
  assert.ok(cacheDir({ LOCALAPPDATA: "C:\\LA" }, "win32").includes("lilbee-npm"));
});

test("release tags order numerically, newest first", () => {
  const tags = ["v0.6.90b423", "v0.6.90b425", "v0.7.0", "v0.6.91"];
  assert.deepEqual(tags.sort(compareReleaseTags), ["v0.7.0", "v0.6.91", "v0.6.90b425", "v0.6.90b423"]);
});

test("LILBEE_BIN wins, and a missing LILBEE_BIN is an error not a fallthrough", async () => {
  const found = await resolveBinary({
    env: { LILBEE_BIN: "/custom/lilbee" },
    release: "vPin",
    assetName: ASSET,
    deps: { ...deps(), existsSync: (p) => p === "/custom/lilbee" },
  });
  assert.deepEqual(found, { path: "/custom/lilbee", source: "env", release: null });

  await assert.rejects(
    resolveBinary({ env: { LILBEE_BIN: "/gone" }, release: "vPin", assetName: ASSET, deps: deps() }),
    /nothing is there/
  );
});

test("a run with any cached release uses the newest one, no network", async () => {
  const d = deps({ releases: ["v0.6.90b423", "v0.6.90b425"] });
  d.latestTag = async () => {
    throw new Error("must not touch the network");
  };
  const r = await resolveBinary({ env: ENV, release: "vPin", assetName: ASSET, deps: d });
  assert.deepEqual(r, { path: at("v0.6.90b425"), source: "cache", release: "v0.6.90b425" });
});

test("a fresh install downloads the latest release", async () => {
  const downloads = [];
  const r = await resolveBinary({
    env: ENV,
    release: "v0.6.90b423",
    assetName: ASSET,
    deps: deps({ latest: "v0.6.90b425", downloads }),
  });
  assert.equal(r.source, "download");
  assert.equal(r.release, "v0.6.90b425");
  assert.deepEqual(downloads.map((o) => o.release), ["v0.6.90b425"]);
});

test("when the latest lookup fails, the pinned release is the fallback", async () => {
  const downloads = [];
  const logs = [];
  const d = { ...deps({ downloads }), log: (m) => logs.push(m) };
  const r = await resolveBinary({ env: ENV, release: "v0.6.90b423", assetName: ASSET, deps: d });
  assert.equal(r.release, "v0.6.90b423");
  assert.equal(r.source, "download");
  assert.match(logs.join("\n"), /tested v0\.6\.90b423/);
});

test("refresh re-resolves latest past a cached binary, and cache satisfies it without a download", async () => {
  const downloads = [];
  const cachedLatest = deps({ releases: ["v0.6.90b425"], latest: "v0.6.90b425", downloads });
  const same = await resolveBinary({
    env: ENV,
    release: "vPin",
    assetName: ASSET,
    refresh: true,
    deps: cachedLatest,
  });
  assert.deepEqual(same, { path: at("v0.6.90b425"), source: "cache", release: "v0.6.90b425" });
  assert.equal(downloads.length, 0);

  const stale = deps({ releases: ["v0.6.90b423"], latest: "v0.6.90b425", downloads });
  const upgraded = await resolveBinary({
    env: ENV,
    release: "vPin",
    assetName: ASSET,
    refresh: true,
    deps: stale,
  });
  assert.equal(upgraded.source, "download");
  assert.equal(upgraded.release, "v0.6.90b425");
});

test("LILBEE_RELEASE overrides latest entirely", async () => {
  const downloads = [];
  const d = deps({ releases: ["v0.6.90b425"], latest: "v0.6.90b426", downloads });
  const r = await resolveBinary({
    env: { ...ENV, LILBEE_RELEASE: "v0.6.90b410" },
    release: "vPin",
    assetName: ASSET,
    deps: d,
  });
  assert.equal(r.release, "v0.6.90b410");
  assert.equal(r.source, "download");
});

test("a finished download prunes the older cached releases", async () => {
  const removed = [];
  const d = deps({ releases: ["v0.6.90b423"], latest: "v0.6.90b425", removed });
  await resolveBinary({ env: ENV, release: "vPin", assetName: ASSET, refresh: true, deps: d });
  assert.deepEqual(removed, ["/cache/v0.6.90b423"]);
});

test("lilbee installs from other package managers are ignored: the launcher's own download wins", async () => {
  // existsSync says yes to everything EXCEPT the launcher's cache, simulating
  // a machine covered in brew/flatpak/pip lilbees.
  const downloads = [];
  const d = {
    ...deps({ latest: "vX", downloads }),
    existsSync: (p) => !p.startsWith("/cache"),
    readdirSync: () => [],
  };
  const r = await resolveBinary({ env: ENV, release: "vPin", assetName: ASSET, deps: d });
  assert.equal(r.source, "download");
  assert.deepEqual(downloads.map((o) => o.dest), [at("vX")]);
});
