import { test } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { cacheDir, cachedBinaryPath, dataRoot, resolveBinary, sharedRootBinary } from "../lib/resolve.mjs";

const noDeps = { existsSync: () => false, whichAllSync: () => [], download: async () => {} };

test("cacheDir honors LILBEE_MCP_CACHE and platform conventions", () => {
  assert.equal(cacheDir({ LILBEE_MCP_CACHE: "/tmp/c" }), "/tmp/c");
  assert.ok(cacheDir({}, "darwin").endsWith(path.join("Library", "Caches", "lilbee-npm")));
  assert.ok(cacheDir({ XDG_CACHE_HOME: "/xdg" }, "linux").startsWith("/xdg"));
  assert.ok(cacheDir({ LOCALAPPDATA: "C:\\LA" }, "win32").includes("lilbee-npm"));
});

test("LILBEE_BIN wins, and a missing LILBEE_BIN is an error not a fallthrough", async () => {
  const found = await resolveBinary({
    env: { LILBEE_BIN: "/custom/lilbee" },
    release: "r",
    assetName: "a",
    deps: { ...noDeps, existsSync: (p) => p === "/custom/lilbee" },
  });
  assert.deepEqual(found, { path: "/custom/lilbee", source: "env" });

  await assert.rejects(
    resolveBinary({ env: { LILBEE_BIN: "/gone" }, release: "r", assetName: "a", deps: noDeps }),
    /nothing is there/
  );
});

test("PATH beats cache beats download", async () => {
  const onPath = await resolveBinary({
    env: {},
    release: "r",
    assetName: "a",
    deps: { ...noDeps, whichAllSync: () => ["/usr/local/bin/lilbee"] },
  });
  assert.deepEqual(onPath, { path: "/usr/local/bin/lilbee", source: "path" });

  const env = { LILBEE_MCP_CACHE: "/cache" };
  const cachedPath = cachedBinaryPath(env, "v1", "lilbee-macos-arm64");
  const cached = await resolveBinary({
    env,
    release: "v1",
    assetName: "lilbee-macos-arm64",
    deps: { ...noDeps, existsSync: (p) => p === cachedPath },
  });
  assert.deepEqual(cached, { path: cachedPath, source: "cache" });

  const calls = [];
  const downloaded = await resolveBinary({
    env,
    release: "v1",
    assetName: "lilbee-macos-arm64",
    deps: { ...noDeps, download: async (o) => calls.push(o.dest) },
  });
  assert.equal(downloaded.source, "download");
  assert.deepEqual(calls, [cachedPath]);
});

test("shared-root binary beats cache, loses to PATH", async () => {
  const env = { LILBEE_MCP_CACHE: "/cache" };
  const shared = sharedRootBinary(env);
  const viaShared = await resolveBinary({
    env,
    release: "v1",
    assetName: "a",
    deps: { ...noDeps, existsSync: (p) => p === shared },
  });
  assert.deepEqual(viaShared, { path: shared, source: "shared-root" });

  const onPath = await resolveBinary({
    env,
    release: "v1",
    assetName: "a",
    deps: { ...noDeps, whichAllSync: () => ["/usr/local/bin/lilbee"], existsSync: (p) => p === shared },
  });
  assert.equal(onPath.source, "path");
});

test("dataRoot honors LILBEE_DATA and platform defaults", () => {
  assert.equal(dataRoot({ LILBEE_DATA: "/d" }), "/d");
  assert.ok(dataRoot({}, "darwin").endsWith(path.join("Application Support", "lilbee")));
  assert.ok(dataRoot({ XDG_DATA_HOME: "/xdg" }, "linux").startsWith("/xdg"));
});
