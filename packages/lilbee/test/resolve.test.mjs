import { test } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { cacheDir, cachedBinaryPath, resolveBinary } from "../lib/resolve.mjs";

const noDeps = { existsSync: () => false, whichSync: () => null, download: async () => {} };

test("cacheDir honors LILBEE_MCP_CACHE and platform conventions", () => {
  assert.equal(cacheDir({ LILBEE_MCP_CACHE: "/tmp/c" }), "/tmp/c");
  assert.ok(cacheDir({}, "darwin").endsWith(path.join("Library", "Caches", "lilbee-mcp")));
  assert.ok(cacheDir({ XDG_CACHE_HOME: "/xdg" }, "linux").startsWith("/xdg"));
  assert.ok(cacheDir({ LOCALAPPDATA: "C:\\LA" }, "win32").includes("lilbee-mcp"));
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
    deps: { ...noDeps, whichSync: () => "/usr/local/bin/lilbee" },
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
