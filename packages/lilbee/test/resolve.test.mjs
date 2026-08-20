import { test } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { cacheDir, cachedBinaryPath, resolveBinary } from "../lib/resolve.mjs";

const noDeps = { existsSync: () => false, download: async () => {} };

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

test("cache beats download", async () => {
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

test("lilbee installs from other package managers are ignored: the pinned download wins", async () => {
  // existsSync says yes to everything EXCEPT the launcher's own cache path,
  // simulating a machine covered in brew/flatpak/pip lilbees. The launcher
  // must still fetch its own pinned binary.
  const env = { LILBEE_MCP_CACHE: "/cache" };
  const cachedPath = cachedBinaryPath(env, "v1", "a");
  const calls = [];
  const r = await resolveBinary({
    env,
    release: "v1",
    assetName: "a",
    deps: {
      existsSync: (p) => p !== cachedPath,
      download: async (o) => calls.push(o.dest),
    },
  });
  assert.equal(r.source, "download");
  assert.deepEqual(calls, [cachedPath]);
});
