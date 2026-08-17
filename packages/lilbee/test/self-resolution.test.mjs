import test from "node:test";
import assert from "node:assert/strict";
import { resolveBinary } from "../lib/resolve.mjs";

const baseEnv = {};

function deps(overrides = {}) {
  return {
    existsSync: () => false,
    whichAllSync: () => [],
    realpathSync: (p) => p,
    selfPath: "/pkg/lib/cli.mjs",
    download: async ({ dest }) => dest,
    ...overrides,
  };
}

test("an npm bin shim on PATH is never accepted as the binary", async () => {
  let downloaded = false;
  const r = await resolveBinary({
    env: baseEnv,
    release: "v1",
    assetName: "lilbee-macos-arm64",
    deps: deps({
      whichAllSync: () => ["/tmp/_npx/abc/node_modules/.bin/lilbee"],
      download: async () => {
        downloaded = true;
      },
    }),
  });
  assert.equal(r.source, "download");
  assert.equal(downloaded, true);
});

test("a shim is rejected by realpath even when PATH shows a clean location", async () => {
  const r = await resolveBinary({
    env: baseEnv,
    release: "v1",
    assetName: "a",
    deps: deps({
      whichAllSync: () => ["/usr/local/bin/lilbee"],
      realpathSync: () => "/Users/u/.npm/_npx/x/node_modules/lilbee/bin/lilbee.mjs",
    }),
  });
  assert.notEqual(r.source, "path");
});

test("the launcher's own entry file is rejected", async () => {
  const r = await resolveBinary({
    env: baseEnv,
    release: "v1",
    assetName: "a",
    deps: deps({
      whichAllSync: () => ["/somewhere/lilbee"],
      realpathSync: () => "/pkg/lib/cli.mjs",
    }),
  });
  assert.notEqual(r.source, "path");
});

test("a real binary on PATH still wins", async () => {
  const r = await resolveBinary({
    env: baseEnv,
    release: "v1",
    assetName: "a",
    deps: deps({ whichAllSync: () => ["/opt/homebrew/bin/lilbee"] }),
  });
  assert.deepEqual(r, { path: "/opt/homebrew/bin/lilbee", source: "path" });
});

test("a real install behind the shim wins over downloading", async () => {
  const r = await resolveBinary({
    env: {},
    release: "v1",
    assetName: "a",
    deps: {
      existsSync: () => false,
      realpathSync: (p) => p,
      selfPath: "/pkg/lib/cli.mjs",
      whichAllSync: () => ["/tmp/_npx/x/node_modules/.bin/lilbee", "/opt/homebrew/bin/lilbee"],
      download: async () => {
        throw new Error("must not download");
      },
    },
  });
  assert.deepEqual(r, { path: "/opt/homebrew/bin/lilbee", source: "path" });
});
