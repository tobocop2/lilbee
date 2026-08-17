import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import { releaseAsset } from "../lib/download.mjs";

const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });

function stubFetch(payload, ok = true, status = 200) {
  globalThis.fetch = async () => ({ ok, status, json: async () => payload });
}

test("releaseAsset finds the asset and parses the sha256 digest", async () => {
  stubFetch({ assets: [{ name: "lilbee-macos-arm64", size: 7, digest: "sha256:abc123", browser_download_url: "https://x/y" }] });
  const a = await releaseAsset("o/r", "v1", "lilbee-macos-arm64");
  assert.deepEqual(a, { url: "https://x/y", size: 7, digest: "abc123" });
});

test("releaseAsset without a digest returns null digest", async () => {
  stubFetch({ assets: [{ name: "a", size: 1, browser_download_url: "u" }] });
  assert.equal((await releaseAsset("o/r", "v1", "a")).digest, null);
});

test("releaseAsset names the available assets when the requested one is missing", async () => {
  stubFetch({ assets: [{ name: "other", size: 1, browser_download_url: "u" }] });
  await assert.rejects(releaseAsset("o/r", "v1", "nope"), /no asset "nope".*other/);
});

test("releaseAsset surfaces HTTP failures with the status", async () => {
  stubFetch({}, false, 404);
  await assert.rejects(releaseAsset("o/r", "vX", "a"), /HTTP 404/);
});
