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

// --- progress line rendering (TTY bar) ---

import { progressLine } from "../lib/download.mjs";

const MB = 1048576;

test("progressLine draws an empty, half, and full bar at a fixed width", () => {
  const at = (done) => progressLine({ done, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(at(0), /│─{10}│ +0%/);
  assert.match(at(50 * MB), /│━{5}─{5}│ +50%/);
  assert.match(at(100 * MB), /│━{10}│ +100%/);
});

test("progressLine marks a partial cell with a half-step edge", () => {
  const line = progressLine({ done: 55 * MB, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(line, /│━{5}╸─{4}│ +55%/);
});

test("progressLine shows sizes, rate, and eta", () => {
  const line = progressLine({ done: 523 * MB, total: 1246 * MB, bytesPerSec: 87 * MB, barWidth: 10, color: false });
  assert.match(line, /523\/1246MB/);
  assert.match(line, /87\.0MB\/s/);
  assert.match(line, /eta 0:0[89]/); // (1246-523)/87 ≈ 8.3s
});

test("progressLine clamps overshoot and hides rate/eta when unknown", () => {
  const over = progressLine({ done: 120 * MB, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(over, /100%/);
  assert.doesNotMatch(over, /eta/);
  assert.doesNotMatch(over, /MB\/s/);
});

test("progressLine without a total reports plain progress", () => {
  const line = progressLine({ done: 42 * MB, total: 0, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(line, /42MB/);
  assert.doesNotMatch(line, /%/);
});

test("progressLine colors the bar only when asked", () => {
  const colored = progressLine({ done: 50 * MB, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: true });
  const plain = progressLine({ done: 50 * MB, total: 100 * MB, bytesPerSec: 0, barWidth: 10, color: false });
  assert.match(colored, /\x1b\[/);
  assert.doesNotMatch(plain, /\x1b\[/);
});
