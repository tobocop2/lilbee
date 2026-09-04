import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { cacheDir, cachedBinaryPath, installedBinary, pruneOtherReleases } from "../lib/cache.mjs";
import { compareReleaseTags } from "../lib/releases.mjs";

const LINUX = { platform: "linux", arch: "x64", variant: "cu125", amdGfxTargets: [] };

function cacheWith(files) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "lilbee-cache-"));
  for (const file of files) {
    fs.mkdirSync(path.dirname(path.join(dir, file)), { recursive: true });
    fs.writeFileSync(path.join(dir, file), "");
  }
  return dir;
}

test("cacheDir honors LILBEE_MCP_CACHE and platform conventions", () => {
  assert.equal(cacheDir({ LILBEE_MCP_CACHE: "/tmp/c" }), "/tmp/c");
  assert.ok(cacheDir({}, "darwin").endsWith(path.join("Library", "Caches", "lilbee-npm")));
  assert.ok(cacheDir({ XDG_CACHE_HOME: "/xdg" }, "linux").startsWith("/xdg"));
  assert.ok(cacheDir({ LOCALAPPDATA: "C:\\LA" }, "win32").includes("lilbee-npm"));
});

test("cachedBinaryPath is <cacheDir>/<release>/<assetName>", () => {
  assert.equal(cachedBinaryPath("/c", "v1", "lilbee-macos-arm64"), path.join("/c", "v1", "lilbee-macos-arm64"));
});

test("release tags order numerically, newest first", () => {
  const tags = ["v0.6.90b423", "v0.6.90b425", "v0.7.0", "v0.6.91"];
  assert.deepEqual(tags.sort(compareReleaseTags), ["v0.7.0", "v0.6.91", "v0.6.90b425", "v0.6.90b423"]);
});

test("installedBinary picks the newest release and, within it, the host's variant", () => {
  const dir = cacheWith([
    "v0.6.90b423/lilbee-linux-x86_64-cu125",
    "v0.6.90b425/lilbee-linux-x86_64",
    "v0.6.90b425/lilbee-linux-x86_64-cu125",
    "v0.6.90b425/lilbee-linux-x86_64-cu125.download.4242",
  ]);
  assert.deepEqual(installedBinary({ cacheDir: dir, host: LINUX }), {
    path: path.join(dir, "v0.6.90b425", "lilbee-linux-x86_64-cu125"),
    release: "v0.6.90b425",
    assetName: "lilbee-linux-x86_64-cu125",
    variant: "cu125",
  });
  fs.rmSync(dir, { recursive: true, force: true });
});

test("a cached build of another variant still counts as installed after a hardware change", () => {
  const dir = cacheWith(["v0.6.90b425/lilbee-linux-x86_64", "v0.6.90b425/lilbee-macos-arm64"]);
  const found = installedBinary({ cacheDir: dir, host: LINUX });
  assert.equal(found.assetName, "lilbee-linux-x86_64");
  assert.equal(found.variant, "default");
  fs.rmSync(dir, { recursive: true, force: true });
});

test("installedBinary ignores other platforms, temp files, and a missing cache", () => {
  const dir = cacheWith(["v1/lilbee-macos-arm64", "v1/lilbee-linux-x86_64.download.7", "v2/notes.txt"]);
  assert.equal(installedBinary({ cacheDir: dir, host: LINUX }), null);
  assert.equal(installedBinary({ cacheDir: path.join(dir, "missing"), host: LINUX }), null);
  fs.rmSync(dir, { recursive: true, force: true });
});

test("pruneOtherReleases keeps one release dir and removes the rest", () => {
  const dir = cacheWith(["v1/lilbee-macos-arm64", "v2/lilbee-macos-arm64", "v3/lilbee-macos-arm64"]);
  pruneOtherReleases(dir, "v2");
  assert.deepEqual(fs.readdirSync(dir), ["v2"]);
  pruneOtherReleases(path.join(dir, "missing"), "v2");
  fs.rmSync(dir, { recursive: true, force: true });
});
