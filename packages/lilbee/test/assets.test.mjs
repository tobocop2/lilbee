import { test } from "node:test";
import assert from "node:assert/strict";
import { assetNameFor } from "../lib/assets.mjs";

test("maps every published platform build", () => {
  assert.equal(assetNameFor("darwin", "arm64"), "lilbee-macos-arm64");
  assert.equal(assetNameFor("darwin", "x64"), "lilbee-macos-x86_64");
  assert.equal(assetNameFor("linux", "x64"), "lilbee-linux-x86_64");
  assert.equal(assetNameFor("win32", "x64"), "lilbee-windows-x86_64.exe");
});

test("maps GPU variants on linux and windows", () => {
  assert.equal(assetNameFor("linux", "x64", "cu125"), "lilbee-linux-x86_64-cu125");
  assert.equal(assetNameFor("linux", "x64", "cu121"), "lilbee-linux-x86_64-cu121");
  assert.equal(assetNameFor("linux", "x64", "rocm"), "lilbee-linux-x86_64-rocm");
  assert.equal(assetNameFor("win32", "x64", "cu124"), "lilbee-windows-x86_64-cu124.exe");
});

test("maps compat builds", () => {
  assert.equal(assetNameFor("darwin", "x64", "compat"), "lilbee-compat-macos-x86_64");
  assert.equal(assetNameFor("linux", "x64", "compat"), "lilbee-compat-linux-x86_64");
  assert.equal(assetNameFor("win32", "x64", "compat"), "lilbee-compat-windows-x86_64.exe");
});

test("rejects impossible combinations with clear messages", () => {
  assert.throws(() => assetNameFor("darwin", "arm64", "cu124"), /single build/);
  assert.throws(() => assetNameFor("darwin", "x64", "rocm"), /no GPU-variant/);
  assert.throws(() => assetNameFor("linux", "arm64"), /No standalone lilbee build/);
  assert.throws(() => assetNameFor("linux", "x64", "cu999"), /Unknown LILBEE_VARIANT/);
});
