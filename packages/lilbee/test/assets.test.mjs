import { test } from "node:test";
import assert from "node:assert/strict";
import { assetNameFor } from "../lib/assets.mjs";

test("maps every published platform build", () => {
  assert.equal(assetNameFor("darwin", "arm64"), "lilbee-macos-arm64");
  assert.equal(assetNameFor("darwin", "x64"), "lilbee-macos-x86_64");
  assert.equal(assetNameFor("linux", "x64"), "lilbee-linux-x86_64");
  assert.equal(assetNameFor("win32", "x64"), "lilbee-windows-x86_64.exe");
});

test("'default' selects the plain build even where detection would pick a GPU one", () => {
  assert.equal(assetNameFor("linux", "x64", "default"), "lilbee-linux-x86_64");
  assert.equal(assetNameFor("win32", "x64", "default"), "lilbee-windows-x86_64.exe");
  assert.equal(assetNameFor("darwin", "arm64", "default"), "lilbee-macos-arm64");
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

import { parseAssetName } from "../lib/assets.mjs";

/** Every asset of the v0.6.90b432 release: the binaries and the files published beside them. */
const RELEASE_ASSETS = [
  "lilbee-0.6.90b432-py3-none-any.whl",
  "lilbee-compat-linux-x86_64",
  "lilbee-compat-linux-x86_64-cu124",
  "lilbee-compat-linux-x86_64-rocm",
  "lilbee-compat-linux-x86_64-rocm.gfx.txt",
  "lilbee-compat-linux-x86_64.snap",
  "lilbee-compat-macos-x86_64",
  "lilbee-compat-windows-x86_64.exe",
  "lilbee-compat.flatpakref",
  "lilbee-cuda.flatpakref",
  "lilbee-linux-x86_64",
  "lilbee-linux-x86_64-cu121",
  "lilbee-linux-x86_64-cu124",
  "lilbee-linux-x86_64-cu125",
  "lilbee-linux-x86_64-cu125.snap",
  "lilbee-linux-x86_64-rocm",
  "lilbee-linux-x86_64-rocm.gfx.txt",
  "lilbee-linux-x86_64-rocm.snap",
  "lilbee-linux-x86_64.snap",
  "lilbee-macos-arm64",
  "lilbee-macos-x86_64",
  "lilbee-rocm.flatpakref",
  "lilbee-windows-x86_64-cu124.exe",
  "lilbee-windows-x86_64-cu125.exe",
  "lilbee-windows-x86_64.exe",
  "lilbee.flatpakref",
  "lilbee_engine-0.6.90b432-1.cpu-py3-none-macosx_11_0_arm64.whl",
  "lilbee_engine-0.6.90b432-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
  "openapi.json",
];

const BINARIES = {
  "lilbee-compat-linux-x86_64": ["linux", "x64", "compat"],
  "lilbee-compat-linux-x86_64-cu124": ["linux", "x64", "compat-cu124"],
  "lilbee-compat-linux-x86_64-rocm": ["linux", "x64", "compat-rocm"],
  "lilbee-compat-macos-x86_64": ["darwin", "x64", "compat"],
  "lilbee-compat-windows-x86_64.exe": ["win32", "x64", "compat"],
  "lilbee-linux-x86_64": ["linux", "x64", "default"],
  "lilbee-linux-x86_64-cu121": ["linux", "x64", "cu121"],
  "lilbee-linux-x86_64-cu124": ["linux", "x64", "cu124"],
  "lilbee-linux-x86_64-cu125": ["linux", "x64", "cu125"],
  "lilbee-linux-x86_64-rocm": ["linux", "x64", "rocm"],
  "lilbee-macos-arm64": ["darwin", "arm64", "default"],
  "lilbee-macos-x86_64": ["darwin", "x64", "default"],
  "lilbee-windows-x86_64-cu124.exe": ["win32", "x64", "cu124"],
  "lilbee-windows-x86_64-cu125.exe": ["win32", "x64", "cu125"],
  "lilbee-windows-x86_64.exe": ["win32", "x64", "default"],
};

test("parseAssetName recognises every binary of the release and nothing else", () => {
  for (const name of RELEASE_ASSETS) {
    const expected = BINARIES[name];
    const parsed = parseAssetName(name);
    if (expected) {
      const [platform, arch, variant] = expected;
      assert.deepEqual(parsed, { platform, arch, variant }, name);
    } else {
      assert.equal(parsed, null, name);
    }
  }
});

test("parseAssetName inverts assetNameFor and reports the plain build as 'default'", () => {
  for (const [name, [platform, arch, variant]] of Object.entries(BINARIES)) {
    assert.equal(assetNameFor(platform, arch, variant), name);
    assert.equal(parseAssetName(name).variant, variant);
  }
  assert.equal(parseAssetName("lilbee-linux-x86_64").variant, "default");
});

test("parseAssetName rejects names outside the published matrix", () => {
  assert.equal(parseAssetName("lilbee-windows-x86_64"), null);
  assert.equal(parseAssetName("lilbee-linux-x86_64.exe"), null);
  assert.equal(parseAssetName("lilbee-macos-arm64-cu124"), null);
  assert.equal(parseAssetName("lilbee-compat-windows-x86_64-cu124.exe"), null);
  assert.equal(parseAssetName("lilbee-compat-linux-x86_64-cu121"), null);
  assert.equal(parseAssetName("lilbee-linux-x86_64.download.123"), null);
  assert.equal(parseAssetName(""), null);
});
