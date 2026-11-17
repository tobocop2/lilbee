import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

import { detectHost } from "../lib/detect.mjs";
import { releaseByTag } from "../lib/releases.mjs";

// The machine-case table that install.sh and install.ps1 are tested against too.
const TABLE_DIR = fileURLToPath(new URL("../../../tests/installer/", import.meta.url));
const TAG = "v0.6.90b692";
const KFD_NODES = "/sys/class/kfd/kfd/topology/nodes";
const BINARY_ASSET = /^lilbee(-compat)?-(macos|linux|windows)-[a-z0-9_]+(-(cu12[0-9]|rocm))?(\.exe)?$/;

const read = (name) => fs.readFileSync(`${TABLE_DIR}${name}`, "utf8");

function readCases() {
  const [header, ...rows] = read("cases.tsv").split("\n").filter(Boolean).map((line) => line.split("\t"));
  return rows.map((cells) => Object.fromEntries(header.map((name, i) => [name, cells[i] ?? ""])));
}

const assetNames = read("SHA256SUMS").split("\n").filter((line) => line && !line.startsWith("#")).map((line) => line.split(/\s+/)[1]);
const gfxManifest = read("rocm.gfx.txt");

const release = {
  tag_name: TAG,
  draft: false,
  prerelease: false,
  assets: assetNames.map((name) => ({ name, size: 1, digest: `sha256:${"0".repeat(64)}`, browser_download_url: `https://example.invalid/${name}` })),
};

const fakeFetch = async (url) => {
  const body = url.endsWith(".gfx.txt") ? gfxManifest : JSON.stringify(release);
  return { ok: true, status: 200, json: async () => JSON.parse(body), text: async () => body };
};

/** gfx name -> KFD gfx_target_version (major*10000 + minor*100 + step; minor and step are hex digits). */
function kfdVersionOf(gfx) {
  const m = /^gfx(\d+)([0-9a-f])([0-9a-f])$/.exec(gfx);
  return Number(m[1]) * 10000 + parseInt(m[2], 16) * 100 + parseInt(m[3], 16);
}

function ioFor(c) {
  const notFound = (cmd) => Object.assign(new Error(`spawn ${cmd} ENOENT`), { code: "ENOENT" });
  const gfx = c.gfx === "-" ? [] : c.gfx.split(",");
  const files = [];
  if (c.kfd === "yes") files.push("/dev/kfd", "/sys/module/amdgpu");
  if (c.rocm_userland === "yes") files.push("/opt/rocm");
  const nodes = Object.fromEntries(gfx.map((g, i) => [String(i + 1), kfdVersionOf(g)]));
  return {
    execFile: async (cmd) => {
      if (cmd === "nvidia-smi" || cmd.endsWith("nvidia-smi.exe")) {
        if (c.cuda === "-") throw notFound(cmd);
        return `| NVIDIA-SMI 560.35   Driver Version: 560.35   CUDA Version: ${c.cuda} |\n`;
      }
      if (cmd === "sysctl") return c.avx2 === "yes" ? "1" : "0";
      if (cmd === "powershell") return c.avx2 === "yes" ? "True" : "False";
      throw notFound(cmd);
    },
    existsSync: (p) => files.includes(p),
    readFileSync: (p) => {
      if (p === "/proc/cpuinfo") return `processor : 0\nflags\t\t: fpu sse4_2 avx${c.avx2 === "yes" ? " avx2" : ""}\n`;
      const m = /^\/sys\/class\/kfd\/kfd\/topology\/nodes\/(\w+)\/properties$/.exec(p);
      if (m && m[1] in nodes) return `simd_count 64\ngfx_target_version ${nodes[m[1]]}\n`;
      throw new Error("no file");
    },
    readdirSync: (p) => {
      if (p === KFD_NODES && gfx.length) return Object.keys(nodes);
      throw new Error("no dir");
    },
  };
}

async function pickFor(c) {
  const env = c.variant ? { LILBEE_VARIANT: c.variant } : {};
  try {
    const host = await detectHost(env, () => {}, ioFor(c), c.os, c.arch);
    return (await releaseByTag(TAG, { host, env, fetch: fakeFetch })).assetName;
  } catch {
    return "ERROR";
  }
}

test("every machine case gets the build the case table names", async () => {
  const cases = readCases();
  assert.ok(cases.length > 100, `the case table has ${cases.length} rows`);
  const wrong = [];
  for (const c of cases) {
    const got = await pickFor(c);
    if (got !== c.expected) wrong.push(`${c.id}: expected ${c.expected}, got ${got}`);
  }
  assert.deepEqual(wrong, []);
});

test("every build the case table expects is on the release", () => {
  const expected = new Set(readCases().map((c) => c.expected));
  const missing = [...expected].filter((name) => name !== "ERROR" && !assetNames.includes(name));
  assert.deepEqual(missing, []);
});

test("every binary build of the release is the answer for some machine case", () => {
  const expected = new Set(readCases().map((c) => c.expected));
  const binaries = assetNames.filter((name) => BINARY_ASSET.test(name));
  assert.ok(binaries.length > 10, `the release lists ${binaries.length} binaries`);
  assert.deepEqual(binaries.filter((name) => !expected.has(name)), []);
});
