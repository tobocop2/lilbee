import test from "node:test";
import assert from "node:assert/strict";
import { cudaVariantFor, detectVariant, gfxNameFor, parseCudaVersion } from "../lib/detect.mjs";
import { assetNameFor } from "../lib/assets.mjs";

const KFD_NODES = "/sys/class/kfd/kfd/topology/nodes";

const io = ({ smi = null, files = [], cpuflags = "fpu avx avx2", sysctlAvx2 = "1", rocminfo = false, kfdNodes = null } = {}) => ({
  execFileSync: (cmd) => {
    if (cmd === "nvidia-smi") {
      if (smi === null) throw new Error("not found");
      return smi;
    }
    if (cmd === "rocminfo") {
      if (!rocminfo) throw new Error("not found");
      return "ROCk module is loaded";
    }
    if (cmd === "sysctl") return sysctlAvx2;
    throw new Error("not found");
  },
  existsSync: (p) => files.includes(p),
  readFileSync: (p) => {
    if (p === "/proc/cpuinfo") return `processor : 0\nflags\t\t: ${cpuflags}\n`;
    const node = kfdNodes && /^\/sys\/class\/kfd\/kfd\/topology\/nodes\/(\w+)\/properties$/.exec(p)?.[1];
    if (node && node in kfdNodes) {
      if (kfdNodes[node] === null) throw new Error("Operation not permitted");
      return `simd_count 0\ngfx_target_version ${kfdNodes[node]}\n`;
    }
    throw new Error("no file");
  },
  readdirSync: (p) => {
    if (p === KFD_NODES && kfdNodes) return Object.keys(kfdNodes);
    throw new Error("no dir");
  },
});

test("driver CUDA version maps to the newest runnable build", () => {
  assert.equal(cudaVariantFor(12, 6), "cu125");
  assert.equal(cudaVariantFor(12, 4), "cu124");
  assert.equal(cudaVariantFor(12, 1), "cu121");
  assert.equal(cudaVariantFor(11, 8), "");
  assert.deepEqual(parseCudaVersion("| NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.4 |"), { major: 12, minor: 4 });
});

test("linux NVIDIA hosts get the matching CUDA build", () => {
  assert.equal(detectVariant("linux", "x64", io({ smi: "CUDA Version: 12.6" })), "cu125");
  assert.equal(detectVariant("linux", "x64", io({ smi: "CUDA Version: 12.4" })), "cu124");
});

test("visible NVIDIA device without runnable nvidia-smi falls back to cu121", () => {
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/nvidia0"] })), "cu121");
});

test("gfx_target_version maps to gfx names", () => {
  assert.equal(gfxNameFor(90402), "gfx942"); // MI300X
  assert.equal(gfxNameFor(90010), "gfx90a"); // MI200
  assert.equal(gfxNameFor(100300), "gfx1030"); // RDNA2
  assert.equal(gfxNameFor(110002), "gfx1102"); // RDNA3
  assert.equal(gfxNameFor(0), null);
});

test("a supported AMD GPU in the KFD topology picks rocm without any host userland", () => {
  const mi300x = { 0: 0, 1: 90402 }; // CPU node + GPU node, as RunPod MI300X pods report
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/kfd"], kfdNodes: mi300x })), "rocm");
});

test("an unsupported AMD GPU falls back to the default build", () => {
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/kfd"], kfdNodes: { 0: 0, 1: 90006 } })), ""); // MI50
});

test("unreadable topology nodes are skipped, readable GPU node still wins", () => {
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/kfd"], kfdNodes: { 0: 0, 1: 90402, 2: null } })), "rocm");
});

test("without KFD topology, ROCm needs /dev/kfd plus a userland; bare /dev/kfd is not enough", () => {
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/kfd", "/opt/rocm"] })), "rocm");
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/kfd"], rocminfo: true })), "rocm");
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/kfd"] })), "");
});

test("no-AVX2 CPUs get the compat family, composed with the GPU", () => {
  assert.equal(detectVariant("linux", "x64", io({ cpuflags: "fpu sse4_2" })), "compat");
  assert.equal(detectVariant("linux", "x64", io({ cpuflags: "fpu sse4_2", smi: "CUDA Version: 12.6" })), "compat-cu124");
  assert.equal(detectVariant("linux", "x64", io({ cpuflags: "fpu sse4_2", files: ["/dev/kfd", "/opt/rocm"] })), "compat-rocm");
  assert.equal(detectVariant("linux", "x64", io({ cpuflags: "fpu sse4_2", files: ["/dev/kfd"], kfdNodes: { 1: 90402 } })), "compat-rocm");
});

test("plain linux hosts get the universal build", () => {
  assert.equal(detectVariant("linux", "x64", io()), "");
});

test("windows maps CUDA to shipped builds only", () => {
  assert.equal(detectVariant("win32", "x64", io({ smi: "CUDA Version: 12.5" })), "cu125");
  assert.equal(detectVariant("win32", "x64", io({ smi: "CUDA Version: 12.4" })), "cu124");
  assert.equal(detectVariant("win32", "x64", io({ smi: "CUDA Version: 12.2" })), "");
  assert.equal(detectVariant("win32", "x64", io()), "");
});

test("intel macs without AVX2 get the compat build", () => {
  assert.equal(detectVariant("darwin", "x64", io({ sysctlAvx2: "0" })), "compat");
  assert.equal(detectVariant("darwin", "x64", io()), "");
  assert.equal(detectVariant("darwin", "arm64", io({ sysctlAvx2: "0" })), "");
});

test("detected variants all map to real release assets", () => {
  assert.equal(assetNameFor("linux", "x64", "compat-cu124"), "lilbee-compat-linux-x86_64-cu124");
  assert.equal(assetNameFor("linux", "x64", "compat-rocm"), "lilbee-compat-linux-x86_64-rocm");
  assert.equal(assetNameFor("linux", "x64", "cu125"), "lilbee-linux-x86_64-cu125");
  assert.equal(assetNameFor("win32", "x64", "cu125"), "lilbee-windows-x86_64-cu125.exe");
  assert.equal(assetNameFor("darwin", "x64", "compat"), "lilbee-compat-macos-x86_64");
});
