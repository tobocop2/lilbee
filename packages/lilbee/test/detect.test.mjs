import test from "node:test";
import assert from "node:assert/strict";
import { cudaVariantFor, detectVariant, parseCudaVersion } from "../lib/detect.mjs";
import { assetNameFor } from "../lib/assets.mjs";

const io = ({ smi = null, files = [], cpuflags = "fpu avx avx2", sysctlAvx2 = "1", rocminfo = false } = {}) => ({
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
    throw new Error("no file");
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

test("ROCm needs /dev/kfd plus a userland; bare /dev/kfd is not enough", () => {
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/kfd", "/opt/rocm"] })), "rocm");
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/kfd"], rocminfo: true })), "rocm");
  assert.equal(detectVariant("linux", "x64", io({ files: ["/dev/kfd"] })), "");
});

test("no-AVX2 CPUs get the compat family, composed with the GPU", () => {
  assert.equal(detectVariant("linux", "x64", io({ cpuflags: "fpu sse4_2" })), "compat");
  assert.equal(detectVariant("linux", "x64", io({ cpuflags: "fpu sse4_2", smi: "CUDA Version: 12.6" })), "compat-cu124");
  assert.equal(detectVariant("linux", "x64", io({ cpuflags: "fpu sse4_2", files: ["/dev/kfd", "/opt/rocm"] })), "compat-rocm");
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
