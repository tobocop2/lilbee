import test from "node:test";
import assert from "node:assert/strict";
import { cudaVariantFor, detectHost, detectVariant, gfxNameFor, nvidiaSmiCandidates, parseCudaVersion } from "../lib/detect.mjs";
import { parseAssetName } from "../lib/assets.mjs";
import { assetNameFor } from "../lib/assets.mjs";

const KFD_NODES = "/sys/class/kfd/kfd/topology/nodes";

/**
 * A simulated host. `smi` is nvidia-smi's stdout, an Error to throw, a function of the
 * command tried, or null for "not installed"; `powershell` answers the Windows AVX2 probe
 * the same way. Every command run lands in `calls`.
 */
const io = ({ smi = null, files = [], cpuflags = "fpu avx avx2", cpuinfo = null, sysctlAvx2 = "1", powershell = null, rocminfo = false, kfdNodes = null, kfdUnreadable = false } = {}) => {
  const calls = [];
  const notFound = (cmd) => Object.assign(new Error(`spawn ${cmd} ENOENT`), { code: "ENOENT" });
  const answer = (cmd, response) => {
    if (response === null) throw notFound(cmd);
    if (response instanceof Error) throw response;
    return typeof response === "function" ? response(cmd) : response;
  };
  return {
    calls,
    execFile: async (cmd) => {
      calls.push(cmd);
      if (cmd === "nvidia-smi" || cmd.endsWith("nvidia-smi.exe")) return answer(cmd, smi);
      if (cmd === "rocminfo") {
        if (!rocminfo) throw notFound(cmd);
        return "ROCk module is loaded";
      }
      if (cmd === "sysctl") return answer(cmd, sysctlAvx2);
      if (cmd === "powershell") return answer(cmd, powershell);
      throw notFound(cmd);
    },
    existsSync: (p) => files.includes(p),
    readFileSync: (p) => {
      if (p === "/proc/cpuinfo") return cpuinfo ?? `processor : 0\nflags\t\t: ${cpuflags}\n`;
      const node = kfdNodes && /^\/sys\/class\/kfd\/kfd\/topology\/nodes\/(\w+)\/properties$/.exec(p)?.[1];
      if (node && node in kfdNodes) {
        if (kfdNodes[node] === null) throw new Error("Operation not permitted");
        return `simd_count 0\ngfx_target_version ${kfdNodes[node]}\n`;
      }
      throw new Error("no file");
    },
    readdirSync: (p) => {
      if (p === KFD_NODES && kfdNodes && !kfdUnreadable) return Object.keys(kfdNodes);
      throw new Error("no dir");
    },
  };
};

const variant = async (platform, arch, probes) => (await detectVariant(platform, arch, io(probes))).variant;
const picked = ({ variant, amdGfxTargets }) => ({ variant, amdGfxTargets });
const report = async (platform, arch, probes, env = {}) => (await detectVariant(platform, arch, io(probes), () => {}, env)).detection;

test("driver CUDA version maps to the newest runnable build", () => {
  assert.equal(cudaVariantFor(12, 6), "cu125");
  assert.equal(cudaVariantFor(12, 4), "cu124");
  assert.equal(cudaVariantFor(12, 1), "cu121");
  assert.equal(cudaVariantFor(11, 8), null);
  assert.deepEqual(parseCudaVersion("| NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.4 |"), { major: 12, minor: 4 });
});

test("linux NVIDIA hosts get the matching CUDA build", async () => {
  assert.equal(await variant("linux", "x64", { smi: "CUDA Version: 12.6" }), "cu125");
  assert.equal(await variant("linux", "x64", { smi: "CUDA Version: 12.4" }), "cu124");
});

test("visible NVIDIA device without runnable nvidia-smi falls back to cu121", async () => {
  assert.equal(await variant("linux", "x64", { files: ["/dev/nvidia0"] }), "cu121");
});

test("gfx_target_version maps to gfx names", () => {
  assert.equal(gfxNameFor(90402), "gfx942"); // MI300X
  assert.equal(gfxNameFor(90010), "gfx90a"); // MI200
  assert.equal(gfxNameFor(100300), "gfx1030"); // RDNA2
  assert.equal(gfxNameFor(110002), "gfx1102"); // RDNA3
  assert.equal(gfxNameFor(0), null);
});

test("an AMD GPU in the KFD topology proposes rocm and reports its gfx targets", async () => {
  const mi300x = { 0: 0, 1: 90402 }; // CPU node + GPU node, as RunPod MI300X pods report
  const host = await detectVariant("linux", "x64", io({ files: ["/dev/kfd"], kfdNodes: mi300x }));
  assert.deepEqual(picked(host), { variant: "rocm", amdGfxTargets: ["gfx942"] });
});

test("gfx targets are unique and sorted, whatever the release later decides about them", async () => {
  const host = await detectVariant("linux", "x64", io({ files: ["/dev/kfd"], kfdNodes: { 0: 0, 1: 90006, 2: 90402, 3: 90006 } }));
  assert.deepEqual(picked(host), { variant: "rocm", amdGfxTargets: ["gfx906", "gfx942"] });
});

test("unreadable topology nodes are skipped, readable GPU node still wins", async () => {
  assert.equal(await variant("linux", "x64", { files: ["/dev/kfd"], kfdNodes: { 0: 0, 1: 90402, 2: null } }), "rocm");
});

test("without KFD topology, ROCm needs /dev/kfd plus a userland; bare /dev/kfd is not enough", async () => {
  assert.deepEqual(picked(await detectVariant("linux", "x64", io({ files: ["/dev/kfd", "/opt/rocm"] }))), { variant: "rocm", amdGfxTargets: [] });
  assert.equal(await variant("linux", "x64", { files: ["/dev/kfd"], rocminfo: true }), "rocm");
  assert.equal(await variant("linux", "x64", { files: ["/dev/kfd"] }), "default");
});

test("no-AVX2 CPUs get the compat family, composed with the GPU", async () => {
  assert.equal(await variant("linux", "x64", { cpuflags: "fpu sse4_2" }), "compat");
  assert.equal(await variant("linux", "x64", { cpuflags: "fpu sse4_2", smi: "CUDA Version: 12.6" }), "compat-cu124");
  assert.equal(await variant("linux", "x64", { cpuflags: "fpu sse4_2", files: ["/dev/kfd", "/opt/rocm"] }), "compat-rocm");
  assert.equal(await variant("linux", "x64", { cpuflags: "fpu sse4_2", files: ["/dev/kfd"], kfdNodes: { 1: 90402 } }), "compat-rocm");
});

test("plain linux hosts get the universal build", async () => {
  assert.equal(await variant("linux", "x64", {}), "default");
});

test("windows maps CUDA to shipped builds only", async () => {
  assert.equal(await variant("win32", "x64", { smi: "CUDA Version: 12.5" }), "cu125");
  assert.equal(await variant("win32", "x64", { smi: "CUDA Version: 12.4" }), "cu124");
  assert.equal(await variant("win32", "x64", { smi: "CUDA Version: 12.2" }), "default");
  assert.equal(await variant("win32", "x64", {}), "default");
});

test("intel macs without AVX2 get the compat build", async () => {
  assert.equal(await variant("darwin", "x64", { sysctlAvx2: "0" }), "compat");
  assert.equal(await variant("darwin", "x64", {}), "default");
  assert.equal(await variant("darwin", "arm64", { sysctlAvx2: "0" }), "default");
});

test("detected variants all map to real release assets", () => {
  assert.equal(assetNameFor("linux", "x64", "compat-cu124"), "lilbee-compat-linux-x86_64-cu124");
  assert.equal(assetNameFor("linux", "x64", "compat-rocm"), "lilbee-compat-linux-x86_64-rocm");
  assert.equal(assetNameFor("linux", "x64", "cu125"), "lilbee-linux-x86_64-cu125");
  assert.equal(assetNameFor("win32", "x64", "cu125"), "lilbee-windows-x86_64-cu125.exe");
  assert.equal(assetNameFor("darwin", "x64", "compat"), "lilbee-compat-macos-x86_64");
});

test("detectHost reports the running platform and never logs on its own", async () => {
  const host = await detectHost({});
  assert.equal(host.platform, process.platform);
  assert.equal(host.arch, process.arch);
  assert.ok(typeof host.variant === "string" && host.variant !== "");
  assert.ok(Array.isArray(host.amdGfxTargets));
  assert.equal(typeof host.detection.detectedAt, "string");
  for (const probe of ["nvidia", "amd", "cpu"]) assert.equal(typeof host.detection[probe].status, "string");
});

test("detectHost honors LILBEE_VARIANT and rejects an unknown value", async () => {
  const lines = [];
  const host = await detectHost({ LILBEE_VARIANT: "cu124" }, (m) => lines.push(m), io(), "linux", "x64");
  assert.equal(host.variant, "cu124");
  assert.deepEqual(lines, []);
  assert.equal((await detectHost({ LILBEE_VARIANT: "default" }, () => {}, io({ smi: "CUDA Version: 12.6" }), "linux", "x64")).variant, "default");
  assert.equal((await detectHost({ LILBEE_VARIANT: "" }, () => {}, io({ smi: "CUDA Version: 12.6" }), "linux", "x64")).variant, "cu125");
  await assert.rejects(detectHost({ LILBEE_VARIANT: "cu999" }, () => {}, io(), "linux", "x64"), /Unknown LILBEE_VARIANT/);
});

test("detectHost logs what it picked through the given logger", async () => {
  const lines = [];
  await detectHost({}, (m) => lines.push(m), io({ smi: "CUDA Version: 12.6" }), "linux", "x64");
  assert.deepEqual(lines, ["lilbee: detected NVIDIA driver (CUDA 12.6) — using the cu125 build (override with LILBEE_VARIANT)."]);
});

const KFD = ["/dev/kfd"];
const NVIDIA_MODULE = "/proc/driver/nvidia/version";

test("the NVIDIA probe records why it ended: skipped, missing, sandboxed, unreadable, detected", async () => {
  assert.deepEqual((await report("darwin", "x64", { smi: "CUDA Version: 12.6" })).nvidia, { status: "skipped" });
  assert.deepEqual((await report("linux", "x64", {})).nvidia, { status: "missing", error: "spawn nvidia-smi ENOENT" });
  assert.deepEqual((await report("linux", "x64", { files: [NVIDIA_MODULE] })).nvidia, { status: "sandboxed" });
  assert.deepEqual((await report("linux", "x64", { smi: "NVIDIA-SMI has failed" })).nvidia, { status: "unreadable" });
  assert.deepEqual((await report("linux", "x64", { smi: "CUDA Version: 12.4" })).nvidia, { status: "detected", cudaCeiling: 1204 });
  assert.deepEqual((await report("win32", "x64", { smi: "CUDA Version: 12.6" })).nvidia, { status: "detected", cudaCeiling: 1206 });
});

test("a failing nvidia-smi is missing, not sandboxed, and its first error is one line", async () => {
  const crash = new Error("Command failed: nvidia-smi\nNVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n  Make sure it is installed.\n");
  const probe = (await report("linux", "x64", { smi: crash, files: [NVIDIA_MODULE] })).nvidia;
  assert.deepEqual(probe, { status: "missing", error: "Command failed: nvidia-smi NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure it is installed." });
});

test("a timed-out nvidia-smi records the timeout", async () => {
  const hung = Object.assign(new Error("spawn nvidia-smi ETIMEDOUT"), { killed: true, signal: "SIGTERM" });
  assert.deepEqual((await report("linux", "x64", { smi: hung })).nvidia, { status: "missing", error: "nvidia-smi did not answer within 10 s" });
  assert.equal(await variant("linux", "x64", { smi: hung, files: ["/dev/nvidia0"] }), "cu121");
});

test("Windows looks for nvidia-smi on PATH, then in System32, then in the NVSMI directory", async () => {
  const env = { SystemRoot: "D:\\Win", ProgramFiles: "D:\\Programs" };
  assert.deepEqual(nvidiaSmiCandidates("win32", env), ["nvidia-smi", "D:\\Win\\System32\\nvidia-smi.exe", "D:\\Programs\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"]);
  assert.deepEqual(nvidiaSmiCandidates("win32", {}), ["nvidia-smi", "C:\\Windows\\System32\\nvidia-smi.exe", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"]);
  assert.deepEqual(nvidiaSmiCandidates("linux", env), ["nvidia-smi"]);

  const offPath = io({ smi: (cmd) => (cmd.startsWith("D:\\Programs") ? "CUDA Version: 12.5" : (() => { throw Object.assign(new Error(`spawn ${cmd} ENOENT`), { code: "ENOENT" }); })()) });
  const host = await detectVariant("win32", "x64", offPath, () => {}, env);
  assert.equal(host.variant, "cu125");
  assert.deepEqual(host.detection.nvidia, { status: "detected", cudaCeiling: 1205 });
  assert.deepEqual(offPath.calls.filter((c) => /nvidia-smi/.test(c)), ["nvidia-smi", "D:\\Win\\System32\\nvidia-smi.exe", "D:\\Programs\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"]);

  const none = io();
  await detectVariant("win32", "x64", none, () => {}, env);
  assert.equal(none.calls.filter((c) => /nvidia-smi/.test(c)).length, 3);
  const onLinux = io();
  await detectVariant("linux", "x64", onLinux, () => {}, env);
  assert.deepEqual(onLinux.calls.filter((c) => /nvidia-smi/.test(c)), ["nvidia-smi"]);
});

test("the AMD probe records why it ended: skipped, missing, sandboxed, unreadable, detected", async () => {
  assert.deepEqual((await report("win32", "x64", { files: KFD })).amd, { status: "skipped" });
  assert.deepEqual((await report("darwin", "x64", {})).amd, { status: "skipped" });
  assert.deepEqual((await report("linux", "x64", {})).amd, { status: "missing" });
  assert.deepEqual((await report("linux", "x64", { files: ["/sys/module/amdgpu"] })).amd, { status: "sandboxed" });
  assert.deepEqual((await report("linux", "x64", { files: KFD })).amd, { status: "unreadable" });
  assert.deepEqual((await report("linux", "x64", { files: KFD, kfdNodes: { 0: 0 }, kfdUnreadable: true })).amd, { status: "unreadable" });
  assert.deepEqual((await report("linux", "x64", { files: KFD, kfdNodes: { 0: 0 } })).amd, { status: "missing" });
  assert.deepEqual((await report("linux", "x64", { files: KFD, kfdNodes: { 0: 0, 1: 90402, 2: 90006, 3: 90402 } })).amd, { status: "detected", gfxTargets: ["gfx906", "gfx942"] });
});

test("the AMD probe runs beside a CUDA driver, though CUDA still picks the build", async () => {
  const host = await detectVariant("linux", "x64", io({ smi: "CUDA Version: 12.6", files: KFD, kfdNodes: { 1: 90402 } }));
  assert.equal(host.variant, "cu125");
  assert.deepEqual(host.amdGfxTargets, []);
  assert.deepEqual(host.detection.amd, { status: "detected", gfxTargets: ["gfx942"] });
});

test("the CPU probe records why it ended: skipped, unreadable, detected", async () => {
  assert.deepEqual((await report("darwin", "arm64", {})).cpu, { status: "skipped" });
  assert.deepEqual((await report("linux", "arm64", {})).cpu, { status: "skipped" });
  assert.deepEqual((await report("linux", "x64", {})).cpu, { status: "detected", avx2: true });
  assert.deepEqual((await report("linux", "x64", { cpuflags: "fpu sse4_2" })).cpu, { status: "detected", avx2: false });
  assert.deepEqual((await report("linux", "x64", { cpuinfo: "processor : 0\n" })).cpu, { status: "unreadable" });
  assert.deepEqual((await report("darwin", "x64", {})).cpu, { status: "detected", avx2: true });
  assert.deepEqual((await report("darwin", "x64", { sysctlAvx2: "0" })).cpu, { status: "detected", avx2: false });
  assert.deepEqual((await report("darwin", "x64", { sysctlAvx2: null })).cpu, { status: "unreadable" });
  assert.deepEqual((await report("darwin", "x64", { sysctlAvx2: "sysctl: unknown oid" })).cpu, { status: "unreadable" });
});

test("Windows asks kernel32 for AVX2 through PowerShell and takes the compat build without it", async () => {
  assert.deepEqual((await report("win32", "x64", { powershell: "True\r\n" })).cpu, { status: "detected", avx2: true });
  assert.deepEqual((await report("win32", "x64", { powershell: "False\r\n" })).cpu, { status: "detected", avx2: false });
  assert.deepEqual((await report("win32", "x64", { powershell: null })).cpu, { status: "unreadable" });
  assert.deepEqual((await report("win32", "x64", { powershell: new Error("Add-Type : Cannot add type.") })).cpu, { status: "unreadable" });
  assert.deepEqual((await report("linux", "x64", {})).cpu, { status: "detected", avx2: true });

  assert.equal(await variant("win32", "x64", { powershell: "False" }), "compat");
  assert.equal(await variant("win32", "x64", { powershell: "False", smi: "CUDA Version: 12.6" }), "compat");
  assert.equal(await variant("win32", "x64", { powershell: "False", smi: "CUDA Version: 12.2" }), "compat");
  assert.equal(await variant("win32", "x64", { powershell: "True", smi: "CUDA Version: 12.6" }), "cu125");
  assert.equal(await variant("win32", "x64", { powershell: null, smi: "CUDA Version: 12.6" }), "cu125");
  assert.equal(assetNameFor("win32", "x64", "compat"), "lilbee-compat-windows-x86_64.exe");
  assert.deepEqual(parseAssetName("lilbee-compat-windows-x86_64.exe"), { platform: "win32", arch: "x64", variant: "compat" });
  assert.throws(() => assetNameFor("win32", "x64", "compat-cu124"), /Windows compat build has no GPU variants/);
});

test("the Windows AVX2 probe is a PowerShell one-liner calling IsProcessorFeaturePresent(40)", async () => {
  const host = io({ powershell: "True" });
  await detectVariant("win32", "x64", host);
  assert.deepEqual(host.calls, ["nvidia-smi", "C:\\Windows\\System32\\nvidia-smi.exe", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe", "powershell"]);
  const probe = io({ powershell: "True" });
  probe.execFile = async (cmd, args) => {
    if (cmd !== "powershell") throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
    assert.deepEqual(args.slice(0, 3), ["-NoProfile", "-NonInteractive", "-EncodedCommand"]);
    const script = Buffer.from(args[3], "base64").toString("utf16le");
    assert.match(script, /Add-Type .*kernel32\.dll.*IsProcessorFeaturePresent/);
    assert.match(script, /IsProcessorFeaturePresent\(40\)$/);
    return "True";
  };
  assert.deepEqual((await detectVariant("win32", "x64", probe)).detection.cpu, { status: "detected", avx2: true });
  assert.equal(io().calls.length, 0);
});

test("the report is stamped with an ISO 8601 time", async () => {
  const before = Date.now();
  const { detectedAt } = await report("linux", "x64", {});
  assert.match(detectedAt, /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  assert.ok(Date.parse(detectedAt) >= before);
});

test("detectHost carries the report, and a forced LILBEE_VARIANT probes nothing", async () => {
  const probes = io({ smi: "CUDA Version: 12.6", cpuflags: "fpu sse4_2" });
  const host = await detectHost({}, () => {}, probes, "linux", "x64");
  assert.equal(host.variant, "compat-cu124");
  assert.deepEqual(host.detection.nvidia, { status: "detected", cudaCeiling: 1206 });
  assert.deepEqual(host.detection.cpu, { status: "detected", avx2: false });

  const forced = io({ smi: "CUDA Version: 12.6" });
  const pinned = await detectHost({ LILBEE_VARIANT: "rocm" }, () => {}, forced, "linux", "x64");
  assert.deepEqual(forced.calls, []);
  assert.equal(pinned.variant, "rocm");
  assert.deepEqual({ ...pinned.detection, detectedAt: "" }, { nvidia: { status: "skipped" }, amd: { status: "skipped" }, cpu: { status: "skipped" }, detectedAt: "" });
});

test("the log lines the CLI prints are unchanged by the report", async () => {
  const lines = async (platform, probes) => {
    const out = [];
    await detectVariant(platform, "x64", io(probes), (m) => out.push(m));
    return out;
  };
  assert.deepEqual(await lines("linux", { smi: "CUDA Version: 12.6", cpuflags: "fpu sse4_2" }), [
    "lilbee: detected NVIDIA driver (CUDA 12.6) — using the cu125 build (override with LILBEE_VARIANT).",
    "lilbee: this CPU has no AVX2 — using the -compat build family (override with LILBEE_VARIANT).",
  ]);
  assert.deepEqual(await lines("linux", { smi: "no version here" }), ["lilbee: detected NVIDIA driver (CUDA unknown) — using the cu121 build (override with LILBEE_VARIANT)."]);
  assert.deepEqual(await lines("linux", { files: [NVIDIA_MODULE] }), ["lilbee: NVIDIA device present but nvidia-smi is not runnable — using the cu121 build (override with LILBEE_VARIANT)."]);
  assert.deepEqual(await lines("linux", { files: KFD, kfdNodes: { 1: 90402 } }), ["lilbee: detected an AMD GPU (gfx942) — using the rocm build (override with LILBEE_VARIANT)."]);
  assert.deepEqual(await lines("linux", { files: [...KFD, "/opt/rocm"] }), ["lilbee: detected a ROCm userland — using the rocm build (override with LILBEE_VARIANT)."]);
  assert.deepEqual(await lines("linux", {}), []);
  assert.deepEqual(await lines("darwin", { sysctlAvx2: "0" }), ["lilbee: this CPU has no AVX2 — using the -compat build (override with LILBEE_VARIANT)."]);
  assert.deepEqual(await lines("win32", { smi: "CUDA Version: 12.2" }), [
    "lilbee: detected NVIDIA driver (CUDA 12.2) — using the cu121 build (override with LILBEE_VARIANT).",
    "lilbee: this NVIDIA driver predates CUDA 12.4 — using the universal Windows build (override with LILBEE_VARIANT).",
  ]);
  assert.deepEqual(await lines("win32", { powershell: "False" }), ["lilbee: this CPU has no AVX2 — using the -compat build (override with LILBEE_VARIANT)."]);
});
