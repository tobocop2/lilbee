/**
 * Hardware detection for the bootstrap download: the build a host can run,
 * read from the NVIDIA driver, the amdgpu KFD topology, and the CPU flags.
 * Every probe records how it ended in a detection report; the build is then
 * chosen from the report. Detection fails toward the default build, which
 * runs everywhere.
 */

import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { promisify } from "node:util";

import { assetNameFor } from "./assets.mjs";

const KFD_DEVICE = "/dev/kfd";
const KFD_NODES = "/sys/class/kfd/kfd/topology/nodes";
const AMDGPU_MODULE = "/sys/module/amdgpu";
const ROCM_USERLAND = "/opt/rocm";
const NVIDIA_DEVICE = "/dev/nvidia0";
const NVIDIA_DRIVER_VERSION = "/proc/driver/nvidia/version";
const PROBE_TIMEOUT_MS = 10000;
const NVIDIA_SMI_TIMED_OUT = `nvidia-smi did not answer within ${PROBE_TIMEOUT_MS / 1000} s`;
/** kernel32 IsProcessorFeaturePresent: PF_AVX2_INSTRUCTIONS_AVAILABLE. */
const PF_AVX2_INSTRUCTIONS_AVAILABLE = 40;
const AVX2_PROBE_SCRIPT = [
  "Add-Type -Namespace Lilbee -Name Cpu -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern bool IsProcessorFeaturePresent(int feature);'",
  `[Lilbee.Cpu]::IsProcessorFeaturePresent(${PF_AVX2_INSTRUCTIONS_AVAILABLE})`,
].join("\n");
const AVX2_PROBE_ARGS = ["-NoProfile", "-NonInteractive", "-EncodedCommand", Buffer.from(AVX2_PROBE_SCRIPT, "utf16le").toString("base64")];

const COMPAT_BUILD_LOG = "lilbee: this CPU has no AVX2 — using the -compat build (override with LILBEE_VARIANT).";
const COMPAT_FAMILY_LOG = "lilbee: this CPU has no AVX2 — using the -compat build family (override with LILBEE_VARIANT).";

const execFileAsync = promisify(execFile);

/** The probes detection runs on the real host: `execFile` resolves to stdout and rejects with the raw error. */
const nodeIo = {
  execFile: async (cmd, args) => {
    const { stdout } = await execFileAsync(cmd, args, {
      encoding: "utf8",
      timeout: PROBE_TIMEOUT_MS,
      windowsHide: true,
    });
    return stdout;
  },
  existsSync: fs.existsSync,
  readFileSync: fs.readFileSync,
  readdirSync: fs.readdirSync,
};

/** Map the driver-reported CUDA version to the newest runnable build, or null when it runs none. */
export function cudaVariantFor(major, minor) {
  return cudaVariantForCeiling(major * 100 + minor);
}

function cudaVariantForCeiling(ceiling) {
  if (ceiling >= 1205) return "cu125";
  if (ceiling >= 1204) return "cu124";
  if (ceiling >= 1201) return "cu121";
  return null;
}

/** Parse "CUDA Version: 12.6" out of nvidia-smi's banner. */
export function parseCudaVersion(smiOutput) {
  const m = /CUDA Version:\s*(\d+)\.(\d+)/.exec(smiOutput || "");
  return m ? { major: Number(m[1]), minor: Number(m[2]) } : null;
}

/** Map a KFD gfx_target_version (major*10000 + minor*100 + step) to its gfx name, e.g. 90402 -> gfx942. */
export function gfxNameFor(version) {
  if (!Number.isInteger(version) || version <= 0) return null;
  const major = Math.floor(version / 10000);
  const minor = Math.floor(version / 100) % 100;
  const step = version % 100;
  return `gfx${major}${minor.toString(16)}${step.toString(16)}`;
}

/** Windows installers put `nvidia-smi` here but leave it off PATH: DCH drivers in System32, older layouts in NVSMI. */
function windowsNvidiaSmiPaths(env) {
  const systemRoot = env.SystemRoot || "C:\\Windows";
  const programFiles = env.ProgramFiles || "C:\\Program Files";
  return [
    path.win32.join(systemRoot, "System32", "nvidia-smi.exe"),
    path.win32.join(programFiles, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
  ];
}

/** Where to look for `nvidia-smi`: PATH first, then the Windows install locations. */
export function nvidiaSmiCandidates(platform, env) {
  return platform === "win32" ? ["nvidia-smi", ...windowsNvidiaSmiPaths(env)] : ["nvidia-smi"];
}

/** One line for the report: execFile's message carries the command's stderr on later lines. */
function probeFailureText(err) {
  if (err && err.killed) return NVIDIA_SMI_TIMED_OUT;
  const text = err instanceof Error ? err.message : String(err);
  return text.replace(/\s+/g, " ").trim();
}

/** Run `nvidia-smi` at each candidate: its stdout, or the first failure and whether no candidate existed at all. */
async function runNvidiaSmi(platform, io, env) {
  let error = "";
  let notFound = true;
  for (const command of nvidiaSmiCandidates(platform, env)) {
    try {
      return { stdout: await io.execFile(command, []), error: null, notFound: false };
    } catch (err) {
      if (!error) error = probeFailureText(err);
      if (!err || err.code !== "ENOENT") notFound = false;
    }
  }
  return { stdout: null, error, notFound };
}

async function probeNvidia(platform, io, env, exists) {
  if (platform === "darwin") return { status: "skipped" };
  const { stdout, error, notFound } = await runNvidiaSmi(platform, io, env);
  if (stdout === null) {
    if (notFound && platform === "linux" && exists(NVIDIA_DRIVER_VERSION)) return { status: "sandboxed" };
    return { status: "missing", error };
  }
  const v = parseCudaVersion(stdout);
  if (!v) return { status: "unreadable" };
  return { status: "detected", cudaCeiling: v.major * 100 + v.minor };
}

/** The gfx names of every GPU node the KFD topology reports, unique and sorted; null when the topology is unreadable. */
function kfdGfxTargets(io) {
  let nodes;
  try {
    nodes = io.readdirSync(KFD_NODES);
  } catch {
    return null;
  }
  const targets = new Set();
  for (const node of nodes) {
    try {
      const props = io.readFileSync(`${KFD_NODES}/${node}/properties`, "utf8");
      const m = /^gfx_target_version\s+(\d+)$/m.exec(props);
      const name = m && gfxNameFor(Number(m[1]));
      if (name) targets.add(name);
    } catch {
      // node not visible from this container
    }
  }
  return [...targets].sort();
}

function probeAmd(platform, io, exists) {
  if (platform !== "linux") return { status: "skipped" };
  if (!exists(KFD_DEVICE)) return { status: exists(AMDGPU_MODULE) ? "sandboxed" : "missing" };
  const gfxTargets = kfdGfxTargets(io);
  if (gfxTargets === null) return { status: "unreadable" };
  if (gfxTargets.length === 0) return { status: "missing" };
  return { status: "detected", gfxTargets };
}

function probeCpuLinux(io) {
  let cpuinfo;
  try {
    cpuinfo = io.readFileSync("/proc/cpuinfo", "utf8");
  } catch {
    return { status: "unreadable" };
  }
  const flagsLine = /^flags\s*:\s*(.+)$/m.exec(cpuinfo);
  if (!flagsLine) return { status: "unreadable" };
  return { status: "detected", avx2: /\bavx2\b/.test(flagsLine[1]) };
}

/** A probe that prints one token for AVX2 present and one for absent; anything else is unreadable. */
async function probeCpuByCommand(tryExec, cmd, args, present, absent) {
  const output = await tryExec(cmd, args);
  const answer = output === null ? null : output.trim();
  if (answer === present) return { status: "detected", avx2: true };
  if (answer === absent) return { status: "detected", avx2: false };
  return { status: "unreadable" };
}

async function probeCpu(platform, arch, io, tryExec) {
  if (arch !== "x64") return { status: "skipped" };
  if (platform === "linux") return probeCpuLinux(io);
  if (platform === "darwin") return probeCpuByCommand(tryExec, "sysctl", ["-n", "hw.optional.avx2_0"], "1", "0");
  if (platform === "win32") return probeCpuByCommand(tryExec, "powershell", AVX2_PROBE_ARGS, "True", "False");
  return { status: "skipped" };
}

/** The report of a host whose build was forced: nothing was probed. */
function skippedDetection() {
  return { nvidia: { status: "skipped" }, amd: { status: "skipped" }, cpu: { status: "skipped" }, detectedAt: new Date().toISOString() };
}

/**
 * The CUDA build the NVIDIA probe calls for, or null; logs the CLI's line for the choice.
 * Only a driver that names its CUDA version gets a CUDA build: without one the default
 * build's Vulkan engine still uses the card, where a CUDA runtime that fails to
 * initialise would fall back to the CPU.
 */
function cudaBuild(nvidia, platform, exists, log) {
  if (nvidia.status === "detected") {
    const cuda = cudaVariantForCeiling(nvidia.cudaCeiling);
    const label = `${Math.floor(nvidia.cudaCeiling / 100)}.${nvidia.cudaCeiling % 100}`;
    if (cuda) log(`lilbee: detected NVIDIA driver (CUDA ${label}) — using the ${cuda} build (override with LILBEE_VARIANT).`);
    return cuda;
  }
  if (nvidia.status === "unreadable") {
    log("lilbee: nvidia-smi names no CUDA version — using the default build (override with LILBEE_VARIANT).");
  } else if (platform === "linux" && (exists(NVIDIA_DEVICE) || exists(NVIDIA_DRIVER_VERSION))) {
    log("lilbee: NVIDIA device present but nvidia-smi is not runnable — using the default build (override with LILBEE_VARIANT).");
  }
  return null;
}

/** Whether the ROCm build applies (Linux, no CUDA, a compute device plus named GPUs or a ROCm userland). */
async function rocmBuild(amd, platform, cuda, exists, tryExec, log) {
  if (platform !== "linux" || cuda || !exists(KFD_DEVICE)) return { rocm: false, amdGfxTargets: [] };
  if (amd.status === "detected") {
    log(`lilbee: detected an AMD GPU (${amd.gfxTargets.join(", ")}) — using the rocm build (override with LILBEE_VARIANT).`);
    return { rocm: true, amdGfxTargets: amd.gfxTargets };
  }
  const rocm = exists(ROCM_USERLAND) || (await tryExec("rocminfo", [])) !== null;
  if (rocm) log("lilbee: detected a ROCm userland — using the rocm build (override with LILBEE_VARIANT).");
  return { rocm, amdGfxTargets: [] };
}

function linuxVariant(cuda, rocm, noAvx2, log) {
  if (noAvx2) {
    log(COMPAT_FAMILY_LOG);
    // compat ships cu124 and rocm flavors only
    if (cuda) return "compat-cu124";
    return rocm ? "compat-rocm" : "compat";
  }
  if (cuda) return cuda;
  return rocm ? "rocm" : "default";
}

function windowsVariant(cuda, noAvx2, log) {
  // Windows ships cu124 and cu125, and a compat build with no GPU flavor
  if (noAvx2) {
    log(COMPAT_BUILD_LOG);
    return "compat";
  }
  if (cuda === "cu121") {
    log("lilbee: this NVIDIA driver predates CUDA 12.4 — using the universal Windows build (override with LILBEE_VARIANT).");
    return "default";
  }
  return cuda ?? "default";
}

/** Probe the host and return its detection report. */
async function probeHost(platform, arch, io, env, exists, tryExec) {
  const nvidia = await probeNvidia(platform, io, env, exists);
  const amd = probeAmd(platform, io, exists);
  const cpu = await probeCpu(platform, arch, io, tryExec);
  return { nvidia, amd, cpu, detectedAt: new Date().toISOString() };
}

/**
 * Detect the build for a host: `{ variant, amdGfxTargets, detection }`. `io`
 * carries the probes ({ execFile, existsSync, readFileSync, readdirSync }) so
 * tests can simulate any host; `log` receives one line per decision; `env`
 * supplies the Windows install locations of `nvidia-smi`.
 */
export async function detectVariant(platform, arch, io = nodeIo, log = () => {}, env = process.env) {
  const tryExec = async (cmd, args) => {
    try {
      return await io.execFile(cmd, args);
    } catch {
      return null;
    }
  };
  const exists = (p) => {
    try {
      return io.existsSync(p);
    } catch {
      return false;
    }
  };
  const detection = await probeHost(platform, arch, io, env, exists, tryExec);
  const noAvx2 = detection.cpu.status === "detected" && !detection.cpu.avx2;
  const host = (variant, amdGfxTargets = []) => ({ variant, amdGfxTargets, detection });

  if (platform === "darwin") {
    if (noAvx2) log(COMPAT_BUILD_LOG);
    return host(noAvx2 ? "compat" : "default");
  }

  const cuda = cudaBuild(detection.nvidia, platform, exists, log);
  const { rocm, amdGfxTargets } = await rocmBuild(detection.amd, platform, cuda, exists, tryExec, log);

  if (platform === "linux") return host(linuxVariant(cuda, rocm, noAvx2, log), amdGfxTargets);
  if (platform === "win32") return host(windowsVariant(cuda, noAvx2, log));
  return host("default");
}

/**
 * The host to resolve releases for: `{ platform, arch, variant, amdGfxTargets, detection }`.
 * A non-empty LILBEE_VARIANT in `env` replaces the detected variant and skips
 * every probe; an unknown value throws. Detection failures fall back to the default build.
 */
export async function detectHost(env = process.env, log = () => {}, io = nodeIo, platform = process.platform, arch = process.arch) {
  const forced = env.LILBEE_VARIANT;
  if (forced) {
    assetNameFor(platform, arch, forced);
    return { platform, arch, variant: forced, amdGfxTargets: [], detection: skippedDetection() };
  }
  const { variant, amdGfxTargets, detection } = await detectVariant(platform, arch, io, log, env);
  return { platform, arch, variant, amdGfxTargets, detection };
}
