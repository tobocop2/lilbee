/**
 * Hardware detection for the bootstrap download: the build a host can run,
 * read from the NVIDIA driver, the amdgpu KFD topology, and the CPU flags.
 * Detection fails toward the default build, which runs everywhere.
 */

import { execFile } from "node:child_process";
import fs from "node:fs";
import { promisify } from "node:util";

import { assetNameFor } from "./assets.mjs";

const KFD_NODES = "/sys/class/kfd/kfd/topology/nodes";
const PROBE_TIMEOUT_MS = 5000;
const VARIANT_HINT = "(override with LILBEE_VARIANT)";

const execFileAsync = promisify(execFile);

/** The probes detection runs on the real host: `execFile` resolves to stdout. */
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
  const v = major * 100 + minor;
  if (v >= 1205) return "cu125";
  if (v >= 1204) return "cu124";
  if (v >= 1201) return "cu121";
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

/** The gfx names of every GPU node the KFD topology reports, unique and sorted; unreadable nodes are skipped. */
function kfdGfxTargets(io) {
  let nodes;
  try {
    nodes = io.readdirSync(KFD_NODES);
  } catch {
    return [];
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

/**
 * Detect the build for a host: `{ variant, amdGfxTargets }`. `io` carries the
 * probes ({ execFile, existsSync, readFileSync, readdirSync }) so tests can
 * simulate any host; `log` receives one line per decision.
 */
export async function detectVariant(platform, arch, io = nodeIo, log = () => {}) {
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
  const host = (variant, amdGfxTargets = []) => ({ variant, amdGfxTargets });

  if (platform === "darwin") {
    if (arch !== "x64") return host("default");
    const avx2 = await tryExec("sysctl", ["-n", "hw.optional.avx2_0"]);
    if (avx2 !== null && avx2.trim() === "0") {
      log(`lilbee: this CPU has no AVX2; using the compat build ${VARIANT_HINT}.`);
      return host("compat");
    }
    return host("default");
  }

  let cuda = null;
  const smi = await tryExec("nvidia-smi", []);
  if (smi) {
    const v = parseCudaVersion(smi);
    cuda = v ? cudaVariantFor(v.major, v.minor) : "cu121";
    if (cuda) log(`lilbee: detected an NVIDIA driver (CUDA ${v ? `${v.major}.${v.minor}` : "unknown"}); using the ${cuda} build ${VARIANT_HINT}.`);
  } else if (platform === "linux" && (exists("/dev/nvidia0") || exists("/proc/driver/nvidia/version"))) {
    cuda = "cu121";
    log(`lilbee: an NVIDIA device is present but nvidia-smi does not run; using the cu121 build ${VARIANT_HINT}.`);
  }

  let rocm = false;
  let amdGfxTargets = [];
  if (platform === "linux" && !cuda && exists("/dev/kfd")) {
    amdGfxTargets = kfdGfxTargets(io);
    if (amdGfxTargets.length) {
      rocm = true;
      log(`lilbee: detected an AMD GPU (${amdGfxTargets.join(", ")}); using the rocm build when the release supports it ${VARIANT_HINT}.`);
    } else {
      rocm = exists("/opt/rocm") || (await tryExec("rocminfo", [])) !== null;
      if (rocm) log(`lilbee: detected a ROCm userland; using the rocm build ${VARIANT_HINT}.`);
    }
  }

  let noAvx2 = false;
  if (platform === "linux") {
    try {
      const cpuinfo = io.readFileSync("/proc/cpuinfo", "utf8");
      const flagsLine = /^flags\s*:\s*(.+)$/m.exec(cpuinfo);
      noAvx2 = !!flagsLine && !/\bavx2\b/.test(flagsLine[1]);
    } catch {
      noAvx2 = false;
    }
    if (noAvx2) log(`lilbee: this CPU has no AVX2; using the compat build family ${VARIANT_HINT}.`);
  }

  if (platform === "linux") {
    if (noAvx2) {
      // compat ships cu124 and rocm flavors only
      if (cuda) return host("compat-cu124", amdGfxTargets);
      if (rocm) return host("compat-rocm", amdGfxTargets);
      return host("compat", amdGfxTargets);
    }
    if (cuda) return host(cuda);
    if (rocm) return host("rocm", amdGfxTargets);
    return host("default");
  }

  if (platform === "win32") {
    // Windows ships cu124 and cu125 only
    if (cuda === "cu121") {
      log(`lilbee: this NVIDIA driver predates CUDA 12.4; using the default Windows build ${VARIANT_HINT}.`);
      return host("default");
    }
    return host(cuda ?? "default");
  }

  return host("default");
}

/**
 * The host to resolve releases for: `{ platform, arch, variant, amdGfxTargets }`.
 * A non-empty LILBEE_VARIANT in `env` replaces the detected variant; an unknown
 * value throws. Detection failures fall back to the default build.
 */
export async function detectHost(env = process.env, log = () => {}, io = nodeIo, platform = process.platform, arch = process.arch) {
  const forced = env.LILBEE_VARIANT;
  if (forced) {
    assetNameFor(platform, arch, forced);
    return { platform, arch, variant: forced, amdGfxTargets: [] };
  }
  const { variant, amdGfxTargets } = await detectVariant(platform, arch, io, log);
  return { platform, arch, variant, amdGfxTargets };
}
