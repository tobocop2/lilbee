/**
 * Hardware detection for the bootstrap download: pick the release variant
 * the way a package manager would, so `npx lilbee` lands the right build
 * with zero configuration. An explicit LILBEE_VARIANT always wins.
 *
 * Detection is best-effort and fails toward the universal default build:
 * a wrong "default" still runs everywhere, a wrong CUDA build does not.
 *
 * Probes (all cheap, no network):
 * - NVIDIA: `nvidia-smi` reports "CUDA Version: N.M" (the driver's max
 *   supported runtime); map it to the newest cuNNN asset it can run.
 *   A visible /dev/nvidia0 or /proc/driver/nvidia/version without a
 *   working nvidia-smi falls back to the oldest CUDA build.
 * - AMD: the rocm build bundles its own userspace, so a host only needs the
 *   amdgpu kernel driver: /dev/kfd plus a supported gfx target read from the
 *   kernel's KFD topology. A host ROCm userland (rocminfo or /opt/rocm) is
 *   the fallback signal when the topology is unreadable.
 * - CPU baseline: x86-64 hosts without AVX2 get the -compat build
 *   (Linux: /proc/cpuinfo flags; macOS x86_64: sysctl hw.optional.avx2_0).
 */

/** Map the driver-reported CUDA version to the newest runnable asset variant. */
export function cudaVariantFor(major, minor) {
  const v = major * 100 + minor;
  if (v >= 1205) return "cu125";
  if (v >= 1204) return "cu124";
  if (v >= 1201) return "cu121";
  return ""; // driver too old for any shipped CUDA build; universal build works
}

/** Parse "CUDA Version: 12.6" out of nvidia-smi's banner. */
export function parseCudaVersion(smiOutput) {
  const m = /CUDA Version:\s*(\d+)\.(\d+)/.exec(smiOutput || "");
  return m ? { major: Number(m[1]), minor: Number(m[2]) } : null;
}

/** The gfx targets the pinned rocm build ships kernels for (its published
 * <asset>-rocm.gfx.txt). Update alongside the release pin. */
const ROCM_GFX = new Set([
  "gfx908", "gfx90a", "gfx942", "gfx950",
  "gfx1030", "gfx1100", "gfx1101", "gfx1102",
  "gfx1150", "gfx1151", "gfx1200", "gfx1201",
]);

const KFD_NODES = "/sys/class/kfd/kfd/topology/nodes";

/** Map a KFD gfx_target_version (major*10000 + minor*100 + step) to its gfx
 * name, e.g. 90402 -> gfx942, 90010 -> gfx90a. CPU nodes report 0 -> null. */
export function gfxNameFor(version) {
  if (!Number.isInteger(version) || version <= 0) return null;
  const major = Math.floor(version / 10000);
  const minor = Math.floor(version / 100) % 100;
  const step = version % 100;
  return `gfx${major}${minor}${step.toString(16)}`;
}

/** The gfx names of every GPU node the kernel's KFD topology reports.
 * Unreadable nodes (a container may only expose its own GPU) are skipped. */
function kfdGfxTargets(io) {
  let nodes;
  try {
    nodes = io.readdirSync(KFD_NODES);
  } catch {
    return [];
  }
  const targets = [];
  for (const node of nodes) {
    try {
      const props = io.readFileSync(`${KFD_NODES}/${node}/properties`, "utf8");
      const m = /^gfx_target_version\s+(\d+)$/m.exec(props);
      const name = m && gfxNameFor(Number(m[1]));
      if (name) targets.push(name);
    } catch {
      // node not visible from this container
    }
  }
  return targets;
}

/**
 * Detect the variant for this host. `io` carries the probes so tests can
 * simulate any host: { execFileSync, existsSync, readFileSync }.
 */
export function detectVariant(platform, arch, io, log = () => {}) {
  const tryExec = (cmd, args) => {
    try {
      return io.execFileSync(cmd, args, { encoding: "utf8", timeout: 5000, stdio: ["ignore", "pipe", "ignore"] });
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

  if (platform === "darwin") {
    if (arch !== "x64") return ""; // arm64: single Metal build
    const avx2 = tryExec("sysctl", ["-n", "hw.optional.avx2_0"]);
    if (avx2 !== null && avx2.trim() === "0") {
      log("lilbee: this CPU has no AVX2 — using the -compat build (override with LILBEE_VARIANT).");
      return "compat";
    }
    return "";
  }

  // NVIDIA first: a machine with both an NVIDIA card and /dev/kfd is rare,
  // and the CUDA build is the better pick when it happens.
  let cuda = "";
  const smi = tryExec("nvidia-smi", []);
  if (smi) {
    const v = parseCudaVersion(smi);
    cuda = v ? cudaVariantFor(v.major, v.minor) : "cu121";
    if (cuda) log(`lilbee: detected NVIDIA driver (CUDA ${v ? `${v.major}.${v.minor}` : "unknown"}) — using the ${cuda} build (override with LILBEE_VARIANT).`);
  } else if (platform === "linux" && (exists("/dev/nvidia0") || exists("/proc/driver/nvidia/version"))) {
    cuda = "cu121";
    log("lilbee: NVIDIA device present but nvidia-smi is not runnable — using the cu121 build (override with LILBEE_VARIANT).");
  }

  let rocm = false;
  if (platform === "linux" && !cuda && exists("/dev/kfd")) {
    const gfx = io.readdirSync ? kfdGfxTargets(io) : [];
    if (gfx.length) {
      const supported = gfx.find((g) => ROCM_GFX.has(g));
      rocm = Boolean(supported);
      if (rocm) log(`lilbee: detected an AMD GPU (${supported}) — using the rocm build (override with LILBEE_VARIANT).`);
      else log(`lilbee: the rocm build does not support this AMD GPU (${gfx.join(", ")}) — using the default build (override with LILBEE_VARIANT).`);
    } else {
      rocm = exists("/opt/rocm") || tryExec("rocminfo", []) !== null;
      if (rocm) log("lilbee: detected a ROCm userland — using the rocm build (override with LILBEE_VARIANT).");
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
    if (noAvx2) log("lilbee: this CPU has no AVX2 — using the -compat build family (override with LILBEE_VARIANT).");
  }

  if (platform === "linux") {
    if (noAvx2) {
      // compat ships with cu124 and rocm flavors only.
      if (cuda) return "compat-cu124";
      if (rocm) return "compat-rocm";
      return "compat";
    }
    if (cuda) return cuda;
    if (rocm) return "rocm";
    return "";
  }

  if (platform === "win32") {
    // Windows ships cu124/cu125 only; older drivers get the universal build.
    if (cuda === "cu121") {
      log("lilbee: this NVIDIA driver predates CUDA 12.4 — using the universal Windows build (override with LILBEE_VARIANT).");
      return "";
    }
    return cuda;
  }

  return "";
}
