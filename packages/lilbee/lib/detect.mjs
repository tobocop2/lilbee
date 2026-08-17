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
 * - AMD: ROCm needs /dev/kfd plus a ROCm userland (rocminfo or /opt/rocm).
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
  if (platform === "linux" && !cuda) {
    rocm = exists("/dev/kfd") && (exists("/opt/rocm") || tryExec("rocminfo", []) !== null);
    if (rocm) log("lilbee: detected a ROCm GPU — using the rocm build (override with LILBEE_VARIANT).");
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
