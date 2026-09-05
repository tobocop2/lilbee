/**
 * Release-asset naming for the standalone lilbee binaries: plain CPU builds
 * per platform, CUDA/ROCm variants on Linux and Windows, and `-compat`
 * AVX-baseline builds for older x86-64 CPUs.
 */

const VARIANTS = new Set(["", "cu121", "cu124", "cu125", "rocm", "compat", "compat-cu124", "compat-rocm"]);

const HOST_BY_TARGET = {
  "macos-arm64": { platform: "darwin", arch: "arm64" },
  "macos-x86_64": { platform: "darwin", arch: "x64" },
  "linux-x86_64": { platform: "linux", arch: "x64" },
  "windows-x86_64": { platform: "win32", arch: "x64" },
};

const ASSET_NAME = /^lilbee(-compat)?-(macos-arm64|macos-x86_64|linux-x86_64|windows-x86_64)(?:-(cu121|cu124|cu125|rocm))?(\.exe)?$/;

/**
 * Return the release asset name for a host, or throw with a clear message.
 *
 * @param {string} platform - process.platform value ("darwin" | "linux" | "win32")
 * @param {string} arch - process.arch value ("arm64" | "x64")
 * @param {string} variant - "" or "default" (plain build) | "cu121" | "cu124" | "cu125" | "rocm" | "compat" | "compat-cu124" | "compat-rocm"
 */
export function assetNameFor(platform, arch, variant = "") {
  if (variant === "default") variant = "";
  if (!VARIANTS.has(variant)) {
    throw new Error(
      `Unknown LILBEE_VARIANT "${variant}". Valid: default, cu121, cu124, cu125, rocm, compat, compat-cu124, compat-rocm, or unset.`
    );
  }
  const compat = variant === "compat" || variant.startsWith("compat-");
  const gpuPart = compat ? variant.replace(/^compat-?/, "") : variant;
  const gpu = gpuPart === "" ? "" : `-${gpuPart}`;
  const prefix = compat ? "lilbee-compat" : "lilbee";

  if (platform === "darwin") {
    if (arch === "arm64") {
      if (variant !== "") {
        throw new Error("macOS arm64 ships a single build; unset LILBEE_VARIANT.");
      }
      return "lilbee-macos-arm64";
    }
    if (arch === "x64") {
      if (gpu) throw new Error("macOS has no GPU-variant builds; unset LILBEE_VARIANT.");
      return `${prefix}-macos-x86_64`;
    }
  }
  if (platform === "linux" && arch === "x64") {
    return `${prefix}-linux-x86_64${gpu}`;
  }
  if (platform === "win32" && arch === "x64") {
    if (compat && gpu) throw new Error("Windows compat build has no GPU variants.");
    return `${prefix}-windows-x86_64${gpu}.exe`;
  }
  throw new Error(
    `No standalone lilbee build for ${platform}/${arch}. ` +
      "Install lilbee yourself (pip/uv/brew) and point LILBEE_BIN at the binary."
  );
}

/** The host and variant an asset name encodes, or null for an asset that is not a lilbee binary. */
export function parseAssetName(name) {
  const m = ASSET_NAME.exec(name);
  if (!m) return null;
  const [, compat, target, gpu] = m;
  const { platform, arch } = HOST_BY_TARGET[target];
  const variant = compat ? (gpu ? `compat-${gpu}` : "compat") : gpu || "default";
  try {
    if (assetNameFor(platform, arch, variant) !== name) return null;
  } catch {
    return null;
  }
  return { platform, arch, variant };
}
