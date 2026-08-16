/**
 * Release-asset naming for the standalone lilbee binaries.
 *
 * Mirrors the asset matrix the release pipeline publishes (see the GitHub
 * releases): plain CPU builds per platform, CUDA/ROCm variants on Linux and
 * Windows, and `-compat` AVX-baseline builds for older x86-64 CPUs.
 */

const VARIANTS = new Set(["", "cu121", "cu124", "cu125", "rocm", "compat"]);

/**
 * Return the release asset name for a host, or throw with a clear message.
 *
 * @param {string} platform - process.platform value ("darwin" | "linux" | "win32")
 * @param {string} arch - process.arch value ("arm64" | "x64")
 * @param {string} variant - "" (default CPU), "cu121" | "cu124" | "cu125" | "rocm" | "compat"
 */
export function assetNameFor(platform, arch, variant = "") {
  if (!VARIANTS.has(variant)) {
    throw new Error(
      `Unknown LILBEE_VARIANT "${variant}". Valid: cu121, cu124, cu125, rocm, compat, or unset.`
    );
  }
  const compat = variant === "compat";
  const gpu = compat || variant === "" ? "" : `-${variant}`;
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
      "Install lilbee yourself (pip/uv/brew) and it will be picked up from PATH, " +
      "or point LILBEE_BIN at a binary."
  );
}
