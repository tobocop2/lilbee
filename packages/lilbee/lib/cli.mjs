/**
 * Launcher entry logic shared by both bins.
 *
 * `lilbee <argv>` routes: `prepare` (download only), `mcp` (stdio MCP server,
 * or the remote bridge when LILBEE_URL is set), everything else passes through
 * to the real binary verbatim. All launcher output goes to stderr; stdout
 * belongs to the command (for `mcp`, it is the MCP wire).
 */

import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { assetNameFor } from "./assets.mjs";
import { detectVariant } from "./detect.mjs";
import { download } from "./download.mjs";
import { exitCodeForSignal, HELP, mcpExec, passthroughExec, remoteExec, routeArgv, selectMode } from "./plan.mjs";
import { resolveBinary } from "./resolve.mjs";

const log = (msg) => console.error(msg);

function pkgMeta() {
  const pkgPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "package.json");
  return JSON.parse(fs.readFileSync(pkgPath, "utf8"));
}

function pinnedRelease() {
  const pkg = pkgMeta();
  return { release: pkg.lilbee.release, repo: pkg.lilbee.repo };
}

function whichAllSync(name) {
  try {
    const tool = process.platform === "win32" ? "where.exe" : "which";
    const args = process.platform === "win32" ? [name] : ["-a", name];
    const out = execFileSync(tool, args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    return out.split(/\r?\n/).filter(Boolean);
  } catch {
    return [];
  }
}

// Belt and braces against self-spawn: if a launcher ever execs another
// launcher (a resolution bug, or an exotic PATH), the child sees the
// sentinel and stops with a clear message instead of recursing forever.
const LAUNCHER_SENTINEL = "LILBEE_LAUNCHER_ACTIVE";

export function assertNotRecursing(env = process.env) {
  if (env[LAUNCHER_SENTINEL] === "1") {
    log(
      "lilbee: the launcher resolved another copy of itself instead of a real lilbee binary. " +
        "Point LILBEE_BIN at your lilbee install, or remove the npm shim from the front of PATH."
    );
    process.exit(1);
  }
}

/** Fail before a multi-hundred-MB download the binary could never run. */
export function assertGlibcFloor(env = process.env) {
  if (process.platform !== "linux" || env.LILBEE_SKIP_GLIBC_CHECK === "1") return;
  const floor = pkgMeta().lilbee?.glibcFloor;
  if (!floor) return;
  let version = "";
  try {
    version = execFileSync("getconf", ["GNU_LIBC_VERSION"], { encoding: "utf8" }).trim().split(" ").pop() || "";
  } catch {
    return; // musl or no getconf: let the download proceed, the exec will speak for itself
  }
  const num = (v) => v.split(".").map(Number);
  const [fa, fb] = num(floor);
  const [va, vb] = num(version);
  if (Number.isFinite(va) && (va < fa || (va === fa && vb < fb))) {
    log(
      `lilbee: this build needs glibc >= ${floor} (Ubuntu 22.04+/Debian 12+); this system has ${version}. ` +
        "Use the flatpak or 'pip install lilbee' instead, or set LILBEE_SKIP_GLIBC_CHECK=1 to try anyway."
    );
    process.exit(1);
  }
}

function spawnAndForward({ cmd, args }) {
  const child = spawn(cmd, args, {
    stdio: "inherit",
    // argv0 keeps the binary's own help reading `lilbee`, not the cache
    // filename; the sentinel stops accidental launcher-in-launcher chains.
    argv0: "lilbee",
    env: { ...process.env, [LAUNCHER_SENTINEL]: "1" },
  });
  child.on("exit", (code, signal) => {
    // Re-raising a trapped signal would loop through our own forwarder, so
    // exit with the conventional 128+signum code instead.
    if (signal) process.exit(exitCodeForSignal(signal));
    else process.exit(code ?? 0);
  });
  child.on("error", (err) => {
    log(`lilbee: failed to start ${cmd}: ${err.message}`);
    process.exit(1);
  });
  for (const sig of ["SIGINT", "SIGTERM"]) {
    process.on(sig, () => child.kill(sig));
  }
}

async function resolveLocalBinary(env) {
  const { release, repo } = pinnedRelease();
  const effectiveRelease = env.LILBEE_RELEASE || release;
  // Explicit LILBEE_VARIANT wins; otherwise detect the host's GPU and CPU
  // baseline so the bootstrap grabs the right build automatically, the way
  // brew or flatpak would.
  const variant =
    env.LILBEE_VARIANT !== undefined && env.LILBEE_VARIANT !== ""
      ? env.LILBEE_VARIANT
      : detectVariant(process.platform, process.arch, { execFileSync, existsSync: fs.existsSync, readFileSync: fs.readFileSync }, console.error);
  const assetName = assetNameFor(process.platform, process.arch, variant);
  const resolved = await resolveBinary({
    env: { LILBEE_REPO: repo, ...env },
    release: effectiveRelease,
    assetName,
    deps: {
      existsSync: fs.existsSync,
      whichAllSync,
      realpathSync: fs.realpathSync,
      selfPath: fileURLToPath(import.meta.url),
      download: async (o) => {
        assertGlibcFloor();
        return download({ ...o, log });
      },
    },
  });
  log(`lilbee: using binary from ${resolved.source} (${resolved.path})`);
  return resolved;
}

export async function run(argv) {
  assertNotRecursing();
  const env = process.env;
  if (argv[0] === "--launcher-help") {
    log(HELP);
    return;
  }
  const route = routeArgv(argv);

  if (route.kind === "prepare") {
    const resolved = await resolveLocalBinary(env);
    log(`lilbee: ready (${resolved.path}).`);
    return;
  }

  if (route.kind === "mcp" && selectMode(env) === "remote") {
    if (!env.LILBEE_TOKEN) {
      log(
        "lilbee: LILBEE_URL is set but LILBEE_TOKEN is not — connecting without " +
          "auth. lilbee servers normally require the session token from server.json."
      );
    }
    log(`lilbee: bridging MCP to ${env.LILBEE_URL}`);
    spawnAndForward(remoteExec(env));
    return;
  }

  const resolved = await resolveLocalBinary(env);
  if (route.kind === "mcp") {
    spawnAndForward(mcpExec(env, resolved.path, route.args));
  } else {
    spawnAndForward(passthroughExec(resolved.path, route.argv));
  }
}

export function runAndReport(argv) {
  run(argv).catch((err) => {
    log(`lilbee: ${err.message}`);
    process.exit(1);
  });
}
