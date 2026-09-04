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
import path from "node:path";
import { fileURLToPath } from "node:url";

import { cacheDir, installRelease } from "./cache.mjs";
import { detectHost } from "./detect.mjs";
import { progressReporter } from "./download.mjs";
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

// Belt and braces against self-spawn: if LILBEE_BIN ever points at another
// launcher, the child sees the sentinel and stops with a clear message
// instead of recursing forever.
const LAUNCHER_SENTINEL = "LILBEE_LAUNCHER_ACTIVE";

export function assertNotRecursing(env = process.env) {
  if (env[LAUNCHER_SENTINEL] === "1") {
    log(
      "lilbee: the launcher started another copy of itself instead of a real lilbee binary. " +
        "Point LILBEE_BIN at a real lilbee install, or unset it to use the bundled download."
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

/**
 * Kill *child* and every descendant. npx runs the real work (mcp-remote) as a
 * grandchild, so signalling only the direct child leaves the bridge alive.
 * POSIX: the child was spawned detached, so it leads its own process group and
 * a negative-pid kill reaches the whole tree, grandchildren included, even
 * after the leader has exited. Windows has no process groups; taskkill /T
 * walks the tree instead (forced — Windows has no cross-process SIGTERM).
 * Only valid for children spawned by spawnAndForward in tieToStdin mode.
 */
function killTree(child, signal) {
  if (process.platform === "win32") {
    try {
      execFileSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], { stdio: "ignore" });
    } catch {} // the tree is already gone
    return;
  }
  try {
    process.kill(-child.pid, signal);
  } catch {} // the group is already gone
}

export function spawnAndForward({ cmd, args }, { tieToStdin = false } = {}) {
  const child = spawn(cmd, args, {
    // The mcp routes pipe stdin through the launcher so it can observe EOF
    // (below). Passthrough keeps full inherit: piping would break TTY raw
    // mode for interactive commands, and a detached child would fall out of
    // the terminal's foreground group (no Ctrl-C, SIGTTIN on reads).
    stdio: tieToStdin ? ["pipe", "inherit", "inherit"] : "inherit",
    // Own process group, so killTree can signal npx and its mcp-remote
    // grandchild at once.
    detached: tieToStdin && process.platform !== "win32",
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
  if (tieToStdin) {
    // An MCP stdio server's contract: when the client's pipe closes, shut
    // down. mcp-remote does not honor stdin EOF (and npx forwards nothing),
    // so orphaned bridge trees outlive their MCP host and latch onto later
    // sessions. The launcher enforces the contract: on EOF, parent exit, or
    // a termination signal, SIGTERM the whole group, then SIGKILL after a
    // grace period long enough for the local server's own teardown.
    let ending = false;
    const shutdown = () => {
      if (ending) return;
      ending = true;
      killTree(child, "SIGTERM");
      setTimeout(() => killTree(child, "SIGKILL"), 5000).unref();
    };
    child.stdin.on("error", () => {}); // child died first; its exit settles things
    process.stdin.pipe(child.stdin);
    for (const ev of ["end", "close", "error"]) process.stdin.on(ev, shutdown);
    for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(sig, shutdown);
    // Last-resort sweep: however the launcher exits, the tree goes with it.
    process.on("exit", () => killTree(child, "SIGKILL"));
  } else {
    for (const sig of ["SIGINT", "SIGTERM"]) {
      process.on(sig, () => child.kill(sig));
    }
  }
}

async function resolveLocalBinary(env, { refresh = false, tag = null } = {}) {
  const debug = env.LILBEE_DEBUG === "1";
  // Routine runs are silent: detection and resolution chatter buffers here
  // and prints only when a download actually happens (or LILBEE_DEBUG=1).
  const pending = [];
  const say = debug ? log : (msg) => pending.push(msg);
  const { release: pinned, repo } = pinnedRelease();
  const host = await detectHost(env, say);
  const dir = cacheDir(env);
  const plan = await resolveBinary({
    env,
    cacheDir: dir,
    host,
    pinned,
    tag,
    refresh,
    includeDev: env.LILBEE_DEV_BUILDS === "1",
    repo: env.LILBEE_REPO || repo,
    log: say,
  });
  if (plan.download) {
    for (const msg of pending.splice(0)) log(msg);
    assertGlibcFloor();
    const report = progressReporter(log);
    try {
      await installRelease(plan.download, { cacheDir: dir, env, onProgress: report, log });
    } finally {
      report.finish();
    }
  }
  if (debug || plan.source === "download") {
    log(`lilbee: using binary from ${plan.source} (${plan.path})`);
  }
  return plan;
}

export async function run(argv) {
  assertNotRecursing();
  const env = process.env;
  if (argv[0] === "--launcher-help") {
    log(HELP);
    return;
  }
  const route = routeArgv(argv);

  if (route.kind === "unprepare") {
    // The inverse of prepare: delete every downloaded binary. npm >= 7 runs
    // no uninstall scripts, so this is the uninstall story:
    //   lilbee unprepare && npm uninstall -g lilbee
    const dir = cacheDir(env);
    fs.rmSync(dir, { recursive: true, force: true });
    log(`lilbee: removed the downloaded binaries (${dir}).`);
    return;
  }

  if (route.kind === "prepare") {
    // prepare is the explicit "get me the newest release": it re-resolves
    // latest even when a binary is already cached.
    const resolved = await resolveLocalBinary(env, { refresh: true, tag: route.tag });
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
    spawnAndForward(remoteExec(env), { tieToStdin: true });
    return;
  }

  const resolved = await resolveLocalBinary(env);
  if (route.kind === "mcp") {
    spawnAndForward(mcpExec(env, resolved.path, route.args), { tieToStdin: true });
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
