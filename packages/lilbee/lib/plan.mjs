/**
 * Pure planning: turn argv + env into the command this launcher runs.
 *
 * Routing:
 *  - `prepare [<tag>]`    download/verify the binary, then exit (launcher-only)
 *  - `mcp [...]`          run the MCP server; with LILBEE_URL set, bridge
 *                         stdio <-> streamable-http via mcp-remote instead
 *  - anything else        resolve the binary and exec argv verbatim
 *
 * Kept side-effect free so tests cover every branch without spawning anything.
 */

/** Classify argv into a launcher route. */
export function routeArgv(argv) {
  const [head, ...rest] = argv;
  if (head === "prepare") return { kind: "prepare", tag: rest[0] ?? null };
  if (head === "unprepare") return { kind: "unprepare" };
  if (head === "mcp") return { kind: "mcp", args: parseMcpArgs(rest) };
  return { kind: "exec", argv };
}

/** Flags the launcher understands for the mcp route. */
export function parseMcpArgs(argv) {
  const out = { dataDir: null, extra: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--data-dir" || a === "-d") out.dataDir = argv[++i] ?? null;
    else out.extra.push(a);
  }
  return out;
}

const EXIT_CODE_BY_SIGNAL = { SIGHUP: 129, SIGINT: 130, SIGQUIT: 131, SIGTERM: 143 };
const EXIT_CODE_SIGNAL_DEFAULT = 128;

/** Conventional 128+signum exit code for a child killed by *signal*. */
export function exitCodeForSignal(signal) {
  return EXIT_CODE_BY_SIGNAL[signal] ?? EXIT_CODE_SIGNAL_DEFAULT;
}

/** "remote" when LILBEE_URL is set (non-empty), else "local". */
export function selectMode(env) {
  return env.LILBEE_URL ? "remote" : "local";
}

/**
 * argv for mcp-remote in remote mode. `mcpRemoteBin` is the resolved path to
 * mcp-remote's executable script; it runs under the current node.
 */
// Pinned bridge version for remote mode. Fetched on demand via npx so the
// launcher itself installs with zero runtime dependencies: local-mode users
// (the majority) never download the bridge's tree, and its engine floor
// never gates installing lilbee.
export const MCP_REMOTE_SPEC = "mcp-remote@0.1.38";

export function remoteExec(env) {
  // Bought, not built: mcp-remote owns the stdio<->streamable-http bridge.
  // The bearer header rides argv (mcp-remote's contract), which is visible in
  // the process list; acceptable for a single-user machine.
  const npx = process.platform === "win32" ? "npx.cmd" : "npx";
  const args = ["-y", MCP_REMOTE_SPEC, env.LILBEE_URL, "--transport", "http-only"];
  if (env.LILBEE_TOKEN) {
    args.push("--header", `Authorization: Bearer ${env.LILBEE_TOKEN}`);
  }
  return { cmd: npx, args };
}

/** argv for the lilbee binary on the mcp route. */
export function mcpExec(env, binaryPath, mcpArgs) {
  const args = ["mcp"];
  const dataDir = mcpArgs.dataDir || env.LILBEE_DATA_DIR;
  if (dataDir) args.push("--data-dir", dataDir);
  args.push(...mcpArgs.extra);
  return { cmd: binaryPath, args };
}

/** argv for the lilbee binary on the passthrough route: verbatim. */
export function passthroughExec(binaryPath, argv) {
  return { cmd: binaryPath, args: argv };
}

export const HELP = `lilbee (npm launcher) — run lilbee anywhere

Usage:
  lilbee <any lilbee command>     bootstrap the binary if needed, then run it
  lilbee mcp [--data-dir <dir>]   start the MCP server (stdio)
  lilbee prepare [<tag>]          download (or upgrade to) the latest lilbee binary and exit;
                                  with a release tag, install exactly that release
  lilbee unprepare                delete every downloaded binary (run before npm uninstall)
  lilbee-mcp [...]                same as \`lilbee mcp [...]\`

Environment:
  LILBEE_URL         remote lilbee server; \`mcp\` bridges to <url> instead of a local binary
  LILBEE_TOKEN       bearer session token for LILBEE_URL
  LILBEE_BIN         explicit path to a lilbee binary
  LILBEE_DATA_DIR    library location for \`mcp\` (same as --data-dir)
  LILBEE_VARIANT     download variant override: default | cu121 | cu124 | cu125 |
                     rocm | compat | compat-cu124 | compat-rocm (unset =
                     auto-detect; default = the plain build on any host)
  LILBEE_RELEASE     run an exact lilbee release tag instead of the latest
  LILBEE_DEV_BUILDS  =1 lets "latest" pick in-development (.dev) builds
  LILBEE_DEBUG       =1 prints binary resolution detail on every run
`;
