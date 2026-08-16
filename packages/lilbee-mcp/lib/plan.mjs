/**
 * Pure planning: turn env + argv into the command this shim runs.
 *
 * Two modes:
 *  - remote (LILBEE_URL set): bridge stdio <-> streamable-http via mcp-remote,
 *    with lilbee's bearer session token when LILBEE_TOKEN is set.
 *  - local (default): run `<binary> mcp`, where the binary comes from
 *    LILBEE_BIN, PATH, or a bootstrapped download (resolved elsewhere).
 *
 * Kept side-effect free so tests cover every branch without spawning anything.
 */

export function parseArgs(argv) {
  const out = { command: "run", dataDir: null, help: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "prepare") out.command = "prepare";
    else if (a === "--data-dir" || a === "-d") out.dataDir = argv[++i] ?? null;
    else if (a === "--help" || a === "-h") out.help = true;
    else throw new Error(`Unknown argument "${a}". See lilbee-mcp --help.`);
  }
  return out;
}

/** "remote" when LILBEE_URL is set (non-empty), else "local". */
export function selectMode(env) {
  return env.LILBEE_URL ? "remote" : "local";
}

/**
 * argv for mcp-remote in remote mode. `mcpRemoteBin` is the resolved path to
 * mcp-remote's executable script; it runs under the current node.
 */
export function remoteExec(env, mcpRemoteBin) {
  const args = [mcpRemoteBin, env.LILBEE_URL, "--transport", "http-only"];
  if (env.LILBEE_TOKEN) {
    args.push("--header", `Authorization: Bearer ${env.LILBEE_TOKEN}`);
  }
  return { cmd: process.execPath, args };
}

/** argv for the lilbee binary in local mode. */
export function localExec(env, binaryPath, parsed) {
  const args = ["mcp"];
  const dataDir = parsed.dataDir || env.LILBEE_DATA_DIR;
  if (dataDir) args.push("--data-dir", dataDir);
  return { cmd: binaryPath, args };
}

export const HELP = `lilbee-mcp — run lilbee's MCP server (stdio)

Usage:
  lilbee-mcp [--data-dir <dir>]   start the MCP server
  lilbee-mcp prepare              download the lilbee binary and exit

Environment:
  LILBEE_URL       remote lilbee server; bridges to <url> instead of a local binary
  LILBEE_TOKEN     bearer session token for LILBEE_URL
  LILBEE_BIN       explicit path to a lilbee binary
  LILBEE_DATA_DIR  library location (same as --data-dir)
  LILBEE_VARIANT   download variant: cu121 | cu124 | cu125 | rocm | compat
  LILBEE_RELEASE   override the pinned lilbee release tag
`;
