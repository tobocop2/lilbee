/**
 * Pure planning: turn argv + env into the command this launcher runs.
 *
 * Routing:
 *  - `prepare`            download/verify the binary, then exit (launcher-only)
 *  - `mcp [...]`          run the MCP server; with LILBEE_URL set, bridge
 *                         stdio <-> streamable-http via mcp-remote instead
 *  - anything else        resolve the binary and exec argv verbatim
 *
 * Kept side-effect free so tests cover every branch without spawning anything.
 */

/** Classify argv into a launcher route. */
export function routeArgv(argv) {
  const [head, ...rest] = argv;
  if (head === "prepare") return { kind: "prepare" };
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
  lilbee prepare                  download the lilbee binary and exit
  lilbee-mcp [...]                same as \`lilbee mcp [...]\`

Environment:
  LILBEE_URL       remote lilbee server; \`mcp\` bridges to <url> instead of a local binary
  LILBEE_TOKEN     bearer session token for LILBEE_URL
  LILBEE_BIN       explicit path to a lilbee binary
  LILBEE_DATA_DIR  library location for \`mcp\` (same as --data-dir)
  LILBEE_VARIANT   download variant: cu121 | cu124 | cu125 | rocm | compat
  LILBEE_RELEASE   override the pinned lilbee release tag
`;
