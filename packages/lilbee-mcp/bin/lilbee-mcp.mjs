#!/usr/bin/env node
/**
 * lilbee-mcp: run lilbee's MCP server (stdio) from npx.
 *
 * Remote mode (LILBEE_URL set) bridges stdio to a lilbee server's
 * streamable-http /mcp endpoint via mcp-remote. Local mode resolves a lilbee
 * binary (LILBEE_BIN -> PATH -> cache -> release download) and runs
 * `lilbee mcp`. All shim output goes to stderr; stdout is the MCP wire.
 */

import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { assetNameFor } from "../lib/assets.mjs";
import { download } from "../lib/download.mjs";
import { HELP, localExec, parseArgs, remoteExec, selectMode } from "../lib/plan.mjs";
import { resolveBinary } from "../lib/resolve.mjs";

const log = (msg) => console.error(msg);

function pinnedRelease() {
  const pkgPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "package.json");
  const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"));
  return { release: pkg.lilbee.release, repo: pkg.lilbee.repo };
}

function whichSync(name) {
  try {
    const tool = process.platform === "win32" ? "where.exe" : "which";
    const out = execFileSync(tool, [name], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    const first = out.split(/\r?\n/).find(Boolean);
    return first || null;
  } catch {
    return null;
  }
}

function mcpRemoteBin() {
  const require = createRequire(import.meta.url);
  const pkgJson = require.resolve("mcp-remote/package.json");
  const pkg = JSON.parse(fs.readFileSync(pkgJson, "utf8"));
  const rel = typeof pkg.bin === "string" ? pkg.bin : pkg.bin["mcp-remote"];
  return path.join(path.dirname(pkgJson), rel);
}

function run({ cmd, args }) {
  const child = spawn(cmd, args, { stdio: "inherit" });
  child.on("exit", (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    else process.exit(code ?? 0);
  });
  child.on("error", (err) => {
    log(`lilbee-mcp: failed to start ${cmd}: ${err.message}`);
    process.exit(1);
  });
  for (const sig of ["SIGINT", "SIGTERM"]) {
    process.on(sig, () => child.kill(sig));
  }
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.help) {
    log(HELP);
    return;
  }
  const env = process.env;
  const mode = selectMode(env);

  if (mode === "remote") {
    if (parsed.command === "prepare") {
      log("lilbee-mcp: remote mode (LILBEE_URL set) needs no download; nothing to prepare.");
      return;
    }
    if (!env.LILBEE_TOKEN) {
      log(
        "lilbee-mcp: LILBEE_URL is set but LILBEE_TOKEN is not — connecting without " +
          "auth. lilbee servers normally require the session token from server.json."
      );
    }
    log(`lilbee-mcp: bridging to ${env.LILBEE_URL}`);
    run(remoteExec(env, mcpRemoteBin()));
    return;
  }

  const { release, repo } = pinnedRelease();
  const effectiveRelease = env.LILBEE_RELEASE || release;
  const assetName = assetNameFor(process.platform, process.arch, env.LILBEE_VARIANT || "");
  const resolved = await resolveBinary({
    env: { LILBEE_REPO: repo, ...env },
    release: effectiveRelease,
    assetName,
    deps: { existsSync: fs.existsSync, whichSync, download: (o) => download({ ...o, log }) },
  });
  log(`lilbee-mcp: using lilbee from ${resolved.source} (${resolved.path})`);

  if (parsed.command === "prepare") {
    log("lilbee-mcp: ready.");
    return;
  }
  run(localExec(env, resolved.path, parsed));
}

main().catch((err) => {
  log(`lilbee-mcp: ${err.message}`);
  process.exit(1);
});
