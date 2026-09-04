import { test } from "node:test";
import assert from "node:assert/strict";
import {
  exitCodeForSignal,
  routeArgv,
  parseMcpArgs,
  selectMode,
  remoteExec,
  MCP_REMOTE_SPEC,
  mcpExec,
  passthroughExec,
} from "../lib/plan.mjs";

test("routeArgv: prepare, mcp, and passthrough", () => {
  assert.deepEqual(routeArgv(["prepare"]), { kind: "prepare", tag: null });
  assert.deepEqual(routeArgv(["unprepare"]), { kind: "unprepare" });
  assert.equal(routeArgv(["mcp"]).kind, "mcp");
  assert.equal(routeArgv(["mcp", "--data-dir", "/x"]).args.dataDir, "/x");
  assert.deepEqual(routeArgv(["chat"]), { kind: "exec", argv: ["chat"] });
  assert.deepEqual(routeArgv(["model", "list"]), { kind: "exec", argv: ["model", "list"] });
  assert.deepEqual(routeArgv([]), { kind: "exec", argv: [] });
});

test("parseMcpArgs: data-dir flags and extras", () => {
  assert.equal(parseMcpArgs(["--data-dir", "/x"]).dataDir, "/x");
  assert.equal(parseMcpArgs(["-d", "/y"]).dataDir, "/y");
  assert.deepEqual(parseMcpArgs(["--global"]).extra, ["--global"]);
});

test("selectMode: LILBEE_URL toggles remote", () => {
  assert.equal(selectMode({}), "local");
  assert.equal(selectMode({ LILBEE_URL: "" }), "local");
  assert.equal(selectMode({ LILBEE_URL: "http://localhost:8383/mcp" }), "remote");
});

test("remoteExec: npx-launched bridge, http-only transport, bearer header when token set", () => {
  const env = { LILBEE_URL: "http://localhost:8383/mcp", LILBEE_TOKEN: "tok123" };
  const { cmd, args } = remoteExec(env);
  assert.match(cmd, /^npx(\.cmd)?$/);
  assert.deepEqual(args, [
    "-y",
    MCP_REMOTE_SPEC,
    "http://localhost:8383/mcp",
    "--transport",
    "http-only",
    "--header",
    "Authorization: Bearer tok123",
  ]);
  assert.match(MCP_REMOTE_SPEC, /^mcp-remote@\d/);
});

test("remoteExec: no header without a token", () => {
  const { args } = remoteExec({ LILBEE_URL: "http://h/mcp" });
  assert.ok(!args.includes("--header"));
});

test("mcpExec: bare, env data dir, flag wins over env, extras pass", () => {
  assert.deepEqual(mcpExec({}, "/bin/lilbee", { dataDir: null, extra: [] }).args, ["mcp"]);
  assert.deepEqual(
    mcpExec({ LILBEE_DATA_DIR: "/e" }, "/bin/lilbee", { dataDir: null, extra: [] }).args,
    ["mcp", "--data-dir", "/e"]
  );
  assert.deepEqual(
    mcpExec({ LILBEE_DATA_DIR: "/e" }, "/bin/lilbee", { dataDir: "/f", extra: ["--global"] }).args,
    ["mcp", "--data-dir", "/f", "--global"]
  );
});

test("passthroughExec: verbatim argv", () => {
  assert.deepEqual(passthroughExec("/bin/lilbee", ["chat", "-d", "/lib"]), {
    cmd: "/bin/lilbee",
    args: ["chat", "-d", "/lib"],
  });
});

test("exitCodeForSignal maps known signals and defaults to 128", () => {
  assert.equal(exitCodeForSignal("SIGINT"), 130);
  assert.equal(exitCodeForSignal("SIGTERM"), 143);
  assert.equal(exitCodeForSignal("SIGWEIRD"), 128);
});

test("routeArgv: prepare takes an optional release tag", () => {
  assert.deepEqual(routeArgv(["prepare", "v0.6.90b432"]), { kind: "prepare", tag: "v0.6.90b432" });
  assert.deepEqual(routeArgv(["prepare"]), { kind: "prepare", tag: null });
});
