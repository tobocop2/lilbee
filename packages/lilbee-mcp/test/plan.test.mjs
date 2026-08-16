import { test } from "node:test";
import assert from "node:assert/strict";
import { parseArgs, selectMode, remoteExec, localExec } from "../lib/plan.mjs";

test("parseArgs: default run, prepare, data-dir, help, unknown", () => {
  assert.deepEqual(parseArgs([]), { command: "run", dataDir: null, help: false });
  assert.equal(parseArgs(["prepare"]).command, "prepare");
  assert.equal(parseArgs(["--data-dir", "/x"]).dataDir, "/x");
  assert.equal(parseArgs(["-d", "/y"]).dataDir, "/y");
  assert.equal(parseArgs(["--help"]).help, true);
  assert.throws(() => parseArgs(["--nope"]), /Unknown argument/);
});

test("selectMode: LILBEE_URL toggles remote", () => {
  assert.equal(selectMode({}), "local");
  assert.equal(selectMode({ LILBEE_URL: "" }), "local");
  assert.equal(selectMode({ LILBEE_URL: "http://localhost:8383/mcp" }), "remote");
});

test("remoteExec: url + http-only transport + bearer header when token set", () => {
  const env = { LILBEE_URL: "http://localhost:8383/mcp", LILBEE_TOKEN: "tok123" };
  const { cmd, args } = remoteExec(env, "/deps/mcp-remote/proxy.js");
  assert.equal(cmd, process.execPath);
  assert.deepEqual(args, [
    "/deps/mcp-remote/proxy.js",
    "http://localhost:8383/mcp",
    "--transport",
    "http-only",
    "--header",
    "Authorization: Bearer tok123",
  ]);
});

test("remoteExec: no header without a token", () => {
  const { args } = remoteExec({ LILBEE_URL: "http://h/mcp" }, "/p");
  assert.ok(!args.includes("--header"));
});

test("localExec: bare, env data dir, flag wins over env", () => {
  assert.deepEqual(localExec({}, "/bin/lilbee", { dataDir: null }).args, ["mcp"]);
  assert.deepEqual(localExec({ LILBEE_DATA_DIR: "/e" }, "/bin/lilbee", { dataDir: null }).args, [
    "mcp",
    "--data-dir",
    "/e",
  ]);
  assert.deepEqual(localExec({ LILBEE_DATA_DIR: "/e" }, "/bin/lilbee", { dataDir: "/f" }).args, [
    "mcp",
    "--data-dir",
    "/f",
  ]);
});
