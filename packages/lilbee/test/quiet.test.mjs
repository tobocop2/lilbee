import { test } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const run = promisify(execFile);
const bin = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "bin", "lilbee.mjs");

// The stub binary is a shell script, so these are POSIX-only.
const posix = process.platform !== "win32";

async function launch(env) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "lilbee-quiet-"));
  const stub = path.join(dir, "lilbee-stub");
  fs.writeFileSync(stub, "#!/bin/sh\necho ok\n");
  fs.chmodSync(stub, 0o755);
  try {
    return await run(process.execPath, [bin, "--version"], {
      env: { ...process.env, LILBEE_BIN: stub, LILBEE_DEBUG: "", ...env },
    });
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

test("a run on an already-resolved binary is silent on stderr", { skip: !posix }, async () => {
  const { stdout, stderr } = await launch({});
  assert.match(stdout, /ok/);
  assert.equal(stderr, "");
});

test("LILBEE_DEBUG=1 prints the resolution source", { skip: !posix }, async () => {
  const { stderr } = await launch({ LILBEE_DEBUG: "1" });
  assert.match(stderr, /using binary from env/);
});
