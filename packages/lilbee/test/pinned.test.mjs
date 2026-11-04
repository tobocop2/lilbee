import { test } from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { assetNameFor } from "../lib/assets.mjs";

const run = promisify(execFile);
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const bin = path.join(root, "bin", "lilbee.mjs");
const pinned = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8")).lilbee.release;

// The stub binaries are shell scripts, so these are POSIX-only.
const posix = process.platform !== "win32";

function seed(cache, release, says) {
  const asset = assetNameFor(process.platform, process.arch, "default");
  fs.mkdirSync(path.join(cache, release), { recursive: true });
  const file = path.join(cache, release, asset);
  fs.writeFileSync(file, `#!/bin/sh\necho ${says}\n`);
  fs.chmodSync(file, 0o755);
}

test("the launcher runs the release package.json pins, not a newer cached one", { skip: !posix }, async () => {
  const cache = fs.mkdtempSync(path.join(os.tmpdir(), "lilbee-pinned-"));
  try {
    seed(cache, pinned, "pinned");
    seed(cache, "v99.0.0", "newer");
    const env = { ...process.env, LILBEE_MCP_CACHE: cache, LILBEE_VARIANT: "default", LILBEE_BIN: "", LILBEE_RELEASE: "" };
    const { stdout } = await run(process.execPath, [bin, "--version"], { env });
    assert.equal(stdout.trim(), "pinned");
  } finally {
    fs.rmSync(cache, { recursive: true, force: true });
  }
});
