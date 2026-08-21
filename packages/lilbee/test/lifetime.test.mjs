import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Fixtures live outside test/: the runner would execute any .mjs in here.
const fixture = (name) => path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "test-fixtures", name);

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function eventually(pred, ms = 5000) {
  const deadline = Date.now() + ms;
  while (Date.now() < deadline) {
    if (pred()) return true;
    await new Promise((r) => setTimeout(r, 50));
  }
  return pred();
}

function spawnTied(arg) {
  const args = [fixture("forward-tied.mjs")];
  if (arg) args.push(arg);
  return spawn(process.execPath, args, { stdio: ["pipe", "pipe", "inherit"] });
}

test("closing stdin kills the whole tied child tree, grandchildren included", async () => {
  const launcher = spawnTied();
  let out = "";
  launcher.stdout.on("data", (d) => (out += d));
  await eventually(() => out.includes("\n"));
  const pids = JSON.parse(out);
  assert.ok(alive(pids.child), "child should be running before EOF");
  assert.ok(alive(pids.grand), "grandchild should be running before EOF");

  launcher.stdin.end();
  await once(launcher, "exit");
  const gone = await eventually(() => !alive(pids.child) && !alive(pids.grand));
  assert.ok(gone, `tree survived stdin EOF (child alive: ${alive(pids.child)}, grand alive: ${alive(pids.grand)})`);
});

test("a tied child's own exit code still reaches the caller", async () => {
  const launcher = spawnTied("exit-code");
  const [code] = await once(launcher, "exit");
  assert.equal(code, 7);
});
