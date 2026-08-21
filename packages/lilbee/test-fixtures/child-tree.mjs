// A two-level process tree that never exits on its own: the stand-in for
// npx -> mcp-remote in the lifetime tests. Prints both pids on stdout so the
// test can watch the tree from outside.
import { spawn } from "node:child_process";

const grand = spawn(process.execPath, ["-e", "setInterval(() => {}, 60000)"], { stdio: "ignore" });
console.log(JSON.stringify({ child: process.pid, grand: grand.pid }));
setInterval(() => {}, 60000);
