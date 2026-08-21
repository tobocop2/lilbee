// Drives spawnAndForward the way the mcp routes do, over a disposable child
// tree, so the lifetime test can observe the whole tree from a real parent.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnAndForward } from "../lib/cli.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const target = process.argv[2] === "exit-code" ? ["-e", "process.exit(7)"] : [path.join(here, "child-tree.mjs")];
spawnAndForward({ cmd: process.execPath, args: target }, { tieToStdin: true });
