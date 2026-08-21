#!/usr/bin/env node
// MCP-first entry: identical to `lilbee mcp [...]`. Kept as its own bin so
// MCP host configs stay one token, and as a stable file path for hosts that
// run a local build directly with `node`.
import { runAndReport } from "../lib/cli.mjs";
runAndReport(["mcp", ...process.argv.slice(2)]);
