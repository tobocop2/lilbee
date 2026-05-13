Source demos/_shared.tape
Output demos/_out/opencode-code.gif

# opencode driving lilbee over MCP, on a focused slice of lilbee's own
# source: src/lilbee/core/ (project-local .lilbee/ discovery),
# src/lilbee/data/ingest/ (the ingest pipeline), and src/lilbee/mcp_server.py.
# Indexing the full src/ tree would take minutes; this slice runs in ~6s of
# real embedding plus the cloud-model turn.
#
# Demo dir prepared by demos/_prep.sh: shallow clone of lilbee with src/ at
# the project root, AGENTS.md, opencode.json, lilbee-worker subagent, and
# lilbee-mcp skill in place, and `lilbee init` already run.
#
# `opencode run` is non-interactive: `⚙ lilbee_<tool>` calls and the
# `Lilbee-Worker Agent` delegation appear inline in the captured terminal.
#
# Model: opencode/deepseek-v4-flash-free (free, fastest of the free models
# I tested). Cloud latency varies (dry-runs landed in 50-90s for the first
# turn; budget 240s here so a slow queue doesn't clip the GIF).
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-code"
Sleep 400ms
Enter
Sleep 800ms

# Show the agent contract briefly.
Type "head -20 AGENTS.md"
Sleep 400ms
Enter
Sleep 4s

Type "clear"
Sleep 300ms
Enter
Sleep 400ms

# First turn: real index + answer. The Lilbee-Worker Agent does the add;
# ⚙ lilbee_search runs after the worker reports done.
Type "opencode run --model opencode/deepseek-v4-flash-free 'Use lilbee_add to index ./src/lilbee/core, ./src/lilbee/data/ingest, and ./src/lilbee/mcp_server.py. Then use lilbee_search to find how lilbee discovers a project-local .lilbee/ directory. Cite the file and line.'"
Sleep 500ms
Enter
Sleep 240s

Screenshot demos/_out/opencode-code.png

# Warm follow-ups (no re-indexing).
Type "opencode run --model opencode/deepseek-v4-flash-free 'Walk me through what lilbee does when I run add on a folder. Cite the code.'"
Sleep 500ms
Enter
Sleep 60s

Type "opencode run --model opencode/deepseek-v4-flash-free 'How is lilbee itself exposed as an MCP server? Cite the file.'"
Sleep 500ms
Enter
Sleep 60s
