Source demos/_shared.tape
Output demos/_out/mcp-code.gif

# opencode driving lilbee over MCP, full TUI, on a focused slice of
# lilbee's own source: src/lilbee/core/, src/lilbee/data/ingest/, and
# src/lilbee/mcp_server.py.
#
# Demo dir prepped by demos/_prep.sh: shallow clone of lilbee with src/ at
# the root, AGENTS.md, opencode.json, lilbee-worker subagent, lilbee-mcp
# skill, and `lilbee init` already run. No pre-indexing -- the agent does
# the add via the lilbee-worker subagent.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-code"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 4s

# One turn: index the slice, then answer.
Type "Index ./src/lilbee/core, ./src/lilbee/data/ingest, and ./src/lilbee/mcp_server.py, then explain how lilbee discovers a project-local .lilbee/ directory. Cite the file and line range."
Sleep 600ms
Enter
Sleep 85s

Screenshot demos/_out/mcp-code.png
Sleep 2s

Ctrl+C
Sleep 1s
