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
Sleep 8s

# One turn: index the slice, then answer. The agent runs through:
# (a) load lilbee-mcp skill, (b) delegate lilbee_add to lilbee-worker
# (~25s), (c) run lilbee_search, (d) cite the answer. Sleep covers the
# whole sequence with margin.
Type "Index ./src/lilbee/core, ./src/lilbee/data/ingest, and ./src/lilbee/mcp_server.py, then show me the function that walks up from the current directory to find a project-local .lilbee/ directory. Quote the function body and cite the file and line range."
Sleep 600ms
Enter
Sleep 150s

Screenshot demos/_out/mcp-code.png
Sleep 3s

Ctrl+C
Sleep 1s
