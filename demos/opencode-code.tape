Source demos/_shared.tape
Output demos/_out/opencode-code.gif

# opencode driving lilbee over MCP, on a focused slice of lilbee's own
# source: src/lilbee/core/, src/lilbee/data/ingest/, and
# src/lilbee/mcp_server.py. One turn end-to-end (index + answer).
#
# Demo dir prepped by demos/_prep.sh: shallow clone of lilbee with src/ at
# the root, AGENTS.md, opencode.json, lilbee-worker subagent, lilbee-mcp
# skill, and `lilbee init` already run. No pre-indexing.
#
# Model: opencode/deepseek-v4-flash-free. Dry-runs landed at 50s for this
# turn; budget 90s.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-code"
Sleep 400ms
Enter
Sleep 600ms

# One turn: index the slice, then answer.
Type "opencode run --model opencode/deepseek-v4-flash-free 'Use lilbee_add to index ./src/lilbee/core, ./src/lilbee/data/ingest, and ./src/lilbee/mcp_server.py. Then use lilbee_search to find how lilbee discovers a project-local .lilbee/ directory. Cite the file and line.'"
Sleep 400ms
Enter
Sleep 90s

Screenshot demos/_out/opencode-code.png
Sleep 2s
