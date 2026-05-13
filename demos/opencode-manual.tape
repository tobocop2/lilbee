Source demos/_shared.tape
Output demos/_out/opencode-manual.gif

# opencode driving lilbee over MCP, on the Crown Victoria owner's manual.
# Caption (when this lands in docs/demos.md): opencode on a cloud model
# (MiniMax) talking to a local lilbee corpus over MCP.
#
# Demo dir was prepared by demos/_prep.sh: an empty dir with AGENTS.md,
# opencode.json, the lilbee-worker subagent, the lilbee-mcp skill, and
# cv-manual.pdf already in place.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"

Type "cd /tmp/lilbee-demo/opencode-manual"
Sleep 300ms
Enter
Sleep 400ms

Type "opencode"
Sleep 400ms
Enter
Sleep 5s

# 1) Index + the headline question.
Type "Index cv-manual.pdf, then tell me what my oil capacity is."
Sleep 400ms
Enter
Sleep 60s

Screenshot demos/_out/opencode-manual.png

# 2) Oil pressure warning (matches the README copy).
Type "What does the oil pressure warning mean?"
Sleep 400ms
Enter
Sleep 40s

# 3) Tire pressure.
Type "What tire pressure does it recommend?"
Sleep 400ms
Enter
Sleep 40s

# 4) Engine + fuel.
Type "What engine does it have, and what fuel and octane does it take?"
Sleep 400ms
Enter
Sleep 40s

# Exit.
Ctrl+C
Sleep 1s
Type "exit"
Sleep 300ms
Enter
Sleep 1s
