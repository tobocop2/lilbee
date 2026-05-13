Source demos/_shared.tape
Output demos/_out/opencode-manual.gif

# opencode driving lilbee over MCP, on the Crown Victoria owner's manual.
# Caption: opencode on a cloud model (MiniMax) talking to a local lilbee
# corpus over MCP.
#
# Demo dir prepared by demos/_prep.sh: empty dir with AGENTS.md,
# opencode.json, the lilbee-worker subagent, the lilbee-mcp skill, and
# cv-manual.pdf already in place.
#
# Timings are estimates; expect to iterate.
#   indexing cv-manual.pdf via lilbee_add  ~25-35s
#   per Q&A round                          ~25-45s
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"

Type "cd /tmp/lilbee-demo/opencode-manual"
Sleep 400ms
Enter
Sleep 800ms

Type "opencode"
Sleep 500ms
Enter
Sleep 6s

# 1) Index + the headline question.
Type "Index cv-manual.pdf, then tell me what my oil capacity is."
Sleep 500ms
Enter
Sleep 60s

Screenshot demos/_out/opencode-manual.png

# 2) Oil pressure warning (matches the README copy).
Type "What does the oil pressure warning mean?"
Sleep 500ms
Enter
Sleep 40s

# 3) Tire pressure.
Type "What tire pressure does it recommend?"
Sleep 500ms
Enter
Sleep 40s

# 4) Engine + fuel.
Type "What engine does it have, and what fuel and octane does it take?"
Sleep 500ms
Enter
Sleep 45s

Ctrl+C
Sleep 1s
Type "exit"
Sleep 400ms
Enter
Sleep 1s
