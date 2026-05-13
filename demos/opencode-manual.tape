Source demos/_shared.tape
Output demos/_out/opencode-manual.gif

# opencode driving lilbee over MCP, on the Crown Victoria owner's manual.
# Uses `opencode run` (non-interactive) so the demo is deterministic and the
# `lilbee_search` / `lilbee_add` tool calls are explicitly visible inline as
# `⚙ lilbee_<tool>` lines.
#
# Demo dir prepared by demos/_prep.sh: AGENTS.md, opencode.json (with
# LILBEE_DATA pointed at the project's .lilbee/), the lilbee-worker subagent,
# the lilbee-mcp skill, cv-manual.pdf in the dir, and `lilbee init` already
# run.
#
# Model: opencode/deepseek-v4-flash-free (free, fast, 20-30s per turn against
# this corpus). Captioned: cloud model + local lilbee corpus over MCP.
#
# Cloud-model latency varies; budget generously so a slow turn doesn't clip
# the GIF. Dry-runs landed: first turn 55-105s, warm follow-up 22-25s.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-manual"
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

# First turn: real index + answer. The agent delegates indexing to
# lilbee-worker (visible as `• Index ... ✓ Index ... Lilbee-Worker Agent`),
# then ⚙ lilbee_search fires once the worker reports done.
Type "opencode run --model opencode/deepseek-v4-flash-free 'Index cv-manual.pdf, then tell me my oil capacity. Cite the page.'"
Sleep 500ms
Enter
Sleep 180s

Screenshot demos/_out/opencode-manual.png

# Warm follow-ups (no re-indexing).
Type "opencode run --model opencode/deepseek-v4-flash-free 'What does the oil pressure warning mean? Cite the page.'"
Sleep 500ms
Enter
Sleep 60s

Type "opencode run --model opencode/deepseek-v4-flash-free 'What tire pressure does it recommend? Cite the page.'"
Sleep 500ms
Enter
Sleep 60s
