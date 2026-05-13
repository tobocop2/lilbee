Source demos/_shared.tape
Output demos/_out/opencode-manual.gif

# opencode driving lilbee over MCP, on the Crown Victoria owner's manual.
# One turn end-to-end (index + answer) to keep the demo short. Tool calls
# and the lilbee-worker delegation are visible inline as `⚙ lilbee_<tool>`
# and `• Index ... ✓ Index ... Lilbee-Worker Agent`.
#
# Demo dir prepped by demos/_prep.sh: AGENTS.md, opencode.json,
# lilbee-worker subagent, lilbee-mcp skill, cv-manual.pdf in the dir, and
# `lilbee init` already run. No pre-indexing — the agent does the add.
#
# Model: opencode/deepseek-v4-flash-free. Cloud variance is real:
# dry-runs landed at 50s for this turn; budget 90s.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-manual"
Sleep 400ms
Enter
Sleep 600ms

# One turn: index + answer. The Lilbee-Worker Agent runs the add; ⚙
# lilbee_search fires once the worker reports done.
Type "opencode run --model opencode/deepseek-v4-flash-free 'Index cv-manual.pdf, then tell me my oil capacity. Cite the page.'"
Sleep 400ms
Enter
Sleep 90s

Screenshot demos/_out/opencode-manual.png
Sleep 2s
