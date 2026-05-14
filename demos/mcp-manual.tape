Source demos/_shared.tape
Output demos/_out/mcp-manual.gif

# opencode driving lilbee over MCP, full TUI. One turn end-to-end (index +
# answer) on the Crown Victoria owner's manual.
#
# What's visible on camera:
#   * opencode TUI loads with the dev's default model
#   * Type the prompt; tool calls render inline as `lilbee_<tool>`
#   * Long-running lilbee_add is delegated to the lilbee-worker subagent;
#     opencode shows `* Index cv-manual.pdf  Lilbee-Worker Agent`
#     followed by `+ Index cv-manual.pdf  Lilbee-Worker Agent`
#   * lilbee_search fires; the answer cites cv-manual.pdf with a page
#
# Demo dir prepped by demos/_prep.sh: AGENTS.md, opencode.json,
# lilbee-worker subagent, lilbee-mcp skill, cv-manual.pdf in the dir, and
# `lilbee init` already run. No pre-indexing -- the agent does the add.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-manual"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

# One turn: index + cited answer. The agent runs lilbee_add via the
# lilbee-worker subagent (~30s), then lilbee_status, then lilbee_search,
# then writes the cited answer. Measured ~75s end-to-end on M1; 90s of
# Sleep covers it with light margin and avoids a long idle tail.
Type "Index cv-manual.pdf, then tell me my oil capacity. Cite the page."
Sleep 600ms
Enter
Sleep 90s

Screenshot demos/_out/mcp-manual.png
Sleep 2s

Ctrl+C
Sleep 1s
