Source _shared.tape
Output _out/mcp-manual.gif

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

# Substantive prompt that produces a multi-row cited table -- the kind
# of structured output a frontier model is good at when the underlying
# data is in front of it. Verified in tmux against the indexed PDF:
# MiniMax returns the engine-compartment fuse list (fuses 19-24 + 101-118
# with mini/cartridge marker, amp rating, and the circuit each protects),
# all from the manual's "Roadside Emergencies / Fuses and Relays" section.
# Measured 1m 2s for the final-token render; the full demo (boot + index
# ~25s + planning + render) is ~2 min end-to-end.
Type "Index cv-manual.pdf and then make me a table of the fuses in the engine compartment with their amp ratings and the circuits they protect. Cite the page."
Sleep 600ms
Enter
# Measured 1m 2s final-token render; trimmed to no trailing dead frame.
Sleep 105s

Screenshot _out/mcp-manual.png

Ctrl+C
