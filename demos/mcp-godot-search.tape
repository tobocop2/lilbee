Source demos/_shared.tape
Output demos/_out/mcp-godot-search.gif

# opencode driving lilbee over MCP, full TUI. Tight live-indexing demo
# on a 6-file Godot pathfinding subset (AStarGrid2D family + Vector2 +
# Rect2i). One turn end-to-end: index, search, cited answer.
#
# Demo dir prepped by demos/_prep.sh: AGENTS.md, opencode.json,
# lilbee-worker subagent, lilbee-mcp skill, godot-pathfinding/ copied
# in (~110 KB), `lilbee init` run, no pre-indexing.
#
# Empirical timing (lilbee 0.6.66b473, opencode 1.14, M1):
#   opencode boot                                 ~10s
#   prompt + planning + worker spawn              ~20s
#   worker indexes 6 XMLs                         ~3s
#   parent's lilbee_search + cited answer         ~60-90s   (cited list takes a while to stream)
# Total real time: ~2 min 20s. PlaybackSpeed 1.5 -> ~95s GIF;
# stays readable, citations don't fly past.
Set PlaybackSpeed 1.5

Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
# 6 XMLs index in seconds; the 300s default is plenty.
Env LILBEE_WORKER_POOL_CALL_TIMEOUT_S "300"

Type "cd /tmp/lilbee-demo/opencode-godot-search"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 10s

# Natural prompt -- no "cite each method" instruction (AGENTS.md
# already requires file:line cites). The "index this folder" framing
# is what a human would actually say to bootstrap a fresh corpus.
Type "Index ./godot-pathfinding/ and tell me how AStarGrid2D handles marking cells as solid or non-solid."
Sleep 600ms
Enter

# Indexing + search + cited answer. The cited method list streams for
# ~90s on MiniMax M2.7; bump the wait so the screenshot lands AFTER the
# answer has fully rendered.
Sleep 120s

Screenshot demos/_out/mcp-godot-search.png
Sleep 2s

Ctrl+C
Sleep 1s
