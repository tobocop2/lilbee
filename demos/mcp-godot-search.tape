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
#   parent's lilbee_search + cited answer         ~30-45s
# Total real time: ~70-90s. PlaybackSpeed 3 -> ~25s GIF.
Set PlaybackSpeed 3.0

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

# Tight, citation-dense prompt. Six XMLs index in ~3s; the search +
# answer reads from AStarGrid2D.xml directly with method-level cites.
Type "Index ./godot-pathfinding/, then list every method on AStarGrid2D for marking cells as solid or non-solid. Cite each method with its file."
Sleep 600ms
Enter

# Indexing + search + cited answer.
Sleep 75s

Screenshot demos/_out/mcp-godot-search.png
Sleep 2s

Ctrl+C
Sleep 1s
