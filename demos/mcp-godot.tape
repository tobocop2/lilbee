Source demos/_shared.tape
Output demos/_out/mcp-godot.gif

# opencode driving lilbee over MCP, full TUI. Heavy cited-codegen demo
# against the FULL Godot 4 class reference (810 XMLs). The corpus is
# pre-indexed off-camera by demos/_prep.sh -- this tape opens straight
# into the cited answer phase, which is what lilbee shines at.
#
# (The companion mcp-godot-search.tape covers the live-indexing flow on
# a small subset; we don't repeat it here.)
#
# Demo dir prepped by demos/_prep.sh: AGENTS.md, opencode.json,
# lilbee-worker subagent, lilbee-mcp skill, godot-classes/ copied in
# (~50 MB of XML), `lilbee init` run, AND `lilbee add ./godot-classes/`
# already executed so the corpus is ready.
#
# Empirical timing (lilbee 0.6.66b473, opencode 1.14, M1):
#   opencode boot                                 ~10s
#   prompt + planning + lilbee_status check       ~10s
#   lilbee_search (1-3 calls)                    ~10-15s
#   cited code answer streaming                  ~150-180s
# Total real time: ~3-3.5 min. PlaybackSpeed 4 -> ~50s GIF.
Set PlaybackSpeed 4.0

Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-godot"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 10s

# Specific prompt: name the godot APIs to use so the agent goes
# straight to lilbee_search and writes code instead of asking
# clarifying questions. Names TileMap + AStarGrid2D + RandomNumberGenerator
# explicitly so each show up as a separate lilbee_search hit on camera.
Type "Use lilbee_search to look up the godot 4 docs you need, then write a procedural roguelike-dungeon level generator: place wall and floor tiles via TileMap, scatter collectibles with RandomNumberGenerator, connect rooms with AStarGrid2D pathfinding. Cite the godot-classes XML for every method you call."
Sleep 800ms
Enter

# Status check (~5s), search calls (~10s), cited code stream (~150-180s).
Sleep 200s

Screenshot demos/_out/mcp-godot.png
Sleep 3s

Ctrl+C
Sleep 1s
