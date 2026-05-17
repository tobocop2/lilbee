Source "_shared.tape"
Output "_out/mcp-self-tune.mp4"

# opencode driving lilbee over MCP. PR-title beat: agent demonstrably
# improves its own answer by tuning lilbee with settings_set.
#
# Two-turn natural conversation:
#   Turn 1: user wants to evaluate lilbee defaults; agent does ONE
#           lilbee_search at defaults and reports what it finds.
#           Empirically: returns AStar3D only, explicitly notes
#           AStar2D may exist but was not in the results.
#   Turn 2: user reacts to the gap ("that's narrow") and asks the
#           agent to tune. Agent runs lilbee_settings_set widening
#           top_k / max_context_sources, runs the SAME search, now
#           finds AStar3D + AStar2D + AStarGrid2D, then resets.
#
# Empirical run (lilbee 0.6.66b478, opencode 1.14, qwen3.6-plus-free,
# M1):
#   opencode boot                                      ~8s
#   turn 1 (single search + narrow answer)             ~15-20s
#   user types turn 2                                  ~7s
#   turn 2 (settings_set + search + tuned answer +     ~40-50s
#           settings_reset)
# Total real time: ~75-90s. PlaybackSpeed 1.5 -> ~50-60s video.
#
# Corpus pre-indexed off-camera: godot 4 class reference (810 XMLs).

Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-self-tune"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

# Turn 1: evaluator framing forces a single search at defaults so the
# narrow baseline is on screen.
Type "I want to evaluate lilbee's defaults. Call lilbee_search ONCE with the query 'A star pathfinding' and answer ONLY from those results: which A* pathfinding classes does Godot 4 give me? Do not call lilbee_status, do not run a second search."
Sleep 800ms
Enter
Sleep 25s

# Turn 2: natural reaction to the gap. The phrasing leaves the
# tuning choice to the agent; SKILL workflow #5 names the pattern.
Type "That's narrow -- I'm pretty sure Godot has more A* classes. Tune lilbee for breadth via lilbee_settings_set (godot reference is dense XML, one class per file) and run the same search. Then answer again. Reset settings when done."
Sleep 800ms
Enter
Sleep 55s

Screenshot "_out/mcp-self-tune.png"

Ctrl+C
