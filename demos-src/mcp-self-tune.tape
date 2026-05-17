Source "_shared.tape"
Output "_out/mcp-self-tune.gif"

# opencode driving lilbee over MCP. The PR-title beat: agent tunes lilbee
# for a better answer. Same question, same MCP server, same model; only
# the settings change. The second answer is measurably richer.
#
# Empirical baseline (lilbee 0.6.66b478, opencode 1.14, qwen3.6-plus-free,
# M1): with defaults, `lilbee_search "A star pathfinding"` against the
# godot-classes XML corpus returns AStar3D as the only A* class (the
# rest are NavigationAgent / Region classes that mention pathfinding but
# aren't A*). One settings_set (`diversity_max_per_source=8 top_k=15
# max_distance=0.85`) and the same query returns AStar3D plus AStarGrid2D
# plus the NavigationPathQueryParameters family. The on-screen answer
# goes from one class to two-plus.
#
# Corpus is the godot 4 class reference (810 XMLs, ~3450 chunks),
# pre-indexed off-camera by demos/_prep.sh.
#
# Tape uses three sequential prompts (baseline / tune+answer / reset)
# rather than one nine-step protocol, so each answer lingers on screen
# instead of being scrolled off by the next agent turn.

Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-self-tune"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

# Prompt 1: defaults baseline. lilbee_settings_reset first to make sure
# the agent isn't carrying over a previous session's tuning, then one
# search with the verbatim query, then a tight cited answer.
Type "lilbee_settings_reset diversity_max_per_source top_k max_distance. Then lilbee_search 'A star pathfinding'. Using ONLY those results, write a 2-line answer naming every A* class with one use case each, cite the file. No prose preamble."
Sleep 800ms
Enter
Sleep 90s

# Prompt 2: same question after one tuning step. The settings_set is
# the visible point of the demo; the second answer should land richer.
Type "lilbee_settings_set diversity_max_per_source=8 top_k=15 max_distance=0.85. Then lilbee_search 'A star pathfinding' (verbatim same query). Write a 2-line answer naming every A* class with one use case each, cite the file. No prose preamble."
Sleep 800ms
Enter
Sleep 90s

# Prompt 3: leave the env clean.
Type "lilbee_settings_reset diversity_max_per_source top_k max_distance"
Sleep 800ms
Enter
Sleep 15s

Screenshot "_out/mcp-self-tune.png"

Ctrl+C
