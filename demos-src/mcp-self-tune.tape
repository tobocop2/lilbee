Source "_shared.tape"
Output "_out/mcp-self-tune.gif"

# Two-turn natural conversation. The visible point is the named knobs
# that get moved and the resulting answer delta.
#
# Turn 1: user wants to evaluate lilbee defaults. Agent runs ONE
# lilbee_search at defaults, returns AStar3D only and flags that
# AStar2D didn't appear.
#
# Turn 2: user reacts, asks the agent to tune. Prompt forces the
# agent to (a) name each key it's moving with baseline/new value
# BEFORE calling lilbee_settings_set, (b) print a one-line tuning
# summary at the end, so what was tuned is visible in the chat
# stream and not buried in the collapsed tool JSON.
#
# Empirical run (lilbee 0.6.66b478, opencode 1.14, qwen3.6-plus-free,
# M1):
#   opencode boot                                      ~8s
#   turn 1 (single search + narrow answer)             ~15-20s
#   user types turn 2                                  ~10s
#   turn 2 (knob list + settings_set + search +        ~45-55s
#           tuned answer + reset + summary line)
# Total real time: ~85-95s. PlaybackSpeed 1.5 -> ~60s GIF.
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

# Turn 2: explicit knob list + summary so the tuning is visible in
# the chat stream, not buried in the collapsed tool JSON.
Type "That's narrow. Tune lilbee to widen retrieval. BEFORE calling lilbee_settings_set, write a short numbered list naming each key you're moving, its baseline value, the new value, and one phrase on why. Then call lilbee_settings_set, run the same lilbee_search, and answer. End with a one-line tuning summary like 'top_k 8 -> 40, max_distance 0.65 -> 0.85'. Reset settings when done."
Sleep 800ms
Enter
Sleep 60s

Screenshot "_out/mcp-self-tune.png"

Ctrl+C
