Source "_shared.tape"
Output "_out/mcp-self-tune.gif"

# opencode driving lilbee over MCP. PR-title beat: agent tunes lilbee
# (visibly, via lilbee_settings_set) to give a better answer against a
# dense pile of reference docs, then resets.
#
# Human-natural prompt -- a developer working on a Godot project asks
# lilbee to optimize for the reference docs before answering a broad
# question. The agent (qwen3.6-plus-free, drives the demo deterministically)
# reads SKILL.md workflow #5, then on its own runs:
#   1. lilbee_status                                      see baseline
#   2. lilbee_settings_list group="Retrieval"             enumerate knobs
#   3. lilbee_settings_set {top_k=50, diversity=20, ...}  the tuning step
#   4-6. lilbee_search "AStar3D" / "AStar2D" / "AStarGrid2D"
#   7. comprehensive answer citing all three classes
#   8. lilbee_settings_reset                              clean env
#
# Empirical run (lilbee 0.6.66b478, opencode 1.14, M1):
#   opencode boot                                      ~8s
#   self-tune + 3 searches + composed answer           ~60-75s
#   reset                                              ~5s
# Total real time: ~80-90s. PlaybackSpeed 1.5 -> ~55-60s GIF.
#
# Corpus is the godot 4 class reference (810 XMLs, ~3450 chunks),
# pre-indexed off-camera by demos/_prep.sh.

Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-self-tune"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

# The prompt: a developer who knows lilbee can be tuned, asking for
# shape-appropriate setup before a broad question. Reset framing at
# the end keeps the env clean for the next user.
Type "I'm researching Godot 4 pathfinding. Use lilbee_settings_set to tune retrieval for breadth before answering -- godot's reference is dense XML files, one class per file. Then walk me through every A* pathfinding class Godot 4 gives me with citations. Reset settings when you're done."
Sleep 800ms
Enter
Sleep 95s

Screenshot "_out/mcp-self-tune.png"

Ctrl+C
