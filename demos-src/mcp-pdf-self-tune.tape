Source "_shared.tape"
Output "_out/mcp-pdf-self-tune.gif"

# Before/after for the rebalanced retrieval defaults.
#
# The opencode-pdf-self-tune dir is staged with cv-manual.pdf indexed
# and .lilbee/config.toml pinned to the OLD defaults (top_k=8,
# max_distance=0.65, diversity_max_per_source=3). The tape never
# types those values; the user just asks a human question and reacts
# to the answer. The agent decides on its own to widen retrieval via
# lilbee_settings_set.
#
# Empirical baseline (lilbee 0.6.66bXXX, opencode 1.14,
# qwen3.6-plus-free, M1, /tmp/lilbee-demo/opencode-pdf-self-tune):
#   turn 1 (old defaults, pinned off-camera): ~2 thin pages, agent
#   visibly falls back to general automotive knowledge.
#   turn 2 (agent self-tunes to wider settings, re-asks): ~10 manual
#   entries across 5 pages, every claim cited to the manual.

Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-pdf-self-tune"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

# Turn 1: just the human question. lilbee starts at the OLD defaults
# from .lilbee/config.toml (set off-camera). The agent calls
# lilbee_search at those settings, gets thin coverage, and falls back
# to general automotive knowledge.
Type "Hey, the steering wheel on my Crown Vic shakes when I'm driving on the highway. What could be wrong?"
Sleep 800ms
Enter
Sleep 75s

# Turn 2: human reacts to the thin answer. Doesn't dictate values;
# the agent reads the current settings, decides what to widen, calls
# lilbee_settings_set, and re-runs the search.
Type "That feels thin. Can you tune lilbee to pull in more from the manual, then try again?"
Sleep 800ms
Enter
Sleep 100s

Screenshot "_out/mcp-pdf-self-tune.png"

Ctrl+C
