Source "_shared.tape"
Output "_out/mcp-pdf-self-tune.gif"

# opencode driving lilbee over MCP. The upgrade-story demo for the
# rebalanced retrieval defaults: same human question, asked at the
# OLD defaults (top_k=8, max_distance=0.65) and then at the NEW
# shipped defaults (top_k=12, max_distance=0.75,
# diversity_max_per_source=5). The before/after lands on a real PDF
# manual where prose-shaped queries genuinely struggled at the old
# code-tuned defaults.
#
# Empirical baseline (lilbee 0.6.66b479, opencode 1.14,
# qwen3.6-plus-free, M1, /tmp/lilbee-cv-demo with cv-manual.pdf):
#   old defaults: ~2 pages, agent falls back to general automotive
#   knowledge because the manual coverage is thin
#   new defaults: ~10 entries across 5 pages, sourced entirely from
#   the manual
#
# Tape uses two sequential prompts so each answer lingers visibly
# instead of scrolling off behind the next agent turn.

Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-pdf-self-tune"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

# Turn 1: simulate the previous lilbee defaults so the upgrade story
# has a baseline. Human framing leads, but the upgrade-context
# instruction makes the comparison fair.
Type "First, restore lilbee's previous narrower defaults so we can see what shipped before: lilbee_settings_set top_k=8 max_distance=0.65 diversity_max_per_source=3. Now: hey, the steering wheel on my Crown Vic shakes when I'm driving on the highway. What could be wrong? Use lilbee_search at those settings. Tell me what the manual covers (and call out if you have to fall back to general knowledge)."
Sleep 800ms
Enter
Sleep 75s

# Turn 2: tune to the new shipped defaults and re-ask. End with the
# one-line tuning summary so the viewer sees exactly which knobs the
# agent moved.
Type "OK that was thin. Tune lilbee to the new shipped defaults: lilbee_settings_set top_k=12 max_distance=0.75 diversity_max_per_source=5. Same question. Tell me everything the manual covers. AFTER the answer, print a one-line summary of which settings changed and to what."
Sleep 800ms
Enter
Sleep 100s

Screenshot "_out/mcp-pdf-self-tune.png"

Ctrl+C
