Source demos/_shared.tape
Output demos/_out/mcp-godot.gif

# opencode driving lilbee over MCP, full TUI, on Godot 4.6 class reference XML.
# Mirrors the prompt from the godot-level-generator benchmark
# (docs/benchmarks/godot-level-generator.md) but skips the "without lilbee"
# baseline -- this just demonstrates the with-lilbee path end-to-end.
#
# Demo dir prepped by demos/_prep.sh: a symlink to ~/projects/godot/doc/classes
# (810 XML class reference files), AGENTS.md, opencode.json, lilbee-worker
# subagent, lilbee-mcp skill, and `lilbee init` already run. No pre-indexing
# -- the agent does the add via the lilbee-worker subagent.
#
# Empirical timings:
#   opencode boot                                ~8s
#   lilbee_add ./godot-classes (810 XML files)   ~5-8 min (the embedding
#                                                model on M1 Pro embeds
#                                                ~3-5 chunks/sec; ~1500
#                                                chunks total)
#   ~3 lilbee_search calls + answer generation   ~60s
#
# The slow indexing minutes are wrapped in Hide so the GIF shows kickoff,
# the cited answer, and ends clean.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-godot"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

Type "Use lilbee to index ./godot-classes (Godot 4 class reference XML), then write a procedural level generator that places wall and floor tiles and scatters collectibles using pathfinding. Cite the API references for every method you call."
Sleep 600ms
Enter

# Show the kickoff for a few seconds (skill load + Lilbee-Worker delegation),
# then elide the long indexing wait, then show the cited answer and the
# generated code.
Sleep 8s
Hide
Sleep 480s
Show
Sleep 30s

Screenshot demos/_out/mcp-godot.png
Sleep 3s

Ctrl+C
Sleep 1s
