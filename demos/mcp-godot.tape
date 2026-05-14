Source demos/_shared.tape
Output demos/_out/mcp-godot.gif

# opencode driving lilbee over MCP, full TUI, against Godot 4 class
# reference XML (810 files). Mirrors the prompt from the
# godot-level-generator benchmark (docs/benchmarks/godot-level-generator.md):
# the agent indexes the docs, then writes a procedural level generator
# with cited Godot 4 API calls.
#
# Demo dir prepped by demos/_prep.sh: symlink at godot-classes/ pointing
# at ~/projects/godot/doc/classes, plus AGENTS.md (with the Plan/Search/
# Write/Verify workflow), opencode.json, lilbee-worker subagent,
# lilbee-mcp skill, and `lilbee init` already run. No pre-indexing --
# the agent does the add via lilbee-worker.
#
# Empirical timings (lilbee 0.6.66b472 with the embed pipe-routing fix):
#   opencode boot                                  ~8s
#   lilbee_add 810 XML files via lilbee-worker     ~5 min 20s (3449 chunks)
#   3-5 lilbee_search calls + code generation      ~60-90s
#   Total wall clock                               ~7 min
#
# The slow indexing minute is wrapped in Hide so the GIF shows the
# kickoff, jumps forward, and lands on the cited final answer.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-godot"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

# One turn: index + write the level generator. The agent loads the
# lilbee-mcp skill, delegates lilbee_add to lilbee-worker, then runs
# Plan / Search / Write / Verify per the workflow in AGENTS.md.
Type "Index ./godot-classes/ then write a procedural level generator that places wall and floor tiles and scatters collectibles using pathfinding. Cite the API references for every method you call."
Sleep 600ms
Enter

# Show the kickoff: skill load + Lilbee-Worker delegation visible on
# camera for ~10s. Then elide the slow indexing minutes.
Sleep 10s
Hide
Sleep 380s
Show
Sleep 60s

Screenshot demos/_out/mcp-godot.png
Sleep 3s

Ctrl+C
Sleep 1s
