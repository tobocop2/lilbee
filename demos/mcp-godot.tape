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
#   3-5 lilbee_search calls + code generation      ~90-120s
#   Total wall clock                               ~7-8 min
#
# What viewers see, in order:
#   1. opencode boots, the prompt is typed and submitted
#   2. ~25s of the agent's initial activity: skill load, planning,
#      Lilbee-Worker subagent kickoff with the lilbee_add tool call
#   3. Hide elides ~5 minutes of bulk indexing (the dead middle)
#   4. ~120s of post-indexing work: lilbee_search calls per Godot
#      class, the verify pass, the code answer streaming in
#   5. Final answer with file:line citations
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

# Show the kickoff: skill load, planning, and the Lilbee-Worker
# delegation are all visible on camera before we Hide.
Sleep 25s

# Hide elides ~5 minutes of indexing dead time. We end the Hide
# window EARLIER than the indexing actually finishes so Show resumes
# while the worker is wrapping up + the agent is starting its
# lilbee_search calls; viewers see real on-camera tool activity.
Hide
Sleep 300s
Show

# Post-indexing: lilbee_search calls, the verify pass, the code answer
# streaming in. This is the most useful on-camera content.
Sleep 120s

Screenshot demos/_out/mcp-godot.png
Sleep 5s

Ctrl+C
Sleep 1s
