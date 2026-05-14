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
# lilbee-mcp skill, and `lilbee init` already run.
#
# Empirical timings (lilbee 0.6.66b473, MacBook Air M1):
#   opencode boot                                  ~8s
#   prompt typed + agent kickoff visible           ~20s
#   lilbee_add 810 XML files via lilbee-worker     ~5 min  (3449 chunks)
#   lilbee_search calls + code answer streaming    ~90s
#   Real wall clock on M1                          ~6 min 40s
#
# NO Hide blocks: every frame is on camera. PlaybackSpeed varies per
# section so the indexing bar gets fast-forwarded (8x) while the
# kickoff, the live searches, and the cited answer play at near-real
# speed (1.5x). Final GIF lands around two minutes.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-godot"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

# Type the prompt and submit. Agent: load lilbee-mcp skill, plan,
# delegate lilbee_add to lilbee-worker, then run Plan/Search/Write/
# Verify per AGENTS.md.
Type "Index ./godot-classes/ then write a procedural level generator that places wall and floor tiles and scatters collectibles using pathfinding. Cite the API references for every method you call."
Sleep 600ms
Enter

# Kickoff phase: skill load, planning, lilbee-worker spawn -- keep at
# the shared playback speed (1.5x).
Sleep 25s

# Indexing phase: the worker streams its lilbee_add progress for ~5
# minutes. No Hide -- speed playback up to 8x so the bar fills on
# camera, fast.
Set PlaybackSpeed 8
Sleep 300s

# Back to demo speed for the agent's lilbee_search calls, the file
# write, and the final cited answer.
Set PlaybackSpeed 1.5
Sleep 90s

Screenshot demos/_out/mcp-godot.png
Sleep 5s

Ctrl+C
Sleep 1s
