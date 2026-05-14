Source demos/_shared.tape
Output demos/_out/mcp-godot.gif
# Record at 1x (override shared.tape's 1.5x default) so a follow-up
# `demos/_postprocess_speed.sh` pass can apply per-section speeds: the
# kickoff and the final cited answer play near real speed, the
# 5-minute indexing window is fast-forwarded. VHS can't vary speed
# mid-tape on its own; it always folds the final PlaybackSpeed into
# one global ffmpeg `setpts=PTS/n` filter.
Set PlaybackSpeed 1.0

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
# 810 XML files index for ~5m 20s, brushing right up against the default
# 300s worker-pool call timeout; bump to 10 minutes so the demo doesn't
# time out before lilbee_add (or a `sync` fallback the agent might pick)
# completes.
Env LILBEE_WORKER_POOL_CALL_TIMEOUT_S "600"

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

# Real time: opencode kickoff (~25s) + worker indexing the 810 XML
# files (~5-7 min depending on MiniMax speed) + post-indexing search
# and cited code generation (~90-120s). Total wall is ~7-9 min on M1.
# All on camera at the global 4x playback speed; the gif lands at
# ~2 min and viewers can pause for the parts they want to read.
Sleep 30s
Sleep 420s
Sleep 120s

Screenshot demos/_out/mcp-godot.png
Sleep 5s

Ctrl+C
Sleep 1s
