Source demos/_shared.tape
Output demos/_out/mcp-godot.gif
# Record at 2x global. The full real-time recording is ~10 min on a
# MacBook M1 (boot + worker indexing 810 XML files + agent's lilbee_search
# pass + the cited code answer). 2x lands the GIF at ~5 min, which is
# long but readable: every frame is on camera, no Hide, no per-section
# fast-forward that drops chars below the eye's scan rate.
Set PlaybackSpeed 2.0

# opencode driving lilbee over MCP, full TUI, against Godot 4 class
# reference XML (810 files). Mirrors the prompt from the
# godot-level-generator benchmark (docs/benchmarks/godot-level-generator.md):
# the agent indexes the docs, then writes a procedural level generator
# with cited Godot 4 API calls.
#
# Demo dir prepped by demos/_prep.sh: AGENTS.md (with the Plan/Search/
# Write/Verify workflow), opencode.json, lilbee-worker subagent,
# lilbee-mcp skill, godot-classes copied in (~50MB of XML), and
# `lilbee init` already run.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
# 810 XML files index for ~5m 20s, comfortably past the default 300s
# worker-pool call timeout; bump to 15 minutes for headroom.
Env LILBEE_WORKER_POOL_CALL_TIMEOUT_S "900"

Type "cd /tmp/lilbee-demo/opencode-godot"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 10s

# Type the prompt and submit. Agent: load lilbee-mcp skill, plan,
# hand `lilbee_add` to the @lilbee-worker subagent, run Plan / Search /
# Write / Verify per AGENTS.md.
Type "Index ./godot-classes/ then write a procedural level generator that places wall and floor tiles and scatters collectibles using pathfinding. Cite the API references for every method you call."
Sleep 800ms
Enter

# Kickoff: skill load, planning, worker spawn (~30s).
Sleep 30s

# Indexing in progress. Drop into the subagent view (Ctrl+x Down) so the
# camera shows the worker's lilbee_add progress instead of the parent
# thread's idle "build" indicator. Hold there while the worker streams.
Ctrl+x
Sleep 100ms
Down
Sleep 5s

# Worker indexes ~810 XMLs at ~3 files/sec; 4-5 min real time.
Sleep 300s

# Pop back to the parent agent (Ctrl+x Up) so we see the cited search
# pass + code answer when the worker reports done.
Ctrl+x
Sleep 100ms
Up
Sleep 3s

# Agent's lilbee_search pass + Plan/Verify + final cited code answer.
Sleep 180s

Screenshot demos/_out/mcp-godot.png
Sleep 4s

Ctrl+C
Sleep 1s
