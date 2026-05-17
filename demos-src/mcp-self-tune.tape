Source "_shared.tape"
Output "_out/mcp-self-tune.gif"

# opencode driving lilbee over MCP, full TUI. First marquee demo for the
# lilbee_settings_* MCP tools (PR #265): an agent answers a code-Q&A
# question, looks at its own retrieval, decides three knobs to move,
# re-answers, then resets. The corpus is pre-indexed off-camera by
# demos/_prep.sh -- this tape opens straight into the tuning loop.
#
# Demo dir prepped by demos/_prep.sh: AGENTS.md + opencode.json (pinned
# to opencode/qwen3.6-plus-free), lilbee-worker subagent, lilbee-mcp
# skill, four-file corpus (AGENTS.md, agent-integration.md,
# app_settings.py, app_settings_map.py) already indexed.
#
# Empirical timing (lilbee 0.6.66b476, opencode 1.14, M1):
#   opencode boot                                    ~8s
#   prompt + settings_list baseline + first search   ~25-35s
#   first cited answer + settings_set + re-search    ~30-40s
#   re-answer + settings_reset                       ~15-20s
# Total real time: ~90-110s. Hide the slow middle (settings_list +
# search + answer + set), Show the re-answer + reset for the visible
# tuning beat. PlaybackSpeed 1.5 -> ~70s GIF.

Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-self-tune"
Sleep 400ms
Enter
Sleep 600ms

Type "opencode"
Sleep 400ms
Enter
Sleep 8s

# Tight prompt: hand the agent the answer location and the exact knobs
# so the on-camera beat is the tuning loop, not the planning. Three
# numbered steps map 1:1 to settings_list / settings_set / settings_reset
# with a single search+answer cycle between set and reset.
Type "How does lilbee stop an MCP agent from reading API keys back? Cite app_settings.py with file:line. Then run: (1) settings_list Retrieval, (2) settings_set diversity_max_per_source=8 max_distance=0.8 top_k=15 (one phrase each), (3) re-answer with new settings, (4) settings_reset those keys."
Sleep 800ms
Enter
Sleep 6s

# Hide the planning + first search + first answer + settings_set.
# The marquee beat is the second answer (looser retrieval, fresh cites)
# plus the reset, which we keep visible.
Hide
Sleep 70s
Show
Sleep 18s

Screenshot "_out/mcp-self-tune.png"

Ctrl+C
