Source "_shared.tape"
Output "_out/mcp-code-self-tune.gif"
Output "_out/mcp-code-self-tune.mp4"

# Self-tune demo against lilbee's own source. Two turns:
#
#   1. User asks a natural engineering question about how
#      lilbee_search works end to end. The agent at the OLD
#      pre-rebalance defaults (top_k=8, max_distance=0.65,
#      diversity_max_per_source=3, pinned off-camera in
#      .lilbee/config.toml) calls lilbee_search once and answers
#      with a structured ten-step outline. Each step names a real
#      lilbee method (Embedder, _store.search, bm25_probe,
#      _llm_expand, etc.), but the answer is summary-level: no
#      file paths, no line numbers, one-line example per step.
#
#   2. User asks for the actual code. The agent reaches for
#      lilbee_settings_set to widen top_k, re-runs lilbee_search
#      against the richer pool, then answers with full function
#      bodies in fenced code blocks, each followed by a
#      file:L<start>-L<end> citation. The agent even mentions
#      lilbee_settings_set in the body of its answer (the
#      knob it just moved appears as a real config field in
#      mcp_server.py:L51-76).
#
# Empirical baseline (Qwen3-8B Q4_K_M, M1):
#   turn 1 (OLD defaults, narrow):     ~5-6 min, ~10 steps,
#                                       method names + one-line
#                                       snippets, no file:line.
#   turn 2 (after settings_set):       ~5-6 min, multi-line
#                                       function bodies with
#                                       file:L<start>-L<end>
#                                       citations.
#
# Prompt note on turn 2: the cue mentions lilbee_settings_set by
# name because Qwen3-8B does not generalise "be exhaustive" or
# "tune retrieval" into the right meta-tool call. On a larger
# model with bigger context, the more natural phrasing
#
#   "I want the actual code with file paths."
#
# is enough; the agent picks the settings move from the skill's
# trigger map. For the local-8B baseline we name the tool to keep
# the recording deterministic.

Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"

Type "cd /tmp/lilbee-demo/opencode-code-self-tune"
Sleep 400ms
Enter
Sleep 600ms

Type "lilbee launch opencode"
Sleep 400ms
Enter
Sleep 12s

# Turn 1: natural question. OLD defaults pinned in
# .lilbee/config.toml so the search hits a narrow chunk window;
# the agent answers with method names but no file:line.
Type "How does lilbee_search work end to end? Walk me through embedding, retrieval, and result formatting."
Sleep 800ms
Enter
Sleep 8s
Hide
Sleep 320s
Show
Sleep 3s
PageUp
Sleep 2500ms
PageUp
Sleep 2500ms
PageDown
Sleep 1500ms
PageDown
Sleep 1500ms

# Turn 2: user wants the source. Names lilbee_settings_set
# explicitly so the 8B reliably reaches for the tune-then-
# re-search loop (see prompt note in the header).
Type "That's a useful outline, but I want the actual code. Use lilbee_settings_set to widen retrieval, then show me the function body and the file:line for each step."
Sleep 800ms
Enter
Sleep 8s
Hide
Sleep 340s
Show
Sleep 3s
PageUp
Sleep 2500ms
PageUp
Sleep 2500ms
PageUp
Sleep 2500ms
PageDown
Sleep 1500ms
PageDown
Sleep 1500ms
PageDown
Sleep 1500ms

Screenshot "_out/mcp-code-self-tune.png"

Ctrl+C
