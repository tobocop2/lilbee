Source demos/_shared.tape
Output demos/_out/tui-crawl.gif

# Crawl a Wikipedia page using the /crawl modal (depth + max-pages picker
# instead of inline flags), watch it complete in the Task Center, then
# ask Qwen3 4B a single-fact question and wait until the cited answer
# fully renders before cutting.
#
# Empirical timings on M1 (lilbee 0.6.66b473, fresh data dir, cold model):
#   lilbee boot + wizard dismiss             ~4s
#   /crawl modal (Type + Tabs + Enter)       ~14s
#   crawl execute (visible in Tasks)         ~12s
#   Tasks -> Chat (`]` + `i` + question)     ~5s
#   cold Qwen3 4B first-token wait           ~134s   <-- the loading bar
#   answer stream + citations                ~10s
# Total real time: ~3 min 10s. Global PlaybackSpeed 3.5 -> ~55s GIF;
# the loading bar is ~38s instead of 134s, the answer reads at ~3s/token
# which is still legible.
Set PlaybackSpeed 3.5

Env LILBEE_DATA "/tmp/lilbee-demo/tui-crawl"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"
Env OLLAMA_HOST "http://127.0.0.1:65535"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# One Escape: dismiss the setup wizard. Chat input stays focused in
# INSERT, so `/crawl` types straight into it and Enter dispatches.
Escape
Sleep 1s

# /crawl with no arg opens the modal. URL input auto-focuses.
Type "/crawl"
Sleep 400ms
Enter
Sleep 1500ms

# Type the URL.
Type "https://en.wikipedia.org/wiki/Chevrolet_Caprice"
Sleep 1500ms

# Tab through the modal: URL -> Recursive checkbox -> Advanced collapsible.
Tab
Sleep 800ms
Tab
Sleep 600ms
# Expand Advanced.
Enter
Sleep 1s
# Tab into Depth, set it to 0 (per the crawler contract: depth=0 ->
# crawl the seed URL only).
Tab
Sleep 400ms
Type "0"
Sleep 800ms
# Tab into Max pages, leave blank (depth=0 fully constrains it).
Tab
Sleep 800ms
# Tab to the Crawl button, fire it.
Tab
Sleep 500ms
Enter
# Generous wait for modal dismiss + crawl kickoff.
Sleep 4s

# Drop chat input from INSERT to NORMAL so `t` fires the Tasks binding.
Escape
Sleep 800ms

# `t` opens Tasks. The crawl progress is visible there.
Type "t"
Sleep 1500ms

# Hold long enough for the single-page crawl to finish (~10s).
Sleep 12s

# Tasks -> Chat wraps. Chat lands in NORMAL after the screen-nav cycle
# (we Escaped earlier to drop INSERT for the `t` binding). Without an
# explicit `i` here, the prompt's leading `W`/`h` get eaten as unbound
# NORMAL-mode keys.
Type "]"
Sleep 1000ms
Type "i"
Sleep 400ms

# Single-fact question whose answer is one specific sentence in the
# article. The previous prompt asked for cross-year engine specs and
# Qwen3 4B confidently hallucinated (claimed the 427 was offered
# 1977-1990; it ended in 1969). This prompt verifies cleanly in tmux
# against the indexed page: "1986" with the article cited.
Type "When was the 9C1 Caprice police package introduced?"
Sleep 600ms
Enter

# Cold Qwen3 4B: ~134s loading bar before first token, ~10s of streaming.
# Single combined wait so the global PlaybackSpeed compresses the whole
# block uniformly -- per-section Set PlaybackSpeed gets folded into one
# global setpts filter by ffmpeg, so it does not act as a local speedup.
Sleep 150s

# Capture the cited answer as the still, AFTER the stream has settled.
Screenshot demos/_out/tui-crawl.png
Sleep 1s

Ctrl+C
Sleep 500ms
