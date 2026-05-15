Source demos/_shared.tape
Output demos/_out/tui-crawl.gif

# Crawl a Wikipedia page using the /crawl modal (depth + max-pages picker
# instead of inline flags), watch it complete in the Task Center, then
# ask Qwen3 4B a single-fact question against the indexed page with a
# cited answer.
#
# Empirical timings (lilbee 0.6.66b473):
#   Single-page crawl + parse + index    ~8-12s
#   Qwen3 4B cold first turn             ~90-120s streaming
#
# Flow on camera:
#   1. /crawl Enter   ->  modal opens
#   2. URL typed; Tab through Recursive -> Advanced; expand Advanced
#   3. Depth=1, Max=1; Tab to Crawl button; Enter submits
#   4. Modal closes -> chat in INSERT -> Escape to NORMAL -> `t` opens
#      the Task Center where the crawl progress bar is visible
#   5. `]` wraps from Tasks back to Chat; `i` re-enters INSERT
#   6. Question is typed; streaming answer with [N] citations
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

# Capture the Task Center mid-crawl as the still.
Screenshot demos/_out/tui-crawl.png

# Hold long enough for the single-page crawl to finish (~10s).
Sleep 12s

# Tasks -> Chat wraps. Chat lands in NORMAL after the screen-nav cycle
# (we Escaped earlier to drop INSERT for the `t` binding). Without an
# explicit ``i`` here, the prompt's leading ``W``/``h`` get eaten as
# unbound NORMAL-mode keys and ``a`` is the vim-style enter-INSERT
# binding -- the question started landing as ``t displacement ...``.
Type "]"
Sleep 1000ms
Type "i"
Sleep 400ms

# Sub-era question that maps cleanly to a single section in the article.
# Verified in tmux: returns the 305 cu in V8 with cite-backed details
# from 1977-1990 (RPO LO3 fuel-injection update in 1989).
Type "What displacement V8 engines were offered in the Caprice between 1977 and 1990?"
Sleep 600ms
Enter

# Streaming: ~90-120s on a cold Qwen3 4B. PlaybackSpeed 3 over ~120s of
# real time keeps the answer on camera but drops gif length to ~40s.
Set PlaybackSpeed 3
Sleep 110s
Set PlaybackSpeed 1.5

# Brief tail so the cited answer is visible at the end -- not the long
# blinking-cursor idle the prior render trailed off into.
Sleep 1s
Ctrl+C
Sleep 500ms
