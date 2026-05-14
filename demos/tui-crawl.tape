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

# Tasks -> Chat (cycle wraps). Re-enter INSERT for the question.
Type "]"
Sleep 1000ms
Type "i"
Sleep 400ms

# Single-fact question that maps to the article's opening section,
# avoiding sub-era ambiguity (multiple generations of the Caprice
# share the same article).
Type "What model years were the Chevrolet Caprice produced over?"
Sleep 600ms
Enter

# Streaming: ~90-120s on a cold Qwen3 4B. No Hide -- speed it up so the
# whole stream is on camera. PlaybackSpeed 3 over ~120s of real time
# produces ~40s of GIF; the answer streams visibly.
Set PlaybackSpeed 3
Sleep 130s
Set PlaybackSpeed 1.5

Sleep 3s
Ctrl+C
Sleep 500ms
