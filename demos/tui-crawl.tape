Source demos/_shared.tape
Output demos/_out/tui-crawl.gif

# Crawl a Wikipedia page using the /crawl modal (depth + max-pages picker
# instead of inline flags), then ask Qwen3 4B a sub-era question against
# the indexed page with cited answers.
#
# Empirically verified in tmux: two Escapes after launch (wizard dismiss
# + chat to NORMAL), then `/crawl` Enter opens the modal with the URL
# input auto-focused. Tab cycles URL -> Recursive -> Advanced. Enter on
# Advanced expands it. Tab into Depth/Max, fill, Tab to Crawl button,
# Enter. After submit the modal dismisses, the toast bar in chat shows
# crawl progress, then the toast clears and we can ask a question
# against the indexed page.
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
Sleep 3s

# Two Escapes: dismiss the setup wizard, then drop chat input to NORMAL.
Escape
Sleep 700ms
Escape
Sleep 700ms

# /crawl with no arg opens the modal. URL input auto-focuses.
Type "/crawl"
Sleep 500ms
Enter
Sleep 1800ms

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
# Tab into Depth, set it to 1 (the seed page only -- depth must be a
# positive int; 0 fails validation).
Tab
Sleep 400ms
Type "1"
Sleep 800ms
# Tab into Max pages, set it to 1.
Tab
Sleep 400ms
Type "1"
Sleep 800ms
# Tab to the Crawl button, fire it.
Tab
Sleep 500ms
Enter
Sleep 3s

# Modal closes. Crawl runs in the background (a toast bar at the bottom
# of chat shows the progress). It typically completes in ~8-12s on a
# single Wikipedia page. Wait for the toast to clear, then capture the
# stable state.
Sleep 14s
Screenshot demos/_out/tui-crawl.png

# Click on the chat input to refocus it in INSERT mode (lilbee promotes
# to INSERT on chat-input click; cleaner than wrestling with Escape
# state after the modal dismiss).
Type "What engines were available in the 1986-1990 Chevrolet Caprice?"
Sleep 600ms
Enter

# Streaming: ~90-120s on a cold Qwen3 4B. Hide the dead middle.
Sleep 5s
Hide
Sleep 110s
Show
Sleep 5s

Sleep 2s
Ctrl+C
Sleep 500ms
