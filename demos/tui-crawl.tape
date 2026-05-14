Source demos/_shared.tape
Output demos/_out/tui-crawl.gif

# Crawl a Wikipedia page using the /crawl modal (depth + max-pages picker
# instead of inline flags), then ask Qwen3 4B a sub-era question against
# the indexed page with cited answers.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-crawl"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

Escape
Sleep 1s

# /crawl with NO arg opens the modal dialog. AUTO_FOCUS lands on the URL
# input.
Type "/crawl"
Sleep 400ms
Enter
Sleep 1500ms

# Type the URL into the focused input. Don't press Enter yet -- want to
# show the depth/max-pages settings being toggled.
Type "https://en.wikipedia.org/wiki/Chevrolet_Caprice"
Sleep 1500ms

# Tab through the modal: URL -> Recursive checkbox -> Advanced collapsible.
Tab
Sleep 800ms
Tab
Sleep 600ms
# Expand the Advanced collapsible (Enter on a focused Collapsible toggles).
Enter
Sleep 800ms
# Tab into Depth, set it to 0 (just this page).
Tab
Sleep 400ms
Type "0"
Sleep 800ms
# Tab into Max pages, set it to 1.
Tab
Sleep 400ms
Type "1"
Sleep 800ms
# Tab to Submit, fire it.
Tab
Sleep 400ms
Enter
Sleep 1500ms

# After the modal closes, focus is back on the chat input (INSERT).
# Escape -> NORMAL so `]` works as a screen-nav binding.
Escape
Sleep 400ms

# Walk to the Task Center so the crawl progress is visible (4 `]`
# from Chat: Catalog -> Status -> Settings -> Tasks).
Type "]"
Sleep 400ms
Type "]"
Sleep 400ms
Type "]"
Sleep 400ms
Type "]"
Sleep 30s

Screenshot demos/_out/tui-crawl.png

# Back to Chat (one `]` from Tasks wraps to Chat). Then `i` to drop
# back into INSERT for the next question.
Type "]"
Sleep 800ms
Type "i"
Sleep 300ms

# The on-camera question, narrowed to one sub-era so retrieval lands clean.
# Cold first turn (no warm-up) inside Hide so the GIF stays short.
Type "What engines were available in the 1986-1990 Chevrolet Caprice?"
Sleep 600ms
Enter

Sleep 4s
Hide
Sleep 100s
Show
Sleep 5s

# Buffer so the GIF doesn't end mid-stream.
Sleep 2s
Escape
Sleep 200ms
Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
