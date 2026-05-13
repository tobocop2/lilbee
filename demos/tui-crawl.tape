Source demos/_shared.tape
Output demos/_out/tui-crawl.gif

# Crawl a Wikipedia page, then ask Qwen3 4B a single-era question against
# the indexed page with citations.
#
# Measured timings:
#   /crawl <url> --depth 0 --max-pages 1     ~30s end-to-end (Playwright +
#                                            kreuzberg extract + embed +
#                                            LanceDB write)
#   Cold first chat turn after warm-up        ~14s
Env LILBEE_DATA "/tmp/lilbee-demo/tui-crawl"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

Escape
Sleep 1s

# Submit the crawl. depth=0 + max-pages=1 = just the page itself, no link
# expansion.
Type "/crawl https://en.wikipedia.org/wiki/Chevrolet_Caprice --depth 0 --max-pages 1"
Sleep 600ms
Enter
Sleep 1s

# Jump to the Task Center so the crawl progress is visible on camera.
Escape
Sleep 300ms
Type "t"
Sleep 30s

Screenshot demos/_out/tui-crawl.png

# Back to Chat. Off-camera warm-up so the on-camera answer is warm.
Type "["
Sleep 1s

Hide
Type "ping"
Sleep 400ms
Enter
Sleep 45s
Type "/clear"
Sleep 400ms
Enter
Sleep 1s
Show

# The on-camera question, narrowed to one sub-era so retrieval lands clean.
Type "What engines were available in the 1986-1990 Chevrolet Caprice?"
Sleep 400ms
Enter
Sleep 14s

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
