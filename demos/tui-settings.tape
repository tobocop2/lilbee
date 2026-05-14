Source demos/_shared.tape
Output demos/_out/tui-settings.gif

# Settings tour: cycle every sub-tab via `>` (next pane, priority binding).
# With the crawler installed there are 8 panes: Models, Ingest, Generation,
# Retrieval, Display, Crawling, API-Keys, System. Without crawl4ai the
# Crawling pane is gated out; the tape still works because `>` just cycles
# through whatever panes are present.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-settings"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

Escape
Sleep 1s

# Open Settings.
Type "/settings"
Sleep 400ms
Enter
Sleep 2s

# Cycle through all the panes. Seven `>` covers an 8-pane install; the last
# one wraps back to Models. Tape stays correct on a 6-pane install too --
# the extra `>` presses are no-ops past the last pane.
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms
Type ">"
Sleep 1500ms

# From System, four `<` presses lands on Retrieval (longest pane, best
# screenshot): API-Keys -> Crawling -> Display -> Retrieval.
Type "<"
Sleep 500ms
Type "<"
Sleep 500ms
Type "<"
Sleep 500ms
Type "<"
Sleep 500ms

Screenshot demos/_out/tui-settings.png

# Demonstrate the model-select dropdown on the Models pane: three more `<`
# back: Generation -> Ingest -> Models.
Type "<"
Sleep 500ms
Type "<"
Sleep 500ms
Type "<"
Sleep 500ms
Tab
Sleep 400ms
Enter
Sleep 1500ms
Escape
Sleep 800ms

Type "q"
Sleep 800ms

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
