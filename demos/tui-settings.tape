Source demos/_shared.tape
Output demos/_out/tui-settings.gif

# Settings tour: navigate to the screen via `]` (priority binding) and
# use Tab to walk through the fields on a couple of representative
# panes. Less jumpy than cycling all 8 panes at once.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-settings"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Two Escapes: first dismisses the wizard, second drops chat into NORMAL
# (Chat starts in INSERT mode after wizard dismiss). Without the second
# Escape, the next `]` types as a literal in the chat input.
Escape
Sleep 500ms
Escape
Sleep 500ms

# Walk to Settings: Chat -> Catalog -> Status -> Settings. 3 `]` presses.
Type "]"
Sleep 800ms
Type "]"
Sleep 800ms
Type "]"
Sleep 1500ms

# Lands on Models pane (the first one). Tab through a few fields.
Tab
Sleep 1500ms
Tab
Sleep 1500ms
Tab
Sleep 1500ms

# Move to the Generation pane (next pane via `>`).
Type ">"
Sleep 1500ms

# Tab through the Generation fields (temperature, top_p, top_k_sampling,
# repeat_penalty, num_ctx ...).
Tab
Sleep 1500ms
Tab
Sleep 1500ms
Tab
Sleep 1500ms
Tab
Sleep 1500ms

# One more pane: Retrieval (longest pane, best screenshot).
Type ">"
Sleep 2s

Screenshot demos/_out/tui-settings.png

# Tab through the Retrieval fields too.
Tab
Sleep 1500ms
Tab
Sleep 1500ms
Tab
Sleep 1500ms

Type "q"
Sleep 800ms

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
