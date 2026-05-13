Source demos/_shared.tape
Output demos/_out/tui-chat.gif

# Chat with the Crown Victoria owner's manual. Models and index are
# pre-staged by demos/_prep.sh; LILBEE_DATA points at the prepared corpus.
#
# Qwen3 4B on M1 Pro takes ~60s for a cold first turn (Python sampler over
# Qwen's 152k-token vocab, py-spy confirmed) and ~10-15s warm. We do a
# silent warm-up turn inside `Hide` so the first on-camera answer is warm.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-chat"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Off-camera warm-up: load model + KV cache, then clear the conversation.
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

# Three single-fact retrieval lookups against the manual.
Type "what's my oil capacity?"
Sleep 400ms
Enter
Sleep 13s

Type "what tire pressure does it recommend?"
Sleep 400ms
Enter
Sleep 11s

Type "what fuel does it take?"
Sleep 400ms
Enter
Sleep 10s

Screenshot demos/_out/tui-chat.png

Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
