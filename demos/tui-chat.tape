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
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
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

# Three single-fact retrieval lookups against the manual. While Qwen3 4B
# is streaming, the chat input is disabled and any queued Enter is dropped
# (the next Type's text appends to whatever's in the field). Sleeps below
# allow each warm answer to fully complete before the next question types.
Type "what's my oil capacity?"
Sleep 400ms
Enter
Sleep 22s

Type "what tire pressure does it recommend?"
Sleep 400ms
Enter
Sleep 22s

Type "what fuel does it take?"
Sleep 400ms
Enter
Sleep 22s

Screenshot demos/_out/tui-chat.png

# Buffer before /quit so the GIF doesn't end mid-stream.
Sleep 2s
Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
