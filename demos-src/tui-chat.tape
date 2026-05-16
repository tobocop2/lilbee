Source _shared.tape
Output _out/tui-chat.gif

# Chat with the Crown Victoria owner's manual. Models and index are
# pre-staged by demos/_prep.sh; LILBEE_DATA points at the prepared corpus.
#
# Qwen3 4B on M1 Pro takes ~60s for a cold first turn (Python sampler over
# Qwen's 152k-token vocab, py-spy confirmed) and ~10-15s warm. We do a
# silent warm-up turn inside `Hide` so the first on-camera answer is warm.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-chat"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_CHAT_MODEL "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"
# Wider retrieval and bigger context for the fuse-table extraction. The
# cv-manual has three different fuse boxes (instrument panel, BJB, smart
# junction box); bumping top_k gives the model enough chunks to cover
# the full panel without missing rows.
Env LILBEE_TOP_K "20"
Env LILBEE_NUM_CTX "16384"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Centerpiece: a single-fact question whose answer is a clean cited
# paragraph with a quoted passage and a single page reference. Tmux
# tested: 20s end-to-end, deterministic answer (oil capacity 6.0
# quarts, page 264). Hide covers the first-token wait so the visible
# stream lands quickly on the cited answer.
Type "What's my oil capacity?"
Sleep 600ms
Enter

Sleep 3s
Hide
Sleep 12s
Show
Sleep 10s

Screenshot _out/tui-chat.png

# Buffer before /quit so the GIF doesn't end mid-stream.
Sleep 2s
Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
