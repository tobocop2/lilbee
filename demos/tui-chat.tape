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

# Centerpiece: a structured-extraction prompt that turns the instrument
# cluster warning-lights section into a markdown table. The cluster
# section is self-contained (one page block, no cross-refs to other
# fuse boxes), so retrieval anchors cleanly and Qwen3 4B can produce a
# full-coverage table with one Sources line.
#
# We DON'T do a warm-up sandwich here -- the warm-up `ping` + `/clear`
# was reducing answer quality (the kv reset seems to leave the model
# more conservative). Cold first turn is slower but produces the
# comprehensive 5-row table the manual tmux test verified. Hide elides
# the cold load + streaming so the GIF stays short.
Type "Provide a markdown table of the instrument cluster warning indicator lights from the Crown Victoria manual, with columns Symbol, Color, and Meaning. Include every warning light documented."
Sleep 600ms
Enter

Sleep 4s
Hide
Sleep 140s
Show
Sleep 10s

Screenshot demos/_out/tui-chat.png

# Buffer before /quit so the GIF doesn't end mid-stream.
Sleep 2s
Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
