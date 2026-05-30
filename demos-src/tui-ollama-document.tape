Source "_shared.tape"
Output "_out/tui-ollama-document.gif"

# Talk to the Crown Victoria manual with a chat model served by OLLAMA.
# No native lilbee models are present (LILBEE_MODELS_DIR is an empty dir),
# so the catalog shows only the Ollama-served model: the external manager
# is clearly what's answering. Index is pre-staged by _prep.sh.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-ollama-document"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/tui-ollama-document-models"
Env LILBEE_LLM_PROVIDER "remote"
Env LILBEE_REMOTE_BASE_URL "http://localhost:11434"
Env LILBEE_CHAT_MODEL "ollama/qwen3:0.6b"
Env LILBEE_EMBEDDING_MODEL "ollama/nomic-embed-text"
Env LILBEE_THEME "rose-pine"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"
Env LILBEE_TOP_K "20"
Env LILBEE_NUM_CTX "16384"

# Provenance: the model lives in Ollama.
Type "ollama ls"
Sleep 400ms
Enter
Sleep 2800ms
Type "clear"
Sleep 200ms
Enter
Sleep 400ms

# Launch the TUI. The model bar shows the active chat + embed models
# (qwen3:0.6b / nomic-embed-text) -- the exact models just listed by
# `ollama ls`, so the viewer sees lilbee is answering from Ollama. The
# native models dir is empty, so nothing native is in play.
Type "lilbee"
Sleep 300ms
Enter
Sleep 3500ms

# Centerpiece: a real diagnostic question whose answer lives in the
# Roadside Emergencies chapter (fuel pump shut-off switch, page 192).
Type "I was in a fender-bender and now my Crown Victoria cranks but won't start. What safety feature likely cut the fuel, and how do I reset it?"
Sleep 600ms
Enter

Sleep 3s
Hide
Sleep 12s
Show
Sleep 11s

Screenshot "_out/tui-ollama-document.png"

# Buffer so the GIF doesn't end mid-stream.
Sleep 2s
Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
