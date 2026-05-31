Source "_shared.tape"
Output "_out/tui-ollama-document.gif"

# Talk to the Crown Victoria manual with a chat model served by OLLAMA.
# The native models dir is empty, so the catalog shows only the
# Ollama-served model (labeled "ollama"). We index the manual on camera,
# then ask a diagnostic question and get a cited answer.
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

# Launch the TUI.
Type "lilbee"
Sleep 300ms
Enter
Sleep 3500ms

# Inspect the catalog: the only installed chat model is the one served by
# Ollama, labeled "ollama". No native models are present.
Type "/models"
Sleep 600ms
Enter
Sleep 4s
Type "q"
Sleep 1500ms

# Index the Crown Victoria manual on camera.
Set TypingSpeed 55ms
Type "/add ~/Downloads/cv-manual.pdf"
Set TypingSpeed 35ms
Sleep 1800ms
Enter
Sleep 1500ms

# Watch the ingest fill in the Task Center (Escape -> normal mode, then t).
Escape
Sleep 400ms
Type "t"
Sleep 4s
Hide
Sleep 16s
Show
Sleep 3s

# Back to chat: [ walks Tasks -> Settings -> Status -> Catalog -> Chat.
Type "["
Sleep 300ms
Type "["
Sleep 300ms
Type "["
Sleep 300ms
Type "["
Sleep 700ms
Type "i"
Sleep 300ms

# Centerpiece: a real diagnostic question whose answer lives in the
# Roadside Emergencies chapter (fuel pump shut-off switch, page 192).
Type "I was in a fender-bender and now my Crown Victoria cranks but won't start. What safety feature likely cut the fuel, and how do I reset it?"
Sleep 600ms
Enter
Sleep 3s
Hide
Sleep 13s
Show
Sleep 11s

Screenshot "_out/tui-ollama-document.png"

Sleep 2s
Escape
Sleep 200ms
Type "/quit"
Sleep 400ms
Enter
Sleep 500ms
