Source demos/_shared.tape
Output demos/_out/tui-setup.gif

# Fresh-slate first run. Wizard appears with no installed models; the user
# picks the curated defaults (Qwen3 0.6B, ~0.5 GB) and Nomic v1.5 (~84 MB),
# then watches the Task Center fill. Tiny models keep the pulls short
# enough to show live -- no Hide/Show eliding.
#
# The wizard's curated FEATURED_CHAT list is the only thing it shows here;
# SmolLM2-class models live behind the catalog's HF Hub search, not the
# wizard. That's why tui-catalog (not this tape) shows the SmolLM2 pull.
Env LILBEE_DATA "/tmp/lilbee-demo/tui-setup"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/setup-models"
Env LILBEE_SKIP_TOML_CONFIG "1"
Env LILBEE_NO_SPLASH "1"
# Point Ollama at an unused port so the wizard can't detect locally
# installed Ollama models. Without this, picking Qwen3 0.6B is a no-op
# (the model is "already available via Ollama") and the Task Center
# only shows the embedding download instead of two parallel pulls.
Env OLLAMA_HOST "http://127.0.0.1:65535"

Type "lilbee"
Sleep 300ms
Enter
Sleep 2500ms

# Wizard opens with the first chat card (Qwen3 0.6B) auto-focused. Pick it.
Enter
Sleep 800ms

# Tab into the Embedding row and pick Nomic v1.5 (first card).
Tab
Sleep 300ms
Enter
Sleep 800ms

# Jump to Task Center; show the two parallel downloads kick off, then
# elide the slow middle (Qwen3 0.6B is ~4min on a slow connection),
# then show the final state with both at 100%.
Type "t"
Sleep 6s

Hide
Sleep 240s
Show
Sleep 4s

Screenshot demos/_out/tui-setup.png

# Back to the wizard so the user sees the two cards now marked installed.
Type "q"
Sleep 1500ms

# Dismiss into Chat.
Escape
Sleep 2000ms

Type "/quit"
Enter
Sleep 500ms
