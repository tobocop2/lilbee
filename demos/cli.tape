Source demos/_shared.tape
Output demos/_out/cli.gif

# CLI tour: what the Ollama CLI is good for. init, pull, list, add, status,
# sync. No JSON ask/stream; no self-check (it's a 4-6 min sanity probe).
#
# This tape pulls SmolLM2 135M COLD (not cached) so the model-pull line shows
# a real download progress bar (~30s, 101 MB). The CLI's own LILBEE_MODELS_DIR
# is isolated from the TUI's, hence `cli-models` in the path.
#
# Measured (uv-tool lilbee 0.6.66b469):
#   lilbee init                                   0.6s   -> Sleep 2.5s
#   lilbee model pull SmolLM2 135M (cold)        ~30s    -> Sleep 32s
#   lilbee model pull nomic (cached)              0.8s   -> Sleep 2.5s
#   lilbee model list                             0.8s   -> Sleep 4s
#   lilbee add cv-manual.pdf (cold)              ~26s    -> Sleep 28s
#   lilbee add man-pages/                         5s     -> Sleep 7s
#   lilbee status                                 1.3s   -> Sleep 4s
#   lilbee sync (no-op)                           1.4s   -> Sleep 3s
#   lilbee search "..."                           3s     -> Sleep 4s
Env LILBEE_DATA "/tmp/lilbee-demo/cli"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/cli-models"
Env LILBEE_SKIP_TOML_CONFIG "1"
# Rich's Live display captures as stacked lines in VHS recordings; this
# escape hatch falls back to a plain \r-overwrite progress that records
# cleanly. Regular users still get the Rich progress.
Env LILBEE_PLAIN_PROGRESS "1"

Type "cd /tmp/lilbee-demo/cli"
Sleep 300ms
Enter
Sleep 600ms

Type "lilbee init"
Sleep 400ms
Enter
Sleep 2500ms

# Real cold pull -- gives the demo a visible progress bar.
Type "lilbee model pull bartowski/SmolLM2-135M-Instruct-GGUF"
Sleep 400ms
Enter
Sleep 32s

Type "lilbee model pull nomic-ai/nomic-embed-text-v1.5-GGUF"
Sleep 400ms
Enter
Sleep 2500ms

Type "lilbee model list"
Sleep 400ms
Enter
Sleep 4s

Screenshot demos/_out/cli.png

Type "lilbee add ~/projects/lilbee-demos/demos/sample-corpus/cv-manual.pdf"
Sleep 400ms
Enter
Sleep 28s

Type "lilbee add /tmp/lilbee-demo/cli/man-pages/"
Sleep 400ms
Enter
Sleep 7s

Type "lilbee status"
Sleep 400ms
Enter
Sleep 4s

Type "lilbee sync"
Sleep 400ms
Enter
Sleep 3s

Type "lilbee search 'recursive find by name'"
Sleep 400ms
Enter
Sleep 4s
