Source demos/_shared.tape
Output demos/_out/cli.gif

# CLI tour: what the Ollama CLI is good for. init, pull, list, add, status,
# sync. No JSON ask/stream, no self-check (the latter is a 4-6 min sanity
# check that doesn't make a good demo).
#
# Measured (uv-tool lilbee 0.6.66b469, models cached):
#   lilbee init                        0.62s -> Sleep 2.5s
#   lilbee model pull qwen (cached)    0.73s -> Sleep 2.5s
#   lilbee model pull nomic (cached)   0.75s -> Sleep 2.5s
#   lilbee model list                  0.75s -> Sleep 4s   (give viewer the table)
#   lilbee add cv-manual.pdf (cold)   20.61s -> Sleep 26s  (real embed)
#   lilbee add man-pages/              4.75s -> Sleep 7s
#   lilbee status                      1.34s -> Sleep 4s   (read time)
#   lilbee sync (no-op)                1.39s -> Sleep 3s
Env LILBEE_DATA "/tmp/lilbee-demo/cli"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"

Type "cd /tmp/lilbee-demo/cli"
Sleep 300ms
Enter
Sleep 600ms

Type "lilbee init"
Sleep 400ms
Enter
Sleep 2500ms

Type "lilbee model pull Qwen/Qwen3-0.6B-GGUF"
Sleep 400ms
Enter
Sleep 2500ms

Type "lilbee model pull nomic-ai/nomic-embed-text-v1.5-GGUF"
Sleep 400ms
Enter
Sleep 2500ms

Type "lilbee model list"
Sleep 400ms
Enter
Sleep 4s

Screenshot demos/_out/cli.png

Type "lilbee add ~/Downloads/cv-manual.pdf"
Sleep 400ms
Enter
Sleep 26s

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
