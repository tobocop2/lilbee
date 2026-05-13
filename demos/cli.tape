Source demos/_shared.tape
Output demos/_out/cli.gif

# CLI tour: what the Ollama CLI is good for. init, pull, list, add, status,
# sync, rm, self-check. No JSON ask/stream.
Env LILBEE_DATA "/tmp/lilbee-demo/cli"
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"

# Land in the project dir for `lilbee init` to make sense.
Type "cd /tmp/lilbee-demo/cli"
Sleep 300ms
Enter
Sleep 600ms

Type "lilbee init"
Sleep 400ms
Enter
Sleep 2s

Type "lilbee model pull Qwen/Qwen3-0.6B-GGUF"
Sleep 400ms
Enter
Sleep 6s

Type "lilbee model pull nomic-ai/nomic-embed-text-v1.5-GGUF"
Sleep 400ms
Enter
Sleep 4s

Type "lilbee model list"
Sleep 400ms
Enter
Sleep 3s

Screenshot demos/_out/cli.png

Type "lilbee add ~/Downloads/cv-manual.pdf"
Sleep 400ms
Enter
Sleep 18s

Type "lilbee add /tmp/lilbee-demo/cli/man-pages/"
Sleep 400ms
Enter
Sleep 15s

Type "lilbee status"
Sleep 400ms
Enter
Sleep 3s

Type "lilbee sync"
Sleep 400ms
Enter
Sleep 3s

Type "lilbee self-check --skip-embedding"
Sleep 400ms
Enter
Sleep 10s
