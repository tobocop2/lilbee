Source demos/_shared.tape
Output demos/_out/opencode-code.gif

# opencode driving lilbee over MCP, on a fresh copy of lilbee's own src/.
# Caption: opencode runs on a cloud model (MiniMax) here; the lilbee corpus
# lives locally on this machine.
#
# Demo dir is prepared by demos/_prep.sh: shallow clone of lilbee with src/
# at the root, AGENTS.md, opencode.json, the lilbee-worker subagent, and the
# lilbee-mcp skill already dropped in.
#
# Timings here are estimates (cloud-model + MCP tool latency wasn't directly
# measured); expect to iterate. The Sleeps assume MiniMax with normal load.
#   indexing src/lilbee via lilbee_add (cold embed, ~hundreds of files)  ~120-180s
#   per Q&A round (search + cloud answer)                                  ~30-60s
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"

Type "cd /tmp/lilbee-demo/opencode-code"
Sleep 400ms
Enter
Sleep 800ms

Type "opencode"
Sleep 500ms
Enter
Sleep 6s

# 1) Ask opencode to index lilbee's core source. It delegates to lilbee-worker,
# which runs lilbee_init + lilbee_add(["src/lilbee"]); tests/ are not in the corpus.
Type "Index this project's core source under src/lilbee so you can answer questions about how lilbee works. Don't index tests."
Sleep 500ms
Enter
Sleep 150s

Screenshot demos/_out/opencode-code.png

# 2) How does the ingest pipeline work?
Type "Walk me through what lilbee does when I run `add` on a folder. What's the ingest pipeline? Cite the code."
Sleep 500ms
Enter
Sleep 60s

# 3) Project-local .lilbee/ discovery.
Type "How does lilbee discover a project-local .lilbee/ directory? Cite the file and line."
Sleep 500ms
Enter
Sleep 45s

# 4) The MCP server itself.
Type "How is lilbee itself exposed as an MCP server, and what tools does it provide? Point me at the file."
Sleep 500ms
Enter
Sleep 45s

Ctrl+C
Sleep 1s
Type "exit"
Sleep 400ms
Enter
Sleep 1s
