Source demos/_shared.tape
Output demos/_out/opencode-code.gif

# opencode driving lilbee over MCP, on a fresh copy of lilbee's own src/.
# Caption (when this lands in docs/demos.md): opencode runs on a cloud model
# (MiniMax) here; the lilbee corpus lives locally on this machine.
#
# The demo dir was prepared by demos/_prep.sh: shallow clone of lilbee at
# /tmp/lilbee-demo/opencode-code, with AGENTS.md, opencode.json, the
# lilbee-worker subagent, and the lilbee-mcp skill already dropped in.
Env LILBEE_MODELS_DIR "/tmp/lilbee-demo/models"
Env LILBEE_SKIP_TOML_CONFIG "1"

Type "cd /tmp/lilbee-demo/opencode-code"
Sleep 300ms
Enter
Sleep 400ms

Type "opencode"
Sleep 400ms
Enter
Sleep 5s

# 1) Ask opencode to index lilbee's core source. It should delegate to
# `lilbee-worker`, which runs lilbee_init + lilbee_add(["src/lilbee"]) and
# reports back. tests/ are not in this corpus.
Type "Index this project's core source under src/lilbee so you can answer questions about how lilbee works. Don't index tests."
Sleep 400ms
Enter
Sleep 120s

Screenshot demos/_out/opencode-code.png

# 2) Walk through the ingest pipeline. The agent should search via lilbee_search
# and cite src/lilbee/data/ingest/pipeline.py with line numbers.
Type "Walk me through what lilbee does when I run `add` on a folder. What's the ingest pipeline? Cite the code."
Sleep 400ms
Enter
Sleep 60s

# 3) How does it discover .lilbee/?
Type "How does lilbee discover a project-local .lilbee/ directory? Cite the file and line."
Sleep 400ms
Enter
Sleep 45s

# 4) How is lilbee itself exposed as MCP?
Type "How is lilbee itself exposed as an MCP server, and what tools does it provide? Point me at the file."
Sleep 400ms
Enter
Sleep 45s

# Exit.
Ctrl+C
Sleep 1s
Type "exit"
Sleep 300ms
Enter
Sleep 1s
