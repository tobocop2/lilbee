# VHS tape template for a giant-tier opencode demo (full TUI).
# Substituted per giant: demos/opencode-GLM-4.6 (output stem), Walk me through lilbee's retrieval architecture end to end and explain how query expansion and reranking improve result quality, with file citations. Then suggest where I'd add a caching layer., 1200s (the
# measured giant generation time, so Hide/Show skips the dead air cleanly).
# Rose-pine + dimensions mirror the existing lilbee demo reel (_shared.tape).

Set Shell "bash"
Set Width 1600
Set Height 1000
Set FontSize 18
Set Padding 30
Set TypingSpeed 30ms
Set Framerate 60
Set PlaybackSpeed 1.5
Set Theme "rose-pine-moon"
Set WindowBar Colorful

Output "demos/opencode-GLM-4.6.gif"
Output "demos/opencode-GLM-4.6.webm"

# opencode reads ./opencode.json (provider + MCP + tools). -m forces our model so
# `opencode` doesn't fall back to its built-in `build` agent / default model.
Type "cd /root/demo-proj && opencode -m lilbee/GLM-4.6"
Sleep 500ms
Enter
Sleep 12s

# The agentic RAG turn: the giant searches lilbee's own source and answers with citations.
Type "Walk me through lilbee's retrieval architecture end to end and explain how query expansion and reranking improve result quality, with file citations. Then suggest where I'd add a caching layer."
Sleep 800ms
Enter
Sleep 6s
# Skip the giant's (slow) generation; resume once the answer has rendered.
Hide
Sleep 1200s
Show
Sleep 4s
PageUp
Sleep 2500ms
PageUp
Sleep 2500ms
PageDown
Sleep 1800ms
PageDown
Sleep 1800ms

Screenshot "demos/opencode-GLM-4.6.png"
Ctrl+C
