# VHS tape template for a giant-tier opencode demo (full TUI).
# Substituted per giant: demos/opencode-Qwen3-Coder-30B-A3B-Instruct (output stem), Search lilbee for how it chunks source code, then explain in two or three sentences how code chunking differs from prose chunking, citing the file., 150s (the
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

Output "demos/opencode-Qwen3-Coder-30B-A3B-Instruct.gif"
Output "demos/opencode-Qwen3-Coder-30B-A3B-Instruct.webm"

# opencode reads ./opencode.json (provider + MCP + tools). -m forces our model so
# `opencode` doesn't fall back to its built-in `build` agent / default model.
Type "cd /root/demo-proj && opencode -m lilbee/Qwen3-Coder-30B-A3B-Instruct"
Sleep 500ms
Enter
Sleep 12s

# The agentic RAG turn: the giant searches lilbee's own source and answers with citations.
Type "Search lilbee for how it chunks source code, then explain in two or three sentences how code chunking differs from prose chunking, citing the file."
Sleep 800ms
Enter
Sleep 6s
# Skip the giant's (slow) generation; resume once the answer has rendered.
Hide
Sleep 150s
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

Screenshot "demos/opencode-Qwen3-Coder-30B-A3B-Instruct.png"
Ctrl+C
