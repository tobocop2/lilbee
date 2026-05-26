# VHS tape template for a giant-tier opencode demo (full TUI).
# Substituted per giant: demos/opencode-llama3 (output stem), Add support for a new model family to lilbee whose tool calls look like <tool_call name=search>{json args}</tool_call>. Implement the family profile and the response-parser schema following the existing families, and wire it into the parser., 90s (the
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

Output "demos/opencode-llama3.gif"
Output "demos/opencode-llama3.webm"

# opencode reads ./opencode.json (model -> the giant on llama-server, MCP -> lilbee)
Type "cd /root/demo-proj && opencode"
Sleep 500ms
Enter
Sleep 12s

# The agentic RAG turn: the giant searches lilbee's own source and answers with citations.
Type "Add support for a new model family to lilbee whose tool calls look like <tool_call name=search>{json args}</tool_call>. Implement the family profile and the response-parser schema following the existing families, and wire it into the parser."
Sleep 800ms
Enter
Sleep 6s
# Skip the giant's (slow) generation; resume once the answer has rendered.
Hide
Sleep 90s
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

Screenshot "demos/opencode-llama3.png"
Ctrl+C
