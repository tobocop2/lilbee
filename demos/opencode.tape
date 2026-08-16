Output demos/opencode.gif
Set Shell zsh
Set FontSize 14
Set Width 1200
Set Height 600
Set Padding 20
Set TypingSpeed 50ms

# Launch opencode
Type "opencode"
Sleep 500ms
Enter
Sleep 3s

# Ask a code question, pointing opencode to lilbee for retrieval
Type "I have lilbee indexed as a local knowledge base. Use lilbee --json search to look up how the chunking pipeline works, then explain how code files are chunked differently from plain text."
Sleep 500ms
Enter
Sleep 30s

# Exit opencode
Ctrl+C
Sleep 1s
