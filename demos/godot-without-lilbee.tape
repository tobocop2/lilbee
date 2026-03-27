Output demos/godot-without-lilbee.gif
Set Shell zsh
Set FontSize 14
Set Width 1200
Set Height 700
Set Padding 20
Set TypingSpeed 40ms
Set PlaybackSpeed 1

# Launch opencode TUI — no lilbee MCP
Type "cd demos/godot-without-lilbee && opencode"
Sleep 500ms
Enter
Sleep 3s

# Same prompt, no lilbee search available
Type "make a procedural level generator that places wall and floor tiles and scatters collectibles using pathfinding. write everything in a single file and save it as level_generator.gd"
Sleep 500ms
Enter
Sleep 180s

# Exit opencode
Ctrl+C
Sleep 1s

# Show the result
Type "cat level_generator.gd"
Sleep 500ms
Enter
Sleep 5s
