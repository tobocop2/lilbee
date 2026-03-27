Output demos/vision-ocr.gif
Set Shell zsh
Set FontSize 14
Set Width 1200
Set Height 600
Set Padding 20
Set TypingSpeed 50ms
Set PlaybackSpeed 1

# Show lilbee status — proves the manual is indexed via vision OCR
Type "lilbee status"
Sleep 500ms
Enter
Sleep 3s

# Start interactive chat (uses qwen3:8b by default)
Type "lilbee"
Sleep 500ms
Enter
Sleep 3s

# Question 1: credits
Type "who was on the dev team for x-wing?"
Sleep 500ms
Enter
Sleep 35s

# Question 2: energy management
Type "how do I transfer laser energy to shields in star wars x-wing collectors edition?"
Sleep 500ms
Enter
Sleep 30s

# Question 3: starfighter speeds
Type "which starfighter has the top speed in star wars x-wing collectors edition?"
Sleep 500ms
Enter
Sleep 25s

# Exit
Type "/quit"
Sleep 500ms
Enter
Sleep 1s
